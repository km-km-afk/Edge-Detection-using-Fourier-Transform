"""
FOURIER EDGE DETECTION — SUMMARY

Edges = rapid intensity changes → high-frequency components.

1. Convert image to frequency domain:
   F(u,v) = FFT{f(x,y)}

2. Apply high-pass filter:
   G(u,v) = H(u,v) * F(u,v)

3. Reconstruct image:
   g(x,y) = IFFT{G(u,v)}

Filters:
- Ideal: sharp cutoff → ringing artifacts
- Gaussian: smooth → natural edges
- Butterworth: tunable trade-off

Key idea:
Edge detection = extracting high frequencies.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import cv2
import os


OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILTER_RADIUS = 30   #D0-cutoff frequency (pixels from centre)
GAUSS_SIGMA = 30     #σ-Gaussian HPF bandwidth
BUTTER_ORDER = 2     #n-Butterworth order

def load_or_create_image(path: str = None) -> np.ndarray:
    if path and os.path.isfile(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        print(f"[INFO] Loaded image: {path}  shape={img.shape}")
        return img.astype(np.float32)

    print("[INFO] No image found — using synthetic test image.")
    H, W = 256, 256
    img  = np.zeros((H, W), dtype=np.float32)

    #filled rectangle
    img[40:120, 40:120] = 200
    #hollow rectangle
    img[140:220, 140:220] = 160
    img[150:210, 150:210] = 0
    #circle
    cy, cx, r = 180, 70, 30
    Y, X = np.ogrid[:H, :W]
    img[(X - cx)**2 + (Y - cy)**2 <= r**2] = 220
    #diagonal edge
    for i in range(H):
        img[i, max(0, i - 10): min(W, i + 10)] += 80
    img = np.clip(img, 0, 255)
    return img


def compute_fft(img: np.ndarray):
    #Return (shifted spectrum, magnitude spectrum for display)
    f      = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)          #move DC to centre
    mag    = np.abs(fshift) + 1          #+1 avoids log(0)
    return fshift, mag


def distance_grid(rows: int, cols: int) -> np.ndarray:
    #Euclidean distance of each frequency component from the DC centre.
    u = np.arange(rows) - rows // 2
    v = np.arange(cols) - cols // 2
    V, U = np.meshgrid(v, u)
    return np.sqrt(U**2 + V**2)


def ideal_hpf(rows: int, cols: int, D0: float) -> np.ndarray:
    """
    Ideal High-Pass Filter.
        H(u,v) = 0  if D(u,v) ≤ D0
                 1  if D(u,v) > D0
    """
    D = distance_grid(rows, cols)
    return (D > D0).astype(np.float32)


def gaussian_hpf(rows: int, cols: int, sigma: float) -> np.ndarray:
    """
    Gaussian High-Pass Filter.
        H(u,v) = 1 - exp(-D² / 2σ²)
    """
    D = distance_grid(rows, cols)
    return (1 - np.exp(-(D**2) / (2 * sigma**2))).astype(np.float32)


def butterworth_hpf(rows: int, cols: int, D0: float, n: int) -> np.ndarray:
    """
    Butterworth High-Pass Filter.
        H(u,v) = 1 / (1 + (D0/D)^2n)
    """
    D = distance_grid(rows, cols)
    D[D == 0] = 1e-6            # avoid division by zero at DC
    return (1 / (1 + (D0 / D)**(2 * n))).astype(np.float32)


def apply_frequency_filter(fshift: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Multiply spectrum by filter mask, then inverse-FFT back.
    """
    filtered   = fshift * H
    f_ishift   = np.fft.ifftshift(filtered)
    img_back   = np.fft.ifft2(f_ishift)
    return np.abs(img_back).astype(np.float32)


def norm(arr: np.ndarray) -> np.ndarray:
    #Normalise array to [0, 255] uint8 for display / saving.
    a = arr - arr.min()
    if a.max() > 0:
        a = a / a.max() * 255
    return a.astype(np.uint8)


def plot_full_pipeline(img, fshift, mag, filters, results):
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor('#0f0f0f')

    # Better layout: 2 rows only
    gs = gridspec.GridSpec(2, 5, figure=fig,
                           hspace=0.35, wspace=0.25)

    cmap_img  = 'gray'
    cmap_edge = 'hot'

    def ax(row, col, title, data, cmap=cmap_img):
        a = fig.add_subplot(gs[row, col])
        a.imshow(data, cmap=cmap)
        a.set_title(title, color='white', fontsize=10, pad=6)
        a.axis('off')
        return a

    # --- Precompute (avoid repeated work) ---
    ideal_edge  = norm(results['ideal'])
    gauss_edge  = norm(results['gaussian'])
    butter_edge = norm(results['butter'])

    ideal_spec  = np.log1p(np.abs(fshift * filters['ideal']))
    gauss_spec  = np.log1p(np.abs(fshift * filters['gaussian']))
    butter_spec = np.log1p(np.abs(fshift * filters['butter']))

    # --- Row 1: Input + Filters ---
    ax(0, 0, "Original Image", img, cmap_img)
    ax(0, 1, "FFT Magnitude", np.log1p(mag), 'inferno')
    ax(0, 2, "Ideal HPF Mask", filters['ideal'], 'Blues')
    ax(0, 3, "Gaussian HPF Mask", filters['gaussian'], 'Blues')
    ax(0, 4, "Butterworth HPF Mask", filters['butter'], 'Blues')

    # --- Row 2: Results ---
    ax(1, 0, "Ideal HPF Output", ideal_edge, cmap_edge)
    ax(1, 1, "Gaussian HPF Output", gauss_edge, cmap_edge)
    ax(1, 2, "Butterworth HPF Output", butter_edge, cmap_edge)
    ax(1, 3, "Ideal Filtered Spectrum", ideal_spec, 'inferno')
    ax(1, 4, "Gaussian Filtered Spectrum", gauss_spec, 'inferno')

    # --- Better annotation (clean + relevant) ---
    fig.text(0.72, 0.02,
             f"Cutoff D₀ = {FILTER_RADIUS}px   |   σ = {GAUSS_SIGMA}px   |   n = {BUTTER_ORDER}",
             color='#cccccc', fontsize=10, ha='center')

    fig.suptitle(
        "Fourier High-Pass Filtering for Edge Detection",
        color='white', fontsize=15, y=0.97, fontweight='bold'
    )

    out = os.path.join(OUTPUT_DIR, "full_pipeline.png")
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())

    print(f"[SAVED] {out}")
    plt.show()




def save_results(img, results, filters):
    cv2.imwrite(os.path.join(OUTPUT_DIR, "00_original.png"),
                norm(img))
    for name, arr in results.items():
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"fourier_{name}.png"),
                    norm(arr))
    for name, mask in filters.items():
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"mask_{name}.png"),
                    norm(mask))
    print(f"[SAVED] individual images → {OUTPUT_DIR}/")


def print_metrics(results: dict):
    print("\n" + "="*52)
    print("  EDGE DENSITY COMPARISON  (% non-zero pixels)")
    print("="*52)

    all_res = {f"Fourier-{k}": v for k, v in results.items()}

    for name, arr in all_res.items():
        binary    = norm(arr) > 30
        density   = 100 * binary.sum() / binary.size
        mean_resp = arr.mean()
        print(f"  {name:<25}  density={density:5.2f}%   mean={mean_resp:7.2f}")
    print("="*52 + "\n")


def main():
    img = load_or_create_image("image.jpg")

    rows, cols = img.shape
    fshift, mag = compute_fft(img)
    filters = {
        'ideal'   : ideal_hpf(rows, cols, FILTER_RADIUS),
        'gaussian': gaussian_hpf(rows, cols, GAUSS_SIGMA),
        'butter'  : butterworth_hpf(rows, cols, FILTER_RADIUS, BUTTER_ORDER),
    }

    results = {k: apply_frequency_filter(fshift, H)
               for k, H in filters.items()}

    save_results(img, results, filters)
    print_metrics(results)
    plot_full_pipeline(img, fshift, mag, filters, results)

    print("\n[DONE] All outputs saved to:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
