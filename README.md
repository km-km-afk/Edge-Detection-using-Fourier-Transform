# Fourier-Based Edge Detection by Krisha Mehta

This project demonstrates **High-Pass Filtering (HPF)** in the frequency domain using the Fast Fourier Transform (FFT). By suppressing low-frequency components (smooth regions) and preserving high-frequency components (rapid intensity changes), we can effectively extract edges from an image.

## 🔬 Theory & Methodology

Edges correspond to rapid changes in pixel intensity. In the Fourier domain, these changes are represented by high frequencies located away from the origin (DC component).

The process follows the standard filtering pipeline:
1. **Transform**: Convert the spatial image $f(x,y)$ to its frequency representation $F(u,v)$ using FFT.
2. **Shift**: Move the zero-frequency component to the center of the spectrum.
3. **Filter**: Apply a transfer function $H(u,v)$ such that $G(u,v) = H(u,v) \cdot F(u,v)$.
4. **Inverse**: Perform the Inverse FFT to return to the spatial domain.

### Implemented Filters

| Filter | Characteristics |
| :--- | :--- |
| **Ideal HPF** | Completely cuts off frequencies below $D_0$. High "ringing" artifacts (Gibbs phenomenon). |
| **Gaussian HPF** | Smooth transition defined by $\sigma$. No ringing artifacts, resulting in clean edges. |
| **Butterworth HPF** | A compromise between Ideal and Gaussian. The order $n$ controls the "steepness" of the cutoff. |



---

## 🛠️ Implementation Details

The script generates a synthetic test image (if no input is provided) containing geometric shapes and diagonal lines to test edge response across various orientations.

### Key Parameters
* `FILTER_RADIUS (D0)`: The cutoff frequency in pixels.
* `GAUSS_SIGMA`: Controls the spread for the Gaussian filter.
* `BUTTER_ORDER (n)`: Controls the decay rate of the Butterworth filter.

### Dependencies
* `numpy`
* `opencv-python`
* `matplotlib`

---

## 📊 Results & Visualization

<img width="1986" height="1249" alt="ex" src="https://github.com/user-attachments/assets/c9dbbf1d-5e45-422a-bba4-cea884af83d4" />

The output includes a comprehensive pipeline visualization:
* **Top Row**: The original image, the magnitude spectrum, and the 2D filter masks.
* **Bottom Row**: The resulting spatial edges and the filtered frequency spectra.



### Edge Density Comparison
The script calculates the percentage of "edge pixels" detected. High-pass filtering often captures more texture and noise than spatial operators like Canny, resulting in higher density metrics.

---

## 🚀 How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/fourier-edge-detection.git
   cd fourier-edge-detection
   ```
2. **Install dependencies**:
   ```bash
   pip install numpy opencv-python matplotlib
   ```
3. **Run the script**:
   ```bash
   python main.py
   ```
4. **Check Outputs**: Results are saved in the `/results` directory, including the full pipeline plot and individual masks.

---

## 📁 Project Structure
```text
├── main.py            # Core logic and filtering implementation
├── input.jpg          # (Optional) Place your input image here
└── results/           # Generated output plots and masks
    ├── full_pipeline.png
    ├── fourier_ideal.png
    └── ...
```
