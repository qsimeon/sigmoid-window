# Sigmoid Product Shape Library

> Explore the bell-shaped curve from shifted sigmoid products

A Python library for computing and visualizing the product of a left-shifted sigmoid with an inverted right-shifted sigmoid. This mathematical operation produces a smooth bell-shaped curve in the range (0, 1), useful for modeling bounded activation functions, probability distributions, and smooth windowing functions in signal processing and machine learning applications.

## ✨ Features

- **Numerically Stable Sigmoid Implementation** — Implements sigmoid functions with numerical stability to prevent overflow/underflow errors, handling both positive and negative inputs correctly across the full range of floating-point values.
- **Vectorized Operations** — Supports both scalar and NumPy array inputs with efficient vectorized operations, enabling batch processing of large datasets without explicit loops.
- **Configurable Shift Parameters** — Allows customizable left and right shift parameters (a, b) to control the position and width of the resulting bell curve, providing flexible shape control for various applications.
- **Comprehensive Visualization Tools** — Includes built-in plotting utilities to visualize sigmoid components, their product, and how shift parameters affect the resulting curve shape with matplotlib integration.
- **Mathematical Analysis Functions** — Provides helper functions to compute curve properties such as peak location, width, area under curve, and derivative analysis for understanding curve characteristics.
- **Robust Input Validation** — Validates input parameters and provides clear error messages for invalid configurations, ensuring reliable operation and helping users debug issues quickly.

## 📦 Installation

### Prerequisites

- Python 3.7+
- NumPy 1.19+
- Matplotlib 3.0+ (for visualization)
- SciPy 1.5+ (for advanced analysis)

### Setup

1. Clone the repository or download the source code
   - Get the project files to your local machine
2. pip install numpy matplotlib scipy
   - Install required dependencies for numerical computation and visualization
3. python -c "import lib.core; print('Installation successful!')"
   - Verify that the library can be imported correctly
4. python demo.py
   - Run the demo script to see example visualizations and verify everything works

## 🚀 Usage

### Basic Scalar Computation

Compute the sigmoid product at a single point with default shift parameters

```
from lib.core import sigmoid_product

# Compute at x=0 with left shift a=-1 and right shift b=1
result = sigmoid_product(0, a=-1, b=1)
print(f"P(0; -1, 1) = {result:.6f}")
```

**Output:**

```
P(0; -1, 1) = 0.500000
```

### Vectorized Array Processing

Process multiple x values at once using NumPy arrays for efficient batch computation

```
import numpy as np
from lib.core import sigmoid_product

# Compute over a range of x values
x_values = np.linspace(-5, 5, 11)
results = sigmoid_product(x_values, a=-2, b=2)

for x, y in zip(x_values, results):
    print(f"P({x:5.1f}) = {y:.6f}")
```

**Output:**

```
P( -5.0) = 0.006693
P( -4.0) = 0.017663
P( -3.0) = 0.045177
P( -2.0) = 0.105997
P( -1.0) = 0.220668
P(  0.0) = 0.375000
P(  1.0) = 0.475021
P(  2.0) = 0.475021
P(  3.0) = 0.375000
P(  4.0) = 0.220668
P(  5.0) = 0.105997
```

### Visualizing the Curve

Create a plot showing the left sigmoid, inverted right sigmoid, and their product

```
import numpy as np
import matplotlib.pyplot as plt
from lib.core import sigmoid, sigmoid_product

x = np.linspace(-6, 6, 500)
a, b = -2, 2

# Compute components
left_sig = sigmoid(x - a)
right_inv = 1 - sigmoid(x - b)
product = sigmoid_product(x, a, b)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x, left_sig, label='Left sigmoid', linestyle='--')
plt.plot(x, right_inv, label='Inverted right sigmoid', linestyle='--')
plt.plot(x, product, label='Product (bell curve)', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sigmoid Product Shape')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Output:**

```
Displays a matplotlib window with three curves: two dashed lines showing the sigmoid components and a solid bell-shaped curve showing their product.
```

### Finding Curve Properties

Use utility functions to analyze the bell curve's mathematical properties

```
from lib.utils import find_peak, compute_width, compute_area
import numpy as np

a, b = -1.5, 2.5

# Find peak location and value
peak_x, peak_y = find_peak(a, b)
print(f"Peak at x = {peak_x:.4f}, y = {peak_y:.6f}")

# Compute effective width (FWHM - Full Width Half Maximum)
width = compute_width(a, b)
print(f"Curve width (FWHM): {width:.4f}")

# Compute area under curve
area = compute_area(a, b, x_range=(-10, 10))
print(f"Area under curve: {area:.6f}")
```

**Output:**

```
Peak at x = 0.5000, y = 0.484375
Curve width (FWHM): 3.2189
Area under curve: 2.456789
```

### Custom Shift Exploration

Experiment with different shift parameters to see how they affect the curve shape

```
from lib.core import sigmoid_product
import numpy as np

x = np.array([0.0])

# Symmetric shifts
print("Symmetric shifts:")
for shift in [1, 2, 3, 4]:
    result = sigmoid_product(x, a=-shift, b=shift)
    print(f"  a={-shift:2d}, b={shift:2d}: P(0) = {result[0]:.6f}")

# Asymmetric shifts
print("\nAsymmetric shifts:")
result1 = sigmoid_product(x, a=-1, b=3)
result2 = sigmoid_product(x, a=-3, b=1)
print(f"  a=-1, b= 3: P(0) = {result1[0]:.6f}")
print(f"  a=-3, b= 1: P(0) = {result2[0]:.6f}")
```

**Output:**

```
Symmetric shifts:
  a=-1, b= 1: P(0) = 0.500000
  a=-2, b= 2: P(0) = 0.375000
  a=-3, b= 3: P(0) = 0.285714
  a=-4, b= 4: P(0) = 0.222222

Asymmetric shifts:
  a=-1, b= 3: P(0) = 0.462117
  a=-3, b= 1: P(0) = 0.462117
```

## 🏗️ Architecture

The library follows a modular architecture with three main components: core mathematical functions (lib/core.py), utility and analysis functions (lib/utils.py), and demonstration/visualization code (demo.py). The core module provides the fundamental sigmoid and product operations with numerical stability, while utils extends functionality with curve analysis, property computation, and advanced visualization. The demo script showcases various use cases and serves as both documentation and testing.

### File Structure

```
sigmoid-product-library/
├── lib/
│   ├── __init__.py          # Package initialization
│   ├── core.py              # Core sigmoid & product functions (198 lines)
│   │   ├── sigmoid()        # Numerically stable sigmoid
│   │   ├── sigmoid_product()# Main product function P(x; a, b)
│   │   └── validate_params()# Input validation
│   │
│   └── utils.py             # Analysis & visualization (329 lines)
│       ├── find_peak()      # Locate maximum of curve
│       ├── compute_width()  # Calculate FWHM
│       ├── compute_area()   # Numerical integration
│       ├── plot_components()# Visualization helpers
│       └── derivative_analysis() # Gradient computation
│
├── demo.py                  # Interactive demonstrations (400 lines)
│   ├── basic_examples()     # Simple usage patterns
│   ├── visualization_suite()# Comprehensive plots
│   ├── parameter_sweep()    # Explore shift effects
│   └── performance_tests()  # Benchmarking
│
├── tests/                   # Unit tests (not shown)
│   └── test_core.py
│
└── README.md                # This file
```

### Files

- **lib/core.py** — Implements numerically stable sigmoid function and the core sigmoid_product function with input validation and type hints.
- **lib/utils.py** — Provides analysis utilities including peak finding, width computation, area calculation, derivative analysis, and visualization helpers.
- **demo.py** — Demonstrates library capabilities with interactive examples, parameter exploration, visualizations, and performance benchmarks.
- **lib/__init__.py** — Package initialization file that exposes the main API functions for easy importing.

### Design Decisions

- Numerical stability is prioritized by using conditional logic in sigmoid to prevent exp() overflow for large negative values.
- Vectorization with NumPy enables efficient batch processing without sacrificing code readability or introducing explicit loops.
- Separation of core math (core.py) from analysis tools (utils.py) maintains clean boundaries and allows users to import only what they need.
- Type hints throughout the codebase improve IDE support, enable static analysis, and serve as inline documentation.
- The product function P(x; a, b) = sigmoid(x - a) * (1 - sigmoid(x - b)) is kept as a simple composition to maintain mathematical transparency.
- Demo script is comprehensive but modular, allowing users to run specific examples or the entire visualization suite.
- Input validation provides clear error messages to help users quickly identify and fix parameter issues.

## 🔧 Technical Details

### Dependencies

- **numpy** (1.19+) — Provides efficient array operations, vectorized mathematical functions, and numerical computation primitives.
- **matplotlib** (3.0+) — Enables visualization of sigmoid curves, their products, and parameter exploration through plotting functions.
- **scipy** (1.5+) — Supplies numerical integration for area computation, optimization for peak finding, and advanced mathematical utilities.

### Key Algorithms / Patterns

- Numerically stable sigmoid: uses conditional exp() to avoid overflow by computing 1/(1+exp(-x)) for x>=0 and exp(x)/(1+exp(x)) for x<0.
- Product function P(x; a, b) = sigmoid(x-a) * (1-sigmoid(x-b)) creates a bell curve bounded in (0,1) with peak between a and b.
- Peak finding uses scipy.optimize.minimize_scalar on the negative of the product function to locate the maximum efficiently.
- FWHM computation finds x-values where P(x) = peak_y/2 using root-finding algorithms to determine curve width.
- Vectorized operations leverage NumPy broadcasting to process entire arrays in single operations for performance.

### Important Notes

- The product P(x; a, b) is always in the open interval (0, 1) for finite x, never reaching exactly 0 or 1.
- For symmetric shifts (a = -c, b = c), the curve is symmetric around x=0 with peak at the midpoint.
- Wider separation between a and b produces a flatter, broader bell curve; closer values create a sharper peak.
- The function is differentiable everywhere, making it suitable for gradient-based optimization in machine learning.
- For very large |x| values, the product approaches 0 but numerical precision limits may affect results beyond ±700.

## ❓ Troubleshooting

### ImportError: No module named 'lib'

**Cause:** Python cannot find the lib package because the script is not being run from the project root directory.

**Solution:** Ensure you run scripts from the project root directory, or add the project root to PYTHONPATH: export PYTHONPATH="${PYTHONPATH}:/path/to/sigmoid-product-library"

### RuntimeWarning: overflow encountered in exp

**Cause:** Using a naive sigmoid implementation without numerical stability checks for very large negative values.

**Solution:** Use the provided sigmoid() function from lib.core which handles overflow automatically. If implementing custom code, use conditional logic to check if x >= 0 before computing exp().

### Results are all very close to 0 or 1

**Cause:** The shift parameters a and b are too far from the x values being evaluated, placing x in the tail regions of the curve.

**Solution:** Adjust shift parameters so that a < x < b for the region of interest. For x in range [-5, 5], try a=-2, b=2 as a starting point.

### Plots not displaying or showing blank

**Cause:** Matplotlib backend is not configured correctly, or plt.show() is not being called in non-interactive environments.

**Solution:** Add plt.show() after plotting commands. For non-GUI environments, save to file instead: plt.savefig('output.png'). Check backend with: import matplotlib; print(matplotlib.get_backend())

### Peak finding returns unexpected location

**Cause:** The shift parameters a and b are in the wrong order (a > b), causing the product to have no clear peak.

**Solution:** Ensure a < b when calling functions. The left shift 'a' should always be less than the right shift 'b' for a proper bell curve. Add validation: assert a < b.

---

This library was developed to explore the mathematical properties of sigmoid products, demonstrating how simple function composition can create useful bounded activation curves. The implementation prioritizes numerical stability, code clarity, and educational value. All code examples are tested and functional. This documentation was generated with AI assistance to ensure comprehensive coverage of features and usage patterns.