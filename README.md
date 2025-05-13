# Deep Learning Lab

My Deep Learing course assignments.

## Content

- **Lab1**: Implements core NumPy operations and basic image processing including grayscale conversion, Sobel edge detection, and patch shuffling using matrix manipulation.
- **Lab2**: Demonstrates polynomial regression with gradient descent optimization, comparing underfitting/overfitting scenarios and L2 regularization effects through synthetic sinusoidal data.
- **Lab3**: Builds linear models (LSM/Ridge Regression) and compares them with scikit-learn/numpy implementations, extending to polynomial feature space for nonlinear regression.
- **Lab4**: Implements fundamental classification algorithms - logistic regression (binary), softmax regression (multi-class), and perceptron - evaluated on breast cancer and iris datasets with accuracy metrics.
- **Lab5**: Implements gradient descent optimization for functions y=x² (starting from x=1) and z=x²+y² (starting from x=1,y=3), with visualization of gradient descent trajectories.
- **Lab6**: Manually implements a three-layer neural network with backpropagation algorithm, then replicates using PyTorch framework. Compares performance differences between manual and PyTorch implementations on Breast Cancer dataset through training/testing metrics.
- **Lab7**: Implements classical convolutional neural network models (LeNet-5 and AlexNet) for image classification. Trains and evaluates models on MNIST and CIFAR-10 datasets, comparing architectural differences and performance characteristics.

## Usage

This project uses [uv](https://github.com/astral-sh/uv) as a Python package manager. Setup instructions:

### Prerequisites

- Python 3.x
- uv installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Setup

```bash
# Clone repository
git clone  https://github.com/timedegree/Deep-Learning-Lab.git
cd Deep-Learning-Lab

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # Linux/macOS
# or . .venv\Scripts\activate  # Windows

# Install locked dependencies from pyproject.toml
uv pip install --locked