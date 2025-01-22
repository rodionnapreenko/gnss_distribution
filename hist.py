#----------------------------------------------------------------
# Research into the law of distribution of satellite measurements 
# Copyright (c) 2025 Andrew Budo
#----------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def get_average(v: List[float]) -> float:
    """Calculate the mean of a list of values"""
    return np.mean(v)

def get_rms(v: List[float], v_mean: float) -> float:
    """Calculate the root mean square deviation (standard deviation)"""
    return np.std(v, ddof=1)  # ddof=1 for sample standard deviation

def get_values_for_ogiva(bins: int, min_val: float, max_val: float) -> np.ndarray:
    """Generate evenly spaced values for the ogive"""
    return np.linspace(min_val, max_val, bins)

def get_ogiva(v: np.ndarray, mean: float, rms: float) -> np.ndarray:
    """Calculate the normal distribution curve (ogive)"""
    return np.exp(-0.5 * ((v - mean) / rms) ** 2) / (rms * np.sqrt(2 * np.pi))

def main():
    # Initialize lists for data
    n, e, h = [], [], []

    # Read input data
    try:
        while True:
            X, Y, H = map(float, input().split())
            n.append(X)
            e.append(Y)
            h.append(H)
    except EOFError:
        pass

    # Convert lists to numpy arrays for easier computation
    n = np.array(n)
    e = np.array(e)
    h = np.array(h)

    # Calculate statistics
    n_mean = get_average(n)
    e_mean = get_average(e)
    h_mean = get_average(h)

    n_rms = get_rms(n, n_mean)
    e_rms = get_rms(e, e_mean)
    h_rms = get_rms(h, h_mean)

    # Calculate min and max values
    n_min, n_max = np.min(n), np.max(n)
    e_min, e_max = np.min(e), np.max(e)
    h_min, h_max = np.min(h), np.max(h)

    bins = 20

    # Generate ogiva values
    n_x_ogiva = get_values_for_ogiva(bins, 250.0, n_max)
    n_y_ogiva = get_ogiva(n_x_ogiva, n_mean, n_rms)

    e_x_ogiva = get_values_for_ogiva(bins, e_min, 820.0)
    e_y_ogiva = get_ogiva(e_x_ogiva, e_mean, e_rms)

    h_x_ogiva = get_values_for_ogiva(bins, 197.0, h_max)
    h_y_ogiva = get_ogiva(h_x_ogiva, h_mean, h_rms)

    # Create plots
    plt.figure(figsize=(12, 7.8))

    # Scatter plots
    plt.subplot(2, 3, 1)
    plt.plot(n, e, 'g.')
    plt.xlabel('north')
    plt.ylabel('east')
    plt.xlim(250, 450)
    plt.ylim(650, 800)

    plt.subplot(2, 3, 2)
    plt.plot(n, h, 'r.')
    plt.xlabel('north')
    plt.ylabel('height')
    plt.xlim(250, 450)
    plt.ylim(199.0, 204.5)

    plt.subplot(2, 3, 3)
    plt.plot(e, h, 'b.')
    plt.xlabel('east')
    plt.ylabel('height')
    plt.xlim(650, 800)
    plt.ylim(199.0, 204.5)

    # Histograms with ogiva curves
    plt.subplot(2, 3, 4)
    plt.hist(n, bins=10, color='b', alpha=0.5, density=True)
    plt.plot(n_x_ogiva, n_y_ogiva, 'r-')
    plt.xlabel('north')

    plt.subplot(2, 3, 5)
    plt.hist(e, bins=10, color='g', alpha=0.5, density=True)
    plt.plot(e_x_ogiva, e_y_ogiva, 'r-')
    plt.xlabel('east')

    plt.subplot(2, 3, 6)
    plt.hist(h, bins=10, color='r', alpha=0.5, density=True)
    plt.plot(h_x_ogiva, h_y_ogiva, 'r-')
    plt.xlabel('height')

    plt.tight_layout()
    
    # Save the plot with incrementing number
    plot_number = 0  # You might want to manage this externally
    plt.savefig(f'./img/anim{plot_number}.png')
    plt.close()
    print(plot_number + 1)

if __name__ == "__main__":
    main()
