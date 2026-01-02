import numpy as np
from scipy import stats
data = np.array([12, 15, 20, 22, 22, 25, 30, 30, 30, 35])
mean_np = np.mean(data)
median_np = np.median(data)
mode_np = stats.mode(data, keepdims=True).mode[0]
variance_np = np.var(data)
std_dev_np = np.std(data)
print("\n--- Using NumPy & SciPy ---")
print("Mean:", mean_np)
print("Median:", median_np)
print("Mode:", mode_np)
print("Variance:", variance_np)
print("Standard Deviation:", std_dev_np)
