import numpy as np
import mvmd_python as mvmd
import torch
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt

def load_roi_timeseries(file_path):
    mat = loadmat(file_path)
    data = mat["m2_ROIsig"]
    
    return data
    
    




# MVMD parameters
alpha = 2000.0       # moderate bandwidth constraint
tau = 0.0            # noise-tolerance (no strict fidelity enforcement)
K = 4                # number of modes
DC = 0               # no DC mode
init = 1             # initialize omegas uniformly
tol = 1e-6           # tolerance
max_N = 500          # maximum number of iterations


def perform_mvmd_on_roi_timeseries():
    data = load_roi_timeseries(r"C:\Users\matsei\Documents\Mats og Odd Arne\Prosjektoppgave\ISC_data\m2_ROIsig.mat")
    for subject_idx in range(data.shape[2]):
        signal = torch.from_numpy(data[:, :, subject_idx]).float()
        u, u_hat, omega = mvmd.mvmd(signal, alpha, tau, K,  DC, init, tol, max_N)
        print(f"Subject {subject_idx}/{data.shape[2]} MVMD completed.")
        np.savez_compressed(f"mvmd_decomposed_modes{subject_idx}.npz", modes=u.cpu().numpy(), u_hat = u_hat.cpu().numpy(), frequencies=omega.cpu().numpy())    
        
perform_mvmd_on_roi_timeseries()
# def save_mvmd_results(u, u_hat, omega):


### Saving the results
# Convert decomposed modes to numpy arrays 







data = np.load("mvmd_decomposed_modes.npz")
u = data["modes"]
omega = data["frequencies"]
#u_hat = data["u_hat"]

# plot the decomposed modes
def plot_imfs(u, roi_idx=0, tr=None, title_prefix="IMFs for ROI"):
    """
    Plot IMFs (modes) for a single ROI.

    Parameters
    ----------
    u : np.ndarray
        Decomposed modes, shape (K, T, C):
        - K = number of modes
        - T = number of time points
        - C = number of ROIs (channels)
    roi_idx : int, optional
        Index of ROI to plot (0-based). Default is 0.
    tr : float or None, optional
        Repetition time (seconds). If given, x-axis will be in seconds.
        If None, x-axis will be sample index.
    title_prefix : str, optional
        Text prefix for the figure title.
    """
    # Check dimensions
    if u.ndim != 3:
        raise ValueError(f"Expected u with shape (K, T, C), got shape {u.shape}")
    
    K, T, C = u.shape
    if not (0 <= roi_idx < C):
        raise ValueError(f"roi_idx must be in [0, {C-1}], got {roi_idx}")
    
    # Extract time series for this ROI: shape (K, T)
    roi_imfs = u[:, :, roi_idx]
    
    # Time axis
    if tr is not None:
        t = np.arange(T) * tr
        x_label = "Time (s)"
    else:
        t = np.arange(T)
        x_label = "Sample index"
    
    # Plot
    fig, axes = plt.subplots(K, 1, figsize=(10, 2 * K), sharex=True)
    
    # If only one mode, axes is not an array
    if K == 1:
        axes = [axes]
    
    for k in range(K):
        ax = axes[k]
        ax.plot(t, roi_imfs[k, :])
        ax.set_ylabel(f"IMF {k+1}")
        ax.grid(True, linestyle="--", alpha=0.4)
    
    axes[-1].set_xlabel(x_label)
    fig.suptitle(f"{title_prefix} {roi_idx}", y=0.95)
    plt.tight_layout()
    plt.show()

plot_imfs(u, roi_idx=0, tr=2.0, title_prefix="Decomposed Modes for ROI 0")