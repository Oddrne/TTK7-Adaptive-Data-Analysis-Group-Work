import numpy as np
import mvmd_python as mvmd
import torch
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.signal import hilbert, medfilt

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
        
# perform_mvmd_on_roi_timeseries()
# def save_mvmd_results(u, u_hat, omega):


### Saving the results
# Convert decomposed modes to numpy arrays 


def load_network_table(path="network_table.txt") -> pd.DataFrame:
    """
    Load network_table.txt and return a DataFrame with useful columns:
    ['label', 'roi', 'hemi', 'network', 'subnet'].
    """
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    data = []
    for i in range(0, len(lines), 2):  # step through pairs
        label = lines[i]
        nums = lines[i + 1].split()
        roi = int(nums[0])

        # Parse components from label: e.g., '7Networks_LH_Default_PFC_1'
        parts = label.split("_")
        hemi = parts[1] if len(parts) > 1 else None
        network = parts[2] if len(parts) > 2 else None
        subnet = parts[3] if len(parts) > 3 else None

        data.append({"label": label, "roi": roi, "hemi": hemi, "network": network, "subnet": subnet})

    return pd.DataFrame(data)


def test_load_network_table():
    # Load the table
    df = load_network_table("network_table.txt")

    # Basic checks
    assert not df.empty, "DataFrame is empty"
    assert all(col in df.columns for col in ["label", "roi", "hemi", "network", "subnet"]), "Missing columns"

    # Check that ROI values are integers
    assert df["roi"].dtype == int, "ROI column is not integer"

    # Check that hemispheres are LH or RH
    assert set(df["hemi"].unique()).issubset({"LH", "RH"}), "Unexpected hemisphere values"

    # Check that networks include expected names
    expected_networks = {"Default", "Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont"}
    assert any(net in expected_networks for net in df["network"].unique()), "Networks not parsed correctly"

    # Test helper function
    dmn_rois = get_rois_by_network(df, "Default")
    assert isinstance(dmn_rois, list), "get_rois_by_network did not return a list"
    assert all(isinstance(r, int) for r in dmn_rois), "ROI list contains non-integers"

    print("✅ All tests passed! Total entries:", len(df))
    print("Example DMN ROIs:", dmn_rois[:10])


def get_rois_by_network(df: pd.DataFrame, network_name: str, hemis=("LH", "RH")) -> list:
    """
    Return ROI indices for a given network (e.g., 'Default') and hemispheres.
    """
    return df[(df["network"] == network_name) & (df["hemi"].isin(hemis))]["roi"].tolist()



import numpy as np

def rms_amplitude(signal):
    return np.sqrt(np.mean(signal**2))

def calculate_freq_amplitude(subject_count=140, mvmd_prefix="mvmd_decomposed_modes", frequency_idx=3, network_name="Default"):
    
    # Load DMN ROI indices once
    dmn_rois = [idx - 1 for idx in get_rois_by_network(load_network_table(), network_name)]  # zero-based

    results = []

    for subject_idx in range(subject_count):
        data = np.load(f"{mvmd_prefix}{subject_idx}.npz")
        u = data["modes"]      # shape: (K, T, R) where K = IMFs
        omega = data["frequencies"]

        # First IMF (lowest frequency)
        imf = frequency_idx

        # Compute RMS amplitude for each DMN ROI in the first IMF
        roi_amplitudes = [rms_amplitude(u[imf, :, roi]) for roi in dmn_rois]

        # Aggregate across DMN ROIs (mean amplitude)
        dmn_mean_amplitude = np.mean(roi_amplitudes)

        results.append({
            "subject": subject_idx,
            "dmn_mean_amplitude": dmn_mean_amplitude,
            "roi_amplitudes": roi_amplitudes,
            "frequency": omega[imf]
        })

        print(f"Subject {subject_idx}: DMN mean amplitude = {dmn_mean_amplitude:.4f}")

    return results

import matplotlib.pyplot as plt
import numpy as np


def plot_sorted_amplitudes(
    results,
    sort_key="dmn_mean_amplitude",
    ascending=False,
    figsize=(10, 5),
    point_size=35,
    alpha=0.8,
    title="Sorted DMN Mean Amplitudes",
):
    """
    Plot DMN mean amplitudes from the results list as a sorted scatterplot.

    Parameters
    ----------
    results : list of dict
        Output from calculate_freq_amplitude().
    sort_key : str
        Which field of the results dict to sort by.
        Default = "dmn_mean_amplitude".
    ascending : bool
        Sort order. Default: descending (largest amplitude first).
    figsize : tuple
        Figure size.
    point_size : int
        Scatter point size.
    alpha : float
        Marker transparency.
    title : str
        Plot title.
    """

    # ---- Extract data ----
    amplitudes = np.array([r[sort_key] for r in results])
    subjects = np.array([r["subject"] for r in results])

    # ---- Sort ----
    sort_idx = np.argsort(amplitudes)
    if not ascending:
        sort_idx = sort_idx[::-1]     # largest → smallest

    sorted_amplitudes = amplitudes[sort_idx]
    sorted_subjects = subjects[sort_idx]

    # ---- Plot ----
    plt.figure(figsize=figsize)
    plt.scatter(range(len(sorted_amplitudes)), sorted_amplitudes,
                s=point_size, alpha=alpha)
    plt.plot(sorted_amplitudes, alpha=0.4)   # simple connecting line

    plt.xlabel("Sorted Subjects")
    plt.ylabel("DMN Mean Amplitude")
    plt.title(title)

    # optional: show original subject labels on hover-like text output
    for i, subj in enumerate(sorted_subjects):
        plt.text(i, sorted_amplitudes[i], str(subj),
                 fontsize=7, alpha=0.6, ha='center', va='bottom')

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return sorted_subjects, sorted_amplitudes


def calculate_amplitude_of_DMN():
    
    for subject_idx in range(140):
        data = np.load(f"mvmd_decomposed_modes{subject_idx}.npz")
        u = data["modes"]
        omega = data["frequencies"]
        for DMN_idx in get_rois_by_network(load_network_table(), "Default"):
            dmni = DMN_idx - 1  # Adjust for 0-based indexing
            dmni_amplitudes = rms_amplitude(u[:, :, dmni])
            #np.savez_compressed(f"DMN_amplitudes_subject{subject_idx}_roi{DMN_idx}.npz", amplitudes=dmni_amplitudes, frequencies=omega)
            print(f"Subject {subject_idx}, ROI {DMN_idx} amplitude: {dmni_amplitudes}.")

#u_hat = data["u_hat"]

# plot the decomposed modes

def plot_imfs(u, omega=None, roi_idx=0, tr=None, title_prefix="IMFs for ROI"):
    """
    Plot IMFs (modes) for a single ROI, with optional center frequencies.
    """

    # ---- checks ----
    if u.ndim != 3:
        raise ValueError(f"Expected u with shape (K, T, C), got shape {u.shape}")

    K, T, C = u.shape
    if not (0 <= roi_idx < C):
        raise ValueError(f"roi_idx must be in [0, {C-1}], got {roi_idx}")

    # ---- handle omega ----
    if omega is not None:
        omega = np.asarray(omega)
        if omega.ndim == 2:
            # Shape (T, K): instantaneous frequencies → reduce to one per IMF
            omega = omega.mean(axis=0)
        elif omega.ndim == 1 and len(omega) == K:
            pass
        else:
            raise ValueError(f"omega must be shape (K,) or (T,K). Got shape {omega.shape}")

    # Extract ROI IMFs: shape (K, T)
    roi_imfs = u[:, :, roi_idx]

    # ---- time axis ----
    if tr is not None:
        t = np.arange(T) * tr
        x_label = "Time (s)"
    else:
        t = np.arange(T)
        x_label = "Sample index"

    # ---- create figure ----
    fig, axes = plt.subplots(
        K, 1,
        figsize=(10, 2.3 * K),
        sharex=True
    )

    if K == 1:
        axes = [axes]

    for k in range(K):
        ax = axes[k]
        ax.plot(t, roi_imfs[k, :], linewidth=1.2)
        ax.grid(True, linestyle="--", alpha=0.4)

        # subplot title with frequency
        if omega is not None:
            ax.set_title(f"IMF {k+1}", fontsize=10, pad=3)
        else:
            ax.set_title(f"IMF {k+1}", fontsize=10, pad=3)

    axes[-1].set_xlabel(x_label)

    # ---- main title (non-overlapping) ----
    fig.suptitle(f"{title_prefix} {roi_idx}", fontsize=14, y=0.99)

    # Add top margin for suptitle
    plt.subplots_adjust(top=0.90)

    plt.show()

def plot_imf_instantaneous_frequency(
    u,
    roi_idx=0,
    tr=2.0,
    title_prefix="Instantaneous frequency (Hilbert) for ROI"
):
    """
    Compute and plot instantaneous frequency of IMFs using Hilbert transform.
    NO smoothing is applied.

    Parameters
    ----------
    u : np.ndarray
        IMFs/modes with shape (K, T, C).

    roi_idx : int
        ROI/channel index to analyze.

    tr : float
        Repetition time in seconds. Converts phase derivative to Hz.

    title_prefix : str
        Figure title prefix.

    Returns
    -------
    inst_freq_hz : np.ndarray
        Instantaneous frequency in Hz, shape (K, T-1)

    inst_phase : np.ndarray
        Instantaneous phase, shape (K, T)

    inst_amp : np.ndarray
        Instantaneous amplitude (Hilbert envelope), shape (K, T)
    """

    if u.ndim != 3:
        raise ValueError(f"Expected u with shape (K,T,C), got {u.shape}")

    K, T, C = u.shape
    if not (0 <= roi_idx < C):
        raise ValueError(f"roi_idx must be in [0,{C-1}]")

    roi_imfs = u[:, :, roi_idx]  # (K,T)
    dt = tr

    inst_phase = np.zeros((K, T))
    inst_amp = np.zeros((K, T))
    inst_freq_hz = np.zeros((K, T - 1))

    for k in range(K):
        x = roi_imfs[k]

        # analytic signal via Hilbert transform
        z = hilbert(x)
        amp = np.abs(z)
        phase = np.unwrap(np.angle(z))

        # instantaneous frequency in Hz:
        # f(t) = (1/2π) * dφ/dt
        dphi = np.diff(phase)
        freq_hz = (1.0 / (2.0 * np.pi)) * (dphi / dt)

        inst_amp[k] = amp
        inst_phase[k] = phase
        inst_freq_hz[k] = freq_hz

    # time axis for frequencies is T-1
    t_freq = np.arange(T - 1) * dt

    # ---- plot ----
    fig, axes = plt.subplots(K, 1, figsize=(10, 2.2 * K), sharex=True)

    if K == 1:
        axes = [axes]

    for k in range(K):
        ax = axes[k]
        ax.plot(t_freq, inst_freq_hz[k], linewidth=1.0)
        ax.set_ylabel(f"IMF {k+1} (Hz)")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Time (s)")

    fig.suptitle(f"{title_prefix} {roi_idx}", fontsize=14, y=0.99)
    plt.subplots_adjust(top=0.90)

    plt.show()

    return inst_freq_hz, inst_phase, inst_amp

def plot_low_high_similarity(
    low_results,
    high_results,
    low_key="dmn_mean_amplitude",
    high_key="dmn_mean_amplitude",
    title_left="Amplitudes sorted by LOW IMF",
    figsize=(7, 5),
    point_size=35,
    alpha=0.85,
):
    """
    Plot LOW vs HIGH IMF amplitudes sorted by the LOW IMF.
    (Second plot removed.)
    """

    # --- Extract aligned amplitudes by subject ---
    low_map = {r["subject"]: r[low_key] for r in low_results}
    high_map = {r["subject"]: r[high_key] for r in high_results}

    subjects = np.array(sorted(set(low_map) & set(high_map)))
    if len(subjects) == 0:
        raise ValueError("No overlapping subjects between low_results and high_results.")

    low_amp = np.array([low_map[s] for s in subjects])
    high_amp = np.array([high_map[s] for s in subjects])

    # --- Sort by LOW IMF amplitude ---
    order = np.argsort(low_amp)
    subjects_sorted = subjects[order]
    low_sorted = low_amp[order]
    high_sorted = high_amp[order]

    # --- Correlations ---
    pearson_r = np.corrcoef(low_amp, high_amp)[0, 1]

    low_rank = low_amp.argsort().argsort()
    high_rank = high_amp.argsort().argsort()
    spearman_r = np.corrcoef(low_rank, high_rank)[0, 1]

    # --- Single plot ---
    plt.figure(figsize=figsize)

    x = np.arange(len(subjects_sorted))
    plt.scatter(x, low_sorted, s=point_size, alpha=alpha, label="LOW IMF")
    plt.scatter(x, high_sorted, s=point_size, alpha=alpha, label="HIGH IMF")
    plt.plot(x, low_sorted, alpha=0.4)
    plt.plot(x, high_sorted, alpha=0.4)

    plt.title(title_left)
    plt.xlabel("Subjects (sorted by LOW IMF amplitude)")
    plt.ylabel("DMN mean amplitude")
    plt.grid(alpha=0.3)
    plt.legend()

    # Correlation annotation
    plt.text(
        0.01, 0.99,
        f"Pearson r = {pearson_r:.3f}\nSpearman ρ = {spearman_r:.3f}",
        transform=plt.gca().transAxes,
        ha="left", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.1)
    )

    plt.tight_layout()
    plt.show()

    return {
        "subjects": subjects,
        "low_amp": low_amp,
        "high_amp": high_amp,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "sorted_subjects": subjects_sorted,
        "low_sorted": low_sorted,
        "high_sorted": high_sorted,
    }



def plot_network_boxplot(network_names, network_amplitude_lists,
                         ylabel="Low-frequency IMF amplitude",
                         title="Distribution of Low-frequency Amplitudes Across Networks",
                         figsize=(10, 5)):
    """
    Create a boxplot showing differences in low-frequency amplitude across networks.
    
    Parameters
    ----------
    network_names : list of str
        Names of the networks, e.g. ["DMN", "Salience", "Limbic", ...].
    
    network_amplitude_lists : list of lists
        Each element must be a list (or array) of amplitude values for that network.
        Example: 
            [
                [amp_sub1, amp_sub2, ...],   # DMN
                [amp_sub1, amp_sub2, ...],   # Salience
                ...
            ]
    
    ylabel : str
        Y-axis label (default: "Low-frequency IMF amplitude")
        
    title : str
        Plot title
        
    figsize : tuple
        Figure size
    """

    plt.figure(figsize=figsize)

    # Create the boxplot
    plt.boxplot(network_amplitude_lists, labels=network_names, 
                patch_artist=True,  # allows colored boxes if desired
                boxprops=dict(facecolor='lightgray', alpha=0.6),
                medianprops=dict(color='black', linewidth=1.5))

    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    
    plt.show()


def plot_network_correlation_heatmap(network_values_dict,
                                     title="High-frequency network amplitude correlations",
                                     figsize=(6.5, 5.5),
                                     vmin=-1, vmax=1):
    """
    Compute and plot a correlation heatmap across networks.

    Parameters
    ----------
    network_values_dict : dict
        {network_name: [amplitude_per_subject]}
        All lists must be same length and aligned by subject order.
    """

    networks = list(network_values_dict.keys())
    mat = np.column_stack([network_values_dict[n] for n in networks])  # subjects × networks

    corr = np.corrcoef(mat, rowvar=False)  # networks × networks

    plt.figure(figsize=figsize)
    im = plt.imshow(corr, vmin=vmin, vmax=vmax, cmap="coolwarm")

    plt.xticks(range(len(networks)), networks, rotation=45, ha="right")
    plt.yticks(range(len(networks)), networks)

    plt.title(title)
    plt.colorbar(im, shrink=0.8, label="Pearson r")
    plt.tight_layout()
    plt.show()

    return corr


data = np.load("mvmd_decomposed_modes0.npz")
u = data["modes"]
#omega = data["frequencies"]
omega_full = data["frequencies"]  # (T,K) or (T,K,2)
omega_real = omega_full[..., 0] if omega_full.ndim == 3 else omega_full
omega_center = omega_real.mean(axis=0)          # (K,)
omega_hz = omega_center / (2*np.pi*2.0)

#test_load_network_table()
# plot_imfs(u, omega_full, roi_idx=0, tr=2.0)
# plot_imf_instantaneous_frequency(u, roi_idx=0, tr=2.0)


DMN_amplitudes_low = calculate_freq_amplitude(140, "mvmd_decomposed_modes", frequency_idx=3)
Vis_amplitudes_low = calculate_freq_amplitude(140, "mvmd_decomposed_modes", frequency_idx=3, network_name="Vis")
SalVentAttn_amplitudes_low = calculate_freq_amplitude(140, "mvmd_decomposed_modes", frequency_idx=3, network_name="SalVentAttn")
SomMot_amplitudes_low = calculate_freq_amplitude(140, "mvmd_decomposed_modes", frequency_idx=3, network_name="SomMot")
DorsAttn_amplitudes_low = calculate_freq_amplitude(140, "mvmd_decomposed_modes", frequency_idx=3, network_name="DorsAttn")
Limbic_amplitudes_low = calculate_freq_amplitude(140, "mvmd_decomposed_modes", frequency_idx=3, network_name="Limbic")
Cont_amplitudes_low = calculate_freq_amplitude(140, "mvmd_decomposed_modes", frequency_idx=3, network_name="Cont")



DMN_values = [r["dmn_mean_amplitude"] for r in DMN_amplitudes_low]
Sal_values = [r["dmn_mean_amplitude"] for r in SalVentAttn_amplitudes_low]
Lim_values = [r["dmn_mean_amplitude"] for r in Limbic_amplitudes_low]
FPN_values = [r["dmn_mean_amplitude"] for r in Cont_amplitudes_low]
Vis_values = [r["dmn_mean_amplitude"] for r in Vis_amplitudes_low]
Som_values = [r["dmn_mean_amplitude"] for r in SomMot_amplitudes_low]
Dor_values = [r["dmn_mean_amplitude"] for r in DorsAttn_amplitudes_low]

network_names = ["DMN", "Salience", "Limbic", "Control", "Visual", "SomMot", "DorsAttn"]
network_data = [DMN_values, Sal_values, Lim_values, FPN_values, Vis_values, Som_values, Dor_values]

low_network_dict = {
    "DMN": DMN_values,
    "SalVentAttn": Sal_values,
    "Limbic": Lim_values,
    "Cont": FPN_values,
    "DorsAttn": Dor_values,
    "SomMot": Som_values,
    "Vis": Vis_values,
}

low_corr = plot_network_correlation_heatmap(low_network_dict)

#plot_network_boxplot(network_names, network_data)
#plot_low_high_similarity(DMN_amplitudes_low, DMN_amplitudes_high)
