import numpy as np
from numpy.typing import ArrayLike

from biobss.ppgtools.ppg_peaks import correct_missing_duplicate_peaks

# Time domain features
FEATURES_VPG = {
    "a_w": lambda vpg_sig, _0, _1, locs_w, _2, _3: np.mean(vpg_sig[locs_w]),
    "t_w": lambda _0, sampling_rate, locs_O, locs_w, _1, _2: np.mean((locs_w - locs_O[:-1]) / sampling_rate),
    "a_y": lambda vpg_sig, _0, _1, locs_w, locs_y, _2: np.mean(vpg_sig[locs_y]),
    "t_y": lambda _0, sampling_rate, locs_O, _1, locs_y, _2: np.mean((locs_y - locs_O[:-1]) / sampling_rate),
    "a_z": lambda vpg_sig, _0, _1, _2, _3, locs_z: np.mean(vpg_sig[locs_z]),
    "t_z": lambda _0, sampling_rate, locs_O, _1, _2, locs_z: np.mean((locs_z - locs_O[:-1]) / sampling_rate),
    "a_y_w": lambda vpg_sig, _0, _1, locs_w, locs_y, _2: np.mean(vpg_sig[locs_y]) / np.mean(vpg_sig[locs_w]),
}


def get_vpg_features(
    vpg_sig: ArrayLike, locs_O: ArrayLike, fiducials: dict, sampling_rate: float, prefix: str = "vpg"
) -> dict:
    """Calculates VPG features.

        a_w: Mean amplitude of w waves
        t_w: Mean duration of w waves
        a_y: Mean amplitude of y waves
        t_y: Mean duration of y waves
        a_z: Mean amplitude of z waves
        t_z: Mean duration of z waves
        a_y_w: Mean ratio of y wave amplitudes to w wave amplitudes

    Args:
        vpg_sig (ArrayLike): VPG signal.
        locs_O (ArrayLike): PPG signal onset locations.
        fiducials (dict): VPG fiducials.
        sampling_rate (float): Sampling rate of the VPG signal (Hz).
        prefix (str, optional): Prefix for the features. Defaults to 'vpg'.

    Raises:
        ValueError: If sampling rate is not greater than 0.

    Returns:
        dict: VPG features
    """

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    feature_list = FEATURES_VPG.copy()

    fiducial_names = ["w_waves", "y_waves", "z_waves"]
    fiducials = {key: fiducials.get(key, []) for key in fiducial_names}

    locs_w = fiducials["w_waves"]
    locs_y = fiducials["y_waves"]
    locs_z = fiducials["z_waves"]

    if len(locs_w) == 0:
        w_features = ["a_w", "t_w", "a_y_w"]
        [feature_list.pop(key, None) for key in w_features]
    else:
        locs_w, _ = correct_missing_duplicate_peaks(locs_valleys=locs_O, locs_peaks=locs_w, peaks=vpg_sig[locs_w])

    if len(locs_y) == 0:
        y_features = ["a_y", "t_y", "a_y_w"]
        [feature_list.pop(key, None) for key in y_features]
    else:
        locs_y, _ = correct_missing_duplicate_peaks(locs_valleys=locs_O, locs_peaks=locs_y, peaks=vpg_sig[locs_y])

    if len(locs_z) == 0:
        z_features = ["a_z", "t_z"]
        [feature_list.pop(key, None) for key in z_features]
    else:
        locs_z, _ = correct_missing_duplicate_peaks(locs_valleys=locs_O, locs_peaks=locs_z, peaks=vpg_sig[locs_z])

    features = {}
    for key, func in feature_list.items():
        try:
            features["_".join([prefix, key])] = func(vpg_sig, sampling_rate, locs_O, locs_w, locs_y, locs_z)
        except:
            features["_".join([prefix, key])] = np.nan

    return features
