import numpy as np
from numpy.typing import ArrayLike

from biobss.ppgtools.ppg_peaks import correct_missing_duplicate_peaks

# Time domain features
FEATURES_APG = {
    "a_a": lambda apg_sig, _0, _1, locs_a, _2, _3, _4, _5: np.mean(apg_sig[locs_a]),
    "t_a": lambda _0, sampling_rate, locs_O, locs_a, _1, _2, _3, _4: np.mean((locs_a - locs_O[:-1]) / sampling_rate),
    "a_b": lambda apg_sig, _0, _1, _2, locs_b, _3, _4, _5: np.mean(apg_sig[locs_b]),
    "t_b": lambda _0, sampling_rate, locs_O, _1, locs_b, _2, _3, _4: np.mean((locs_b - locs_O[:-1]) / sampling_rate),
    "a_c": lambda apg_sig, _0, _1, _2, _3, locs_c, _4, _5: np.mean(apg_sig[locs_c]),
    "t_c": lambda _0, sampling_rate, locs_O, _1, _2, locs_c, _3, _4: np.mean((locs_c - locs_O[:-1]) / sampling_rate),
    "a_d": lambda apg_sig, _0, _1, _2, _3, _4, locs_d, _5: np.mean(apg_sig[locs_d]),
    "t_d": lambda _0, sampling_rate, locs_O, _1, _2, _3, locs_d, _4: np.mean((locs_d - locs_O[:-1]) / sampling_rate),
    "a_e": lambda apg_sig, _0, _1, _2, _3, _4, locs_d, locs_e: np.mean(apg_sig[locs_e]),
    "t_e": lambda _0, sampling_rate, locs_O, _1, _2, _3, _4, locs_e: np.mean((locs_e - locs_O[:-1]) / sampling_rate),
    "a_b_a": lambda apg_sig, _0, _1, locs_a, locs_b, _2, _3, _4: np.mean(apg_sig[locs_b]) / np.mean(apg_sig[locs_a]),
    "a_c_a": lambda apg_sig, _0, _1, locs_a, _2, locs_c, _3, _4: np.mean(apg_sig[locs_c]) / np.mean(apg_sig[locs_a]),
    "a_d_a": lambda apg_sig, _0, _1, locs_a, _2, _3, locs_d, _4: np.mean(apg_sig[locs_d]) / np.mean(apg_sig[locs_a]),
    "a_e_a": lambda apg_sig, _0, _1, locs_a, _2, _3, _4, locs_e: np.mean(apg_sig[locs_e]) / np.mean(apg_sig[locs_a]),
    "a_cdb_a": lambda apg_sig, _0, _1, locs_a, locs_b, locs_c, locs_d, _2: (
        np.mean(apg_sig[locs_c]) + np.mean(apg_sig[locs_d]) - np.mean(apg_sig[locs_b])
    )
    / np.mean(apg_sig[locs_a]),
    "a_bcde_a": lambda apg_sig, _0, _1, locs_a, locs_b, locs_c, locs_d, locs_e: (
        np.mean(apg_sig[locs_b]) - np.mean(apg_sig[locs_c]) - np.mean(apg_sig[locs_d]) - np.mean(apg_sig[locs_e])
    )
    / np.mean(apg_sig[locs_a]),
    "a_bcd_a": lambda apg_sig, _0, _1, locs_a, locs_b, locs_c, locs_d, _2: (
        np.mean(apg_sig[locs_b]) - np.mean(apg_sig[locs_c]) - np.mean(apg_sig[locs_d])
    )
    / np.mean(apg_sig[locs_a]),
    "a_be_a": lambda apg_sig, _0, _1, locs_a, locs_b, _2, _3, locs_e: (
        np.mean(apg_sig[locs_b]) - np.mean(apg_sig[locs_e])
    )
    / np.mean(apg_sig[locs_a]),
}


def get_apg_features(
    apg_sig: ArrayLike, locs_O: ArrayLike, fiducials: dict, sampling_rate: float, prefix: str = "apg"
) -> dict:
    """Calculates APG features.

        a_a: Mean amplitude of a waves
        t_a: Mean duration of a waves
        a_b: Mean amplitude of b waves
        t_b: Mean duration of b waves
        a_c: Mean amplitude of c waves
        t_c: Mean duration of c waves
        a_d: Mean amplitude of d waves
        t_d: Mean duration of d waves
        a_e: Mean amplitude of e waves
        t_e: Mean duration of e waves
        a_b_a: Mean ratio of b wave amplitude to a wave amplitude
        a_c_a: Mean ratio of c wave amplitude to a wave amplitude
        a_d_a: Mean ratio of d wave amplitude to a wave amplitude
        a_e_a: Mean ratio of e wave amplitude to a wave amplitude
        a_cdb_a: Mean ratio of a_c + a_d - a_b to a wave amplitude
        a_bcde_a: Mean ratio of a_b - a_c - a_d - a_e to a wave amplitude
        a_bcd_a: Mean ratio of a_b - a_c - a_d to a wave amplitude
        a_be_a: Mean ratio of a_b - a_e to a wave amplitude

    Args:
        apg_sig (ArrayLike): APG signal.
        locs_O (ArrayLike): PPG signal onset locations.
        fiducials (dict): APG fiducials.
        sampling_rate (float): Sampling rate of the APG signal (Hz).
        prefix (str, optional): Prefix for the features. Defaults to 'apg'.

    Raises:
        ValueError: If sampling rate is not greater than 0.

    Returns:
        dict: APG features
    """

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    feature_list = FEATURES_APG.copy()

    fiducial_names = ["a_waves", "b_waves", "c_waves", "d_waves", "e_waves"]
    fiducials = {key: fiducials.get(key, []) for key in fiducial_names}

    locs_a = fiducials["a_waves"]
    locs_b = fiducials["b_waves"]
    locs_c = fiducials["c_waves"]
    locs_d = fiducials["d_waves"]
    locs_e = fiducials["e_waves"]

    if len(locs_a) == 0:
        a_features = ["a_a", "t_a", "a_b_a", "a_c_a", "a_d_a", "a_e_a", "a_cdb_a", "a_bcde_a", "a_bcd_a", "a_be_a"]
        [feature_list.pop(key, None) for key in a_features]
    else:
        locs_a, _ = correct_missing_duplicate_peaks(locs_valleys=locs_O, locs_peaks=locs_a, peaks=apg_sig[locs_a])

    if len(locs_b) == 0:
        b_features = ["a_b", "t_b", "a_b_a", "a_cdb_a", "a_bcde_a", "a_bcd_a", "a_be_a"]
        [feature_list.pop(key, None) for key in b_features]
    else:
        locs_b, _ = correct_missing_duplicate_peaks(locs_valleys=locs_O, locs_peaks=locs_b, peaks=apg_sig[locs_b])

    if len(locs_c) == 0:
        c_features = ["a_c", "t_c", "a_c_a", "a_cdb_a", "a_bcde_a", "a_bcd_a"]
        [feature_list.pop(key, None) for key in c_features]
    else:
        locs_c, _ = correct_missing_duplicate_peaks(locs_valleys=locs_O, locs_peaks=locs_c, peaks=apg_sig[locs_c])

    if len(locs_d) == 0:
        d_features = [
            "a_d",
            "t_d",
            "a_d_a",
            "a_cdb_a",
            "a_bcde_a",
            "a_bcd_a",
        ]
        [feature_list.pop(key, None) for key in d_features]
    else:
        locs_d, _ = correct_missing_duplicate_peaks(locs_valleys=locs_O, locs_peaks=locs_d, peaks=apg_sig[locs_d])

    if len(locs_e) == 0:
        e_features = ["a_e", "t_e", "a_e_a", "a_bcde_a", "a_be_a"]
        [feature_list.pop(key, None) for key in e_features]
    else:
        locs_e, _ = correct_missing_duplicate_peaks(locs_valleys=locs_O, locs_peaks=locs_e, peaks=apg_sig[locs_e])

    features = {}
    for key, func in feature_list.items():
        try:
            features["_".join([prefix, key])] = func(
                apg_sig, sampling_rate, locs_O, locs_a, locs_b, locs_c, locs_d, locs_e
            )
        except:
            features["_".join([prefix, key])] = np.nan

    return features
