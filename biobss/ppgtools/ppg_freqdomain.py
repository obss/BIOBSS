from numpy.typing import ArrayLike

from biobss.common.signal_fft import *
from biobss.common.signal_psd import *

# Frequency domain features
FUNCTIONS_FREQ_SEGMENT = {
    "p_1": lambda sigfft, freq, _0, _1: fft_peaks(sigfft, freq, 1),
    "f_1": lambda sigfft, freq, _0, _1: fft_peaks(sigfft, freq, 1, loc=True),
    "p_2": lambda sigfft, freq, _0, _1: fft_peaks(sigfft, freq, 2),
    "f_2": lambda sigfft, freq, _0, _1: fft_peaks(sigfft, freq, 2, loc=True),
    "p_3": lambda sigfft, freq, _0, _1: fft_peaks(sigfft, freq, 3),
    "f_3": lambda sigfft, freq, _0, _1: fft_peaks(sigfft, freq, 3, loc=True),
    "pow": lambda _0, _1, pxx, fxx: sig_power(pxx, fxx, [0, 2]),
    "rpow": lambda _0, _1, pxx, fxx: sig_power(pxx, fxx, [0, 2.25]) / sig_power(pxx, fxx, [0, 5]),
}


def ppg_freq_features(
    sig: ArrayLike, sampling_rate: float, input_types: list, fiducials: dict = None, prefix: str = "ppg"
) -> dict:
    """Calculates frequency-domain features

    Segment-based features:
        p_1: The amplitude of the first peak from the fft of the signal
        f_1: The frequency at which the first peak from the fft of the signal occurred
        p_2: The amplitude of the second peak from the fft of the signal
        f_2: The frequency at which the second peak from the fft of the signal occurred
        p_3: The amplitude of the third peak from the fft of the signal
        f_3: The frequency at which the third peak from the fft of the signal occurred
        pow: Power of the signal at a given range of frequencies ([0,2] Hz).
        rpow: Ratio of the powers of the signal at given ranges of frequencies ([0,2.25] Hz/[0,5] Hz).

    Args:
        sig (ArrayLike): Signal to be analyzed.
        sampling_rate (float): Sampling rate of the signal (Hz).
        input_types (list): Type of feature calculation, should be 'segment'.
        fiducials (dict, optional): Dictionary of fiducial point locations. Defaults to None.
        prefix (str, optional): Prefix for signal type. Defaults to 'ppg'.

    Raises:
        ValueError: If sampling rate is not greater than 0.
        ValueError: If type is not 'segment'.

    Returns:
        dict: Dictionary of calculated features
    """
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    input_types = [x.lower() for x in input_types]

    features_freq = {}
    for type in input_types:

        if type == "segment":

            freq, sigfft = sig_fft(sig=sig, sampling_rate=sampling_rate)
            f, pxx = sig_psd(sig=sig, sampling_rate=sampling_rate, method="welch")

            features_freq = {}
            for key, func in FUNCTIONS_FREQ_SEGMENT.items():
                try:
                    features_freq["_".join([prefix, key])] = func(sigfft, freq, pxx, f)
                except:
                    features_freq["_".join([prefix, key])] = np.nan

        else:
            raise ValueError("Undefined type for frequency domain.")

    return features_freq
