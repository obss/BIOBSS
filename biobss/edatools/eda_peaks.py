import neurokit2 as nk


def find_peaks(phasic_signal, sampling_rate):

    peak_signal, info = nk.eda_peaks(
        phasic_signal.values,
        sampling_rate=sampling_rate,
        method="neurokit",
        amplitude_min=0.1,
    )

    return peak_signal, info
