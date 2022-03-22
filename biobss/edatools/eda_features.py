from .hjorth import *
from .signal_features import *







def from_scr(signal_phasic, sr):

    features = {}
    min = np.min(signal_phasic)
    max = np.max(signal_phasic)
    features['scr_mean'] = signal_phasic.mean()
    features['scr_std'] = np.std(signal_phasic)
    features['scr_max'] = max
    features['scr_min'] = max
    features["scr_dynamic_range"] = (max-min)
    s_d1 = np.gradient(signal_phasic)
    features["fmsc"] = s_d1.mean()
    features["fdsc"] = np.std(s_d1)
    s_d2 = np.gradient(s_d1)
    features["smsc"] = s_d2.mean()
    features["sdsc"] = np.std(s_d2)
    alsc = calculate_alsc(signal_phasic)
    features["alsc"] = alsc/len(signal_phasic)
    insc = calculate_insc(signal_phasic)
    features["insc"] = insc
    apsc = calculate_apsc(signal_phasic)
    features["apsc"] = apsc
    rmsc = calculate_rmsc(apsc)
    features["rmsc"] = rmsc
    ilsc = insc/rmsc
    elsc = insc/alsc
    features["ilsc"] = ilsc
    features["elsc"] = elsc
    features["kusc"] = stats.kurtosis(signal_phasic)
    features["sksc"] = stats.skew(signal_phasic)

    features["mosc"] = stats.moment(signal_phasic, 2)

    sig_fft = fft(np.array(signal_phasic))
    freqs, psd = signal.welch(sig_fft, return_onesided=False)
    f1sc = psd[np.where(np.logical_and(freqs > 0.1, freqs < 0.2))]
    f2sc = psd[np.where(np.logical_and(freqs > 0.2, freqs < 0.3))]
    f3sc = psd[np.where(np.logical_and(freqs > 0.3, freqs < 0.4))]

    features["f1sc"] = f1sc.mean()
    features["f2sc"] = f2sc.mean()
    features["f3sc"] = f3sc.mean()

    return features