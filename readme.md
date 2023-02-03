# <div align="center"> __BIOBSS__ </div>

A package for processing signals recorded using wearable sensors, such as Electrocardiogram (ECG), Photoplethysmogram (PPG), Electrodermal activity (EDA) and 3-axis acceleration (ACC). 

BIOBSS's main focus is to generate end-to-end pipelines by adding required processes from BIOBSS or other Python packages. Some preprocessing methods were not implemented from scratch but imported from the existing packages.

Main features:

- Applying basic preprocessing steps 
- Assessing quality of PPG and ECG signals
- Extracting features for ECG, PPG, EDA and ACC signals
- Performing Heart Rate Variability (HRV) analysis using PPG or ECG signals
- Extracting respiratory signals from PPG or ECG signals and estimating respiratory rate
- Calculating activity indices from ACC signals
- Generating and saving pipelines 

The table shows the capabilites of BIOBSS and the other Python packages for physiological signal processing.

<table>
  <thead>
    <tr>
      <th style="text-align:center" colspan="2"><b>Functionality</b></th>
      <th style="text-align:center">BIOBSS</th>
      <th style="text-align:center">BioSPPy</th>
      <th style="text-align:center">HeartPy</th>
      <th style="text-align:center">HRV</th>
      <th style="text-align:center">hrv-analysis</th>
      <th style="text-align:center">pyHRV</th>
      <th style="text-align:center">pyPhysio</th>
      <th style="text-align:center">PySiology</th>
      <th style="text-align:center">Neurokit2</th>
      <th style="text-align:center">FLIRT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" colspan="2"><b>File reader</b></td>
      <td align="center"><b>&check;</b></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center">&check;</td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center">&check;</td>
    </tr>
    <tr>
      <td align="center" colspan="2"><b>Sliding window</b></td>
      <td align="center"><b>&check;</b></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center">&check;</td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center">&check;</td>
    </tr>
    <tr>
      <td align="center" colspan="2"><b>Preprocessing</b></td>
      <td align="center"><b>&check;</b></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center">&check;</td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center">&check;</td>
      <td align="center">&check;</td>
    </tr>
    <tr>
      <td align="center" colspan="2"><b>Pipeline</b></td>
      <td align="center"><b>&check;(*)</b></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center">&check;</td>
      <td align="center"></td>
    </tr>
    <tr>
      <td align="center" style="text-align:center" rowspan="5"><b>Processing</b></td>
      <td align="center"><b>ECG</b></td>
      <td align="center"><b>&check;</b></td>
      <td align="center">&check;</td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center">&check;</td>
      <td align="center">&check;</td>
      <td align="center">&check;</td>
      <td align="center"></td>
    </tr>
    <tr>
      <td align="center" style="text-align:center"><b>PPG</b></td>
      <td align="center"><b>&check;</b></td>
      <td align="center">&check;</td>
      <td align="center">&check;</td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center">&check;</td>
      <td align="center"></td>
      <td align="center">&check;</td>
      <td align="center">&check;</td>
    </tr>
    <tr>
      <td align="center" style="text-align:center"><b>IBI / RRI</b></td>
      <td align="center"><b>&check;</b></td>
      <td align="center">&check;</td>
      <td align="center">&check;</td>
      <td align="center">&check;</td>
      <td align="center">&check;</td>
      <td align="center">&check;</td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center">&check;</td>
      <td align="center">&check;</td>
    </tr>
    <tr>
      <td align="center" style="text-align:center"><b>EDA</b></td>
      <td align="center"><b>&check;</b></td>
      <td align="center">&check;</td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center">&check;</td>
      <td align="center">&check;</td>
      <td align="center">&check;</td>
      <td align="center">&check;</td>
    </tr>
    <tr>
      <td align="center" style="text-align:center"><b>ACC</b></td>
      <td align="center"><b>&check;</b></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center">&check;</td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center">&check;</td>
    </tr>
    <tr>
      <td align="center" style="text-align:center" rowspan="4"><b>Feature Extraction</b></td>
      <td align="center"><b>ECG</b></td>
      <td align="center"><b>&check;</b></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
    </tr>
    <tr>
      <td align="center" style="text-align:center"><b>PPG</b></td>
      <td align="center"><b>&check;</b></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
    </tr>
    <tr>
      <td align="center" style="text-align:center"><b>EDA</b></td>
      <td align="center"><b>&check;</b></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center">&check;</td>
    </tr>
    <tr>
      <td align="center" style="text-align:center"><b>ACC</b></td>
      <td align="center"><b>&check;</b></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center">&check;</td>
    </tr>
  </tbody>
</table>

(*): Pipeline module differs between the two packages. BIOBSS pipeline aims to provide a more flexible and customizable pipeline for the user.

__Modified from__ Föll, Simon, et al. “FLIRT: A feature generation toolkit for wearable data.” Computer Methods and Programs in Biomedicine 212 (2021): 106461.

## <div align="left"> __Preprocessing__ </div>
BIOBSS has modules with basic signal preprocessing functionalities. These include:
- Resampling
- Segmentation
- Normalization
- Filtering (basic filtering functions with commonly used filter parameters for each signal type)
- Peak detection 

## <div align="left"> __Visualization__ </div>
BIOBSS has basic plotting modules specific to each signal type. Using the modules, the signals and peaks can be plotted using Matplotlib or Plotly packages.

## <div align="left"> __Signal Quality Assessment__ </div>
Signal quality assessment steps listed below can be used with PPG and ECG signals.
- Clipping detection
- Flatline detection
- Physiological checks
- Morphological checks
- Template matching

## <div align="left"> __Feature Extraction__ </div>

<table>
<thead>
<tr>
<th style="text-align:center">Signal</th>
<th style="text-align:center" width="110">Domain / Type</th>
<th style="text-align:center">Features</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">ECG</td>
<td align="center">Time</td>
<td align="center">Morphological features related to fiducial point locations and amplitudes</td>
</tr>
<tr>
<td align="center" rowspan="3">PPG</td>
<td align="center">Time</td>
<td align="center">Morphological features related to fiducial point locations and amplitudes, zero-crossing rate, signal to noise ratio</td>
</tr>
<tr>
<td align="center">Frequency</td>
<td align="center">Amplitude and frequency of FFT peaks, signal power</td>
</tr>
<tr>
<td align="center">Statistical</td>
<td align="center">Mean, median, standard deviation, percentiles, mean absolute deviation, skewness, kurtosis, entropy</td>
</tr>
<tr>
<td align="center">VPG</td>
<td align="center">Time</td>
<td align="center">Morphological features related to fiducial point locations and amplitudes</td>
</tr>
<tr>
<td align="center">APG</td>
<td align="center">Time</td>
<td align="center">Morphological features related to fiducial point locations and amplitudes</td>
</tr>
<tr>
<td align="center" rowspan="3">ACC</td>
<td align="center">Frequency</td>
<td align="center">Mean, median, standard deviation, min, max, range, mean absolute deviation, median absolute deviation, interquartile range, skewness, kurtosis, energy, entropy of fft signal; fft-peak related features and signal power</td>
</tr>
<tr>
<td align="center">Statistical</td>
<td align="center">Mean, median, standard deviation, min, max, range, mean absolute deviation, median absolute deviation, interquartile range, skewness, kurtosis, energy, momentum of ACC signals; peak related features</td>
</tr>
<tr>
<td align="center">Correlation</td>
<td align="center">Correlation of ACC signals of different axes</td>
</tr>
<tr>
<td align="center" rowspan="4">EDA</td>
<td align="center">Time</td>
<td align="center">Rms, acr length, integral, average power</td>
</tr>
<tr>
<td align="center">Frequency</td>
<td align="center">FFT peak related features, energy, entropy of fft signal</td>
</tr>
<tr>
<td align="center">Statistical</td>
<td align="center">Mean, standard deviation, min, max, range, kurtosis, skewness, momentum</td>
</tr>
<tr>
<td align="center">Hjorth</td>
<td align="center">Activity, complexity, mobility</td>
</tr>
</tbody>
</table>

## <div align="left"> __Heart Rate Variability Analysis__ </div>
Heart rate variability analysis can be performed with BIOBSS and the parameters given below can be calculated for PPG or ECG signals.

<table>
<thead>
<tr>
<th style="text-align:center">Domain</th>
<th style="text-align:center">Parameters</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">Time-domain</td>
<td align="center">mean_nni, sdnn, rmssd, sdsd, nni_50, pnni_50, nni_20, pnni_20, cvnni, cvsd, median_nni, range_nni mean_hr, min_hr, max_hr, std_hr, mad_nni, mcv_nni, iqr_nni</td>
</tr>
<tr>
<td align="center">Frequency-domain</td>
<td align="center">vlf, lf, hf, lf_hf_ratio, total_power, lfnu, hfnu, lnLF, lnHF, vlf_peak, lf_peak, hf_peak</td>
</tr>
<tr>
<td align="center">Nonlinear</td>
<td align="center">SD1, SD2, SD2_SD1, CSI, CVI, CSI_mofidied, ApEn, SampEn</td>
</tr>
</tbody>
</table>

## <div align="left"> __Activity Indices__ </div>
BIOBSS has functionality to calculate activity indices from 3-axis acceleration signals. These indices are:
- Proportional Integration Method (PIM)
- Zero Crossing Method (ZCM)
- Time Above Threshold (TAT)
- Mean Amplitude Deviation (MAD)
- Euclidian Norm Minus One (ENMO)
- High-pass Filtered Euclidian (HFEN)
- Activity Index (AI)

Reference: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0261718

The preprocessing steps which should be applied on the raw acceleration differs for each of the activity indices listed above. In other words, each activity index can be calculated only from specific datasets. These datasets can be generated using BIOBSS both independently or as a part of activity index calculation pipeline.

The generated datasets are:
- UFXYZ: unfiltered acc signals 
- UFM: magnitude of unfiltered acc signals 
- UFM_modified: modified magnitude of unfiltered signals (absolute(UFM-length(UFM)))
- UFNM: normalized magnitude of unfiltered acc signals 
- FXYZ: filtered acc signals
- FXYZ_modified: modified filtered acc signals (absolute(FXYZ))   
- FMpre: magnitude of filtered acc signals
- SpecialXYZ: filtered acc signals (special filter parameters)  
- SpecialM: magnitude of filtered acc signals (special filter parameters)
- FMpost: filtered magnitude of acc signals
- FMpost_modified: modified of filtered magnitude of acc signals (absolute(FMpost))

## <div align="left"> __Respiratory Analysis__ </div>
BIOBSS has modules to perform basic respiratory analyses. The functionalities are:
- Preprocessing PPG or ECG signals for respiratory analysis using predefined filter parameters
- Extracting respiratory signals from modulations (amplitude modulation, frequency modulation, baseline wander) in PPG or ECG signals
- Estimating respiratory rate from the extracted respiratory signals
- Calculation respiratory quality indices (RQI)
- Fusing respiratory rate estimates 


## <div align="left"> __Pipeline Generation__ </div>

The main focus of BIOBSS is to generate and save pipelines for signal processing and feature extraction problems. Thus, it is aimed to :
- Simplify the preprocessing procedures by generating signal and event channels
- Make it easy to use processes 
- Decrease the amount of work for repetitive processes and for those who work on multiple datasets
- Make it possible to save and share pipelines to compare results of different works


To learn more, visit the [Documentation page](https://biobss.readthedocs.io/en/latest/).


## <div align="center"> Installation </div>

Through pip,

    pip install biobss

or build from source,

    git clone https://github.com/obss/biobss.git
    cd BIOBSS
    python setup.py install

## <div align="center"> Dependencies </div> 

- neurokit2
- antropy
- cvxopt
- heartpy
- scipy
- py_ecg_detectors


## <div align="center"> Tutorial notebooks </div>

- [PPG Signal Processing](https://github.com/obss/BIOBSS/blob/main/examples/ppg_processing.ipynb)
- [ECG Signal Processing](https://github.com/obss/BIOBSS/blob/main/examples/ecg_processing.ipynb)
- [ACC Signal Processing](https://github.com/obss/BIOBSS/blob/main/examples/acc_processing.ipynb)
- [HRV Analysis](https://github.com/obss/BIOBSS/blob/main/examples/hrv_analysis.ipynb)
- [Respiratory Analysis](https://github.com/obss/BIOBSS/blob/main/examples/respiratory_analysis.ipynb)


## <div align="center"> License </div>

Licensed under the [MIT](LICENSE) License.


## <div align="center"> Contributing </div>

If you have ideas for improving existing features or adding new features to BIOBSS, please contact us. 


## <div align="center"> Contributors </div>
[Çağatay Taşcı](https://github.com/tascic)

[İpek Karakuş](https://github.com/karakusipek)

[Devrim Çavuşoğlu](https://github.com/devrimcavusoglu)

[Fatih Çağatay Akyön](https://github.com/fcakyon)

