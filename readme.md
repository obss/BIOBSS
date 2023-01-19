# <div align="center"> BIOBSS </div>

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

| Package      | File reader | Sliding window | Preprocessing |   ECG   |   PPG   |   IBI   |   EDA   |   ACC   |  Pipeline  |
| ------------ | ----------- | -------------- | ------------- | ------- | ------- | ------- | --------| ------- | ---------- |
| BioSPPy      |             |                |               |         | &check; | &check; | &check; |         |            |
| HeartPy      |             |                |               |         | &check; | &check; |         |         |            |
| HRV          | &check;     |                | &check;       |         |         | &check; |         |         |            |
| hrv-analysis |             |                |               |         |         | &check; |         |         |            |
| pyHRV        |             |                |               |         | &check; | &check; |         |         |            |
| PyPhysio     |             | &check;        |               |         |         | &check; | &check; |         |            |
| PySiology    |             |                |               |         | &check; | &check; | &check; |         |            |
| FLIRT        | &check;     | &check;        | &check;       |         | &check; | &check; | &check; |         |            |
| Neurokit2    |             |                | &check;       |         | &check; | &check; | &check; | &check; | &check;    |
| BIOBSS       |             | &check;        | &check;       | &check; | &check; | &check; | &check; | &check; | &check;(*) |

(*): 


## <div align="left"> Preprocessing </div>
BIOBSS has modules with basic signal preprocessing functionalities. These include:
- Resampling
- Segmentation
- Normalization
- Filtering (basic filtering functions with commonly used filter parameters for each signal type)
- Peak detection 

## <div align="left"> Visualization </div>
BIOBSS has basic plotting modules specific to each signal type. Using the modules, the signals and peaks can be plotted using Matplotlib or Plotly packages.

## <div align="left"> Signal Quality Assessment </div>
Signal quality assessment steps listed below can be used with PPG and ECG signals.
- Clipping detection
- Flatline detection
- Physiological checks
- Morphological checks
- Template matching

## <div align="left"> Feature Extraction </div>
|   Signal   |   Domain / Type    |   Features   |   
| ---------- | ------------------ | ------------ |
| ECG        |  Time              | Morphological features related to fiducial point locations and amplitudes |  
| PPG        | Time               | Morphological features related to fiducial point locations and amplitudes, zero-crossing rate, signal to noise ratio |
| PPG        | Frequency          | Amplitude and frequency of FFT peaks, signal power|
| PPG        | Statistical        | Mean, median, standard deviation, percentiles, mean absolute deviation, skewness, kurtosis, entropy |
| VPG        | Time               | Morphological features related to fiducial point locations and amplitudes |
| APG        | Time               | Morphological features related to fiducial point locations and amplitudes |
| ACC        | Frequency          | Mean, median, standard deviation, min, max, range, mean absolute deviation, median absolute deviation, interquartile range, skewness, kurtosis, energy, entropy of fft signal; fft-peak related features and signal power |
| ACC        | Statistical        | Mean, median, standard deviation, min, max, range, mean absolute deviation, median absolute deviation, interquartile range, skewness, kurtosis, energy, momentum of ACC signals; peak related features |
| ACC        | Correlation        | Correlation of ACC signals of different axes | 
| EDA        | Time               | Rms, acr length, integral, average power              |        
| EDA        | Frequency          | FFT peak related features, energy, entropy of fft signal             | 
| EDA        | Statistical        | Mean, standard deviation, min, max, range, kurtosis, skewness, momentum             |
| EDA        | Hjorth             | Activity, complexity, mobility              | 


## <div align="left"> Heart Rate Variability Analysis </div>
Heart rate variability analysis can be performed with BIOBSS and the parameters given below can be calculated for PPG or ECG signals.

|   Domain          |   Parameters   |
| ----------        | -------------- |
| Time-domain       | mean_nni, sdnn, rmssd, sdsd, nni_50, pnni_50, nni_20, pnni_20, cvnni, cvsd, median_nni, range_nni mean_hr, min_hr, max_hr, std_hr, mad_nni, mcv_nni, iqr_nni |   
| Frequency-domain  | vlf, lf, hf, lf_hf_ratio, total_power, lfnu, hfnu, lnLF, lnHF, vlf_peak, lf_peak, hf_peak               |  
| Nonlinear         | SD1, SD2, SD2_SD1, CSI, CVI, CSI_mofidied, ApEn, SampEn               |


## <div align="left"> Activity Indices </div>
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

## <div align="left"> Respiratory Analysis </div>
BIOBSS has modules to perform basic respiratory analyses. The functionalities are:
- Preprocessing PPG or ECG signals for respiratory analysis using predefined filter parameters
- Extracting respiratory signals from modulations (amplitude modulation, frequency modulation, baseline wander) in PPG or ECG signals
- Estimating respiratory rate from the extracted respiratory signals
- Calculation respiratory quality indices (RQI)
- Fusing respiratory rate estimates 

To learn more, visit the [Documentation page](biobss.readthedocs.io/en/latest/).

 

## <div align="left"> Pipeline Generation </div>

The main focus of BIOBSS is to generate and save pipelines for signal processing and feature extraction problems. Thus, it is aimed to :
- Simplify the preprocessing procedures by generating signal and event channels
- Make it easy to use processes 
- Decrease the amount of work for repetitive processes and for those who work on multiple datasets
- Make it possible to save and share pipelines to compare results of different works


