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
| ECG        |                    | Morphological features related to fiducial point locations and amplitudes |  
| PPG        | Time               | Morphological features related to fiducial point locations and amplitudes, zero-crossing rate, signal to noise ratio |
| ^^        | Frequency          | Amplitude and frequency of FFT peaks, signal power|
| ^^        | Statistical        | Mean, median, standard deviation, percentiles, mean absolute deviation, skewness, kurtosis, entropy |
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
| Time-domain       |                |   
| Frequency-domain  |                |  
| Nonlinear         |                |



## <div align="left"> Activity Indices </div>


## <div align="left"> Respiratory Analysis </div>

To learn more, visit the [Documentation page](biobss.readthedocs.io/en/latest/).

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
- [PPG Pipeline Generation](https://github.com/obss/BIOBSS/blob/main/examples/ppg_pipeline.ipynb)
- [ECG Signal Processing](https://github.com/obss/BIOBSS/blob/main/examples/ecg_processing.ipynb)
- [ECG Pipeline Generation]()
- [EDA Pipeline Generation](https://github.com/obss/BIOBSS/blob/main/examples/gsr_pipeline.ipynb)
- [ACC Signal Processing](https://github.com/obss/BIOBSS/blob/main/examples/acc_processing.ipynb)
- [ACC Pipeline Generation]()
- [HRV Analysis](https://github.com/obss/BIOBSS/blob/main/examples/hrv_analysis.ipynb)
- [Respiratory Analysis](https://github.com/obss/BIOBSS/blob/main/examples/respiratory_analysis.ipynb)
- [Custom Pipeline Generation]()


## <div align="center"> Citation </div>

If you use this package in your work, please cite it as:

    @software{obss2022biobss,
      author       = {},
      title        = {{BIOBSS}},
      month        = {},
      year         = {2022},
      publisher    = {},
      doi          = {},
      url          = {}
    }

## <div align="center"> License </div>

Licensed under the [MIT](LICENSE) License.

## <div align="center"> Contributing </div>

If you have ideas for improving existing features or adding new features to BIOBSS, please [follow the instructions](). 

## <div align="center"> Contributors </div>
Çağatay Taşcı

İpek Karakuş