# <div align="center"> BIOBSS </div>

A package for processing signals recorded using wearable sensors, such as Electrocardiogram (ECG), Photoplethysmogram (PPG), Electrodermal activity (EDA) and 3-axis acceleration (ACC). 

BIOBSS's main focus is to generate end-to-end pipelines by adding appropriate processes from BIOBSS or other Python packages. Some preprocessing methods were not implemented from scratch but imported from the existing packages listed in [dependencies]().

Main features:

- Applying basic preprocessing steps 
- Extracting features for ECG, PPG, EDA and ACC signals
- Assessing quality of PPG and ECG signals
- Performing Heart Rate Variability (HRV) analysis using PPG or ECG signals
- Extracting respiratory signals from PPG or ECG signals and estimating respiratory rate
- Calculating activity indices from ACC signals
- Generating and saving pipelines 


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
- py_ecg_detectors


## <div align="center"> Tutorial notebooks </div>

- [PPG Signal Processing]()
- [PPG Pipeline Generation]()
- [ECG Signal Processing]()
- [ECG Pipeline Generation]()
- [EDA Pipeline Generation]()
- [ACC Signal Processing]()
- [ACC Pipeline Generation]()
- [HRV Analysis]()
- [Respiratory Analysis]()
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
TEXT