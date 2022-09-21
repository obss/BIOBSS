# %%
import importlib
import sys
import pandas as pd
import numpy as np

sys.path.append("./")
import biobss


# %%
import matplotlib.pyplot as plt

# %%
import neurokit2 as nk
sample_data = pd.read_pickle(".\sample_data\\EDA_Chest.pkl")
sample_data = sample_data.flatten()

# %%
decompose = biobss.pipelinev2.Bio_Process(process_method=biobss.edatools.eda_decompose,method="highpass")

clean=biobss.pipelinev2.Bio_Process(
    process_method=biobss.edatools.eda_clean,argmap={"sampling_rate":"sampling_rate"},method="neurokit")

normalize = biobss.pipelinev2.Bio_Process(
    process_method=biobss.signaltools.normalize_signal)
resample = biobss.pipelinev2.Bio_Process(
    process_method=biobss.signaltools.resample_signal_object, modality="EDA", sigtype="EDA", target_sample_rate=350)
signal_features = biobss.pipelinev2.Feature(name="signal_features", function=biobss.edatools.signal_features.get_signal_features, parameters={
                                           "modality": "EDA", "sigtype": "EDA"}, input_signals={'EDA_Raw':'EDA_Raw', 'EDA_Tonic':'EDA_Tonic', 'EDA_Phasic':'EDA_Phasic'})
stat_features = biobss.pipelinev2.Feature(name="stat_features", function=biobss.common.stat_features.get_stat_features, parameters={
                                         "modality": "EDA", "sigtype": "EDA"}, input_signals={'EDA_Raw':'EDA_Raw', 'EDA_Tonic':'EDA_Tonic', 'EDA_Phasic':'EDA_Phasic'})
corr_features = biobss.pipelinev2.Feature(name="corr_features", function=biobss.common.correlation_features, parameters={
                                        "modality": "EDA", "sigtype": "EDA","signal_names":['EDA_Raw','EDA_Tonic','EDA_Phasic']}, input_signals={'EDA':['EDA_Raw','EDA_Tonic','EDA_Phasic']})

# %%
pipeline = biobss.pipelinev2.Bio_Pipeline()

# %%
channel=biobss.pipelinev2.Bio_Channel(signal=sample_data,name="EDA_Raw",sampling_rate=700)

# %%
channel

# %%
pipeline.set_input(sample_data,sampling_rate=700,name='EDA_Raw')

# %%
pipeline.preprocess_queue.add_process(clean)
pipeline.preprocess_queue.add_process(normalize)
pipeline.preprocess_queue.add_process(resample)
pipeline.preprocess_queue.add_process(decompose)

# %%
pipeline.run_pipeline()

# %%
eda_data = biobss.pipelinev2.Bio_Channel(sample_data, sampling_rate=700, name="EDA_Raw",timestamp_resolution='s')

# %%


# %%
plt.plot(clean_eda)

# %%
plt.plot(eda_data.timestamp,eda_data.channel)
plt.ylabel("EDA_Raw(\u03BCs)")
plt.xlabel("Seconds")
plt.suptitle("EDA_Raw")

# %%
eda_decomposed=decompose.process(eda_data)

# %%
plt.plot(eda_decomposed['EDA_Tonic'].timestamp,eda_decomposed['EDA_Tonic'].channel)
plt.ylabel("EDA_Tonic(\u03BCs)")
plt.xlabel("Seconds")
plt.suptitle("EDA_Tonic")


# %%
plt.plot(eda_decomposed['EDA_Phasic'].timestamp,eda_decomposed['EDA_Phasic'].channel)
plt.ylabel("EDA_Tonic(\u03BCs)")
plt.xlabel("Seconds")
plt.suptitle("EDA_Phasic")



