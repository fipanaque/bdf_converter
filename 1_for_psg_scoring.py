# ACCEPT EDF FILE FROM BIOSEMI CONVERTER SOFTWARE
#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
from mne.io import read_raw_bdf
import mne
import pandas as pd
import numpy as np
from collections import Counter
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import glob
import re
import sys

# In[10]:


# Linux

PATH = "/work/SLEEP/fipanaque/nextcloud/data_chs_reduce/"
#PATH_1 = "/work/SLEEP/fipanaque/nextcloud/"
CP_PATH = "/work/SLEEP/fipanaque/nextcloud/checkpoints/"
# windows
#PATH = r"C:\\Users\\_s2030535\\Documents\\data\\"
#CP_PATH = r"C:\\Users\\_s2030535\\Documents\\checkpoints\\"

_subject = ["s0644", "s0645", "s0654", "s0655", "s0656", "s0657", "s0658"][int(sys.argv[1])]  #"s0646", "s0646"
_session = [1,2][int(sys.argv[2])]

eeg_chs = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30', 'A31', 'A32', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20', 'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B28', 'B29', 'B30', 'B31', 'B32', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C30', 'C31', 'C32', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32','EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Resp', 'Plet']

mapping_2 = {'A1': 'A1 (Cz)','A2': 'A2','A3': 'A3 (CPz)','A4': 'A4','A5': 'A5 (approx: P1)','A6': 'A6','A7': 'A7 (approx: P3)','A8': 'A8','A9': 'A9','A10': 'A10 (PO7)','A11': 'A11','A12': 'A12','A13': 'A13','A14': 'A14','A15': 'A15 (O1)','A16': 'A16','A17': 'A17 (approx: PO3)','A18': 'A18 (approx: CMS)','A19': 'A19 (Pz)','A20': 'A20','A21': 'A21 (POz)','A22': 'A22','A23': 'A23 (Oz)','A24': 'A24','A25': 'A25 (Iz, Inion)','A26': 'A26','A27': 'A27','A28': 'A28 (O2)','A29': 'A29','A30': 'A30 (approx: PO4)','A31': 'A31 (approx: DRL)','A32': 'A32 (approx: P2)','B1': 'B1','B2': 'B2 (approx: CP2)','B3': 'B3','B4': 'B4 (approx: P4)','B5': 'B5','B6': 'B6','B7': 'B7 (PO8)','B8': 'B8','B9': 'B9','B10': 'B10 (approx: P10)','B11': 'B11 (P8)','B12': 'B12','B13': 'B13 (approx: P6)','B14': 'B14 (TP8)','B15': 'B15','B16': 'B16 (approx: CP6)','B17': 'B17','B18': 'B18 (approx: CP4)','B19': 'B19','B20': 'B20 (C2)','B21': 'B21','B22': 'B22 (C4)','B23': 'B23','B24': 'B24 (C6)','B25': 'B25','B26': 'B26 (T8)','B27': 'B27 (FT8)','B28': 'B28','B29': 'B29 (approx: FC6)','B30': 'B30','B31': 'B31 (approx: FC4)','B32': 'B32','C1': 'C1','C2': 'C2','C3': 'C3','C4': 'C4 (approx: F4)','C5': 'C5 (approx: F6)','C6': 'C6','C7': 'C7 (F8)','C8': 'C8 (AF8)','C9': 'C9','C10': 'C10','C11': 'C11 (approx: FC2)','C12': 'C12 (approx: F2)','C13': 'C13','C14': 'C14','C15': 'C15 (approx: AF4)','C16': 'C16 (Fp2)','C17': 'C17 (Fpz)','C18': 'C18','C19': 'C19 (AFz)','C20': 'C20','C21': 'C21 (Fz)','C22': 'C22','C23': 'C23 (FCz)','C24': 'C24 (approx: FC1)','C25': 'C25 (approx: F1)','C26': 'C26','C27': 'C27','C28': 'C28 (approx: AF3)','C29': 'C29 (Fp1)','C30': 'C30 (AF7)','C31': 'C31','C32': 'C32','D1': 'D1','D2': 'D2','D3': 'D3','D4': 'D4 (approx: F3)','D5': 'D5 (approx: F5)','D6': 'D6','D7': 'D7 (F7)','D8': 'D8 (FT7)','D9': 'D9','D10': 'D10 (approx: FC5)','D11': 'D11','D12': 'D12 (approx: FC3)','D13': 'D13','D14': 'D14 (C1)','D15': 'D15','D16': 'D16 (approx: CP1)','D17': 'D17','D18': 'D18','D19': 'D19 (C3)','D20': 'D20','D21': 'D21 (C5)','D22': 'D22','D23': 'D23 (T7)','D24': 'D24','D25': 'D25','D26': 'D26','D27': 'D27','D28': 'D28','D29': 'D29','D30': 'D30','D31': 'D31','D32': 'D32'}

new_sfreq = 256


# ### 1. Reading .bdf file + resampling

# In[4]:


raw = mne.io.read_raw_edf(os.path.join(PATH, _subject, "0{0}_session{0}".format(_session), "03_sleep_recording", "{0}.edf".format(_subject)), preload=True)
#print(raw.info["ch_names"])
raw.resample(new_sfreq)
sfreq = raw.info["sfreq"]


# In[53]:


events = mne.find_events(raw, stim_channel="Status")
print(events)


# In[54]:


annotations = mne.annotations_from_events(events,sfreq=256) #Convert an event array into an Annotations object.
raw.set_annotations(annotations)


#### Open 2nd file

# In[12]:

#PATH_2 = r"C:\\Users\\_s2030535\\Documents\\data\\s0658\\02_session2\\03_sleep_recording\\"
PATH_2 = os.path.join(PATH, _subject, "0{0}_session{0}".format(_session), "03_sleep_recording")
file_name= PATH_2+ "/{0}_1.edf".format(_subject)
raw1 = mne.io.read_raw_edf(file_name)

# In[9]:

raw1.resample(new_sfreq)


# In[57]:

events1 = mne.find_events(raw1, stim_channel="Status")
print(events1)
annotations1 = mne.annotations_from_events(events1,sfreq=256) #Convert an event array into an Annotations object.
raw1.set_annotations(annotations1)

#### Open 3rd file

# In[10]:

#PATH_2 = r"C:\\Users\\_s2030535\\Documents\\data\\s0658\\02_session2\\03_sleep_recording\\"
#PATH_2 = os.path.join(PATH, _subject, "0{0}_session{0}".format(_session), "03_sleep_recording")
#file_name= PATH_2+ "/{0}_1.bdf".format(_subject)
#raw2 = mne.io.read_raw_bdf(file_name)

#raw2.resample(new_sfreq)

##### Concatenating raw files

# In[58]:

raw3 = mne.io.concatenate_raws([raw,raw1], on_mismatch='ignore')
raw3.info


# In[59]:


events3 = mne.find_events(raw3, stim_channel="Status")
print(events3)


# In[60]:


events4 = events3[events3[:,-1]!=65536]
events4


# In[61]:


events5 = events4[1:][::2]
print(events5)


# In[62]:


annotations4 = mne.annotations_from_events(events5,sfreq=256) #Convert an event array into an Annotations object.
raw3.set_annotations(annotations4)


# In[63]:


raw3.notch_filter(freqs=50, filter_length='auto', phase='zero',trans_bandwidth=9, notch_widths=2, method='fir')


# Rename extra_channels
mapping = {'EXG1':'M1', 'EXG2':'M2', 'EXG3':'EMG_Center', 'EXG4':'EMG_left', 'EXG5':'EMG_right', 'EXG6':'EOG_under', 'EXG7':'EOG_left', 'EXG8':'EOG_right'}
raw3.rename_channels(mapping)


# Set channel types
ch_types = {'M1':'emg', 'M2':'emg', 'EMG_Center':'emg', 'EMG_left':'emg', 'EMG_right':'emg', 'EOG_under':'eog', 'EOG_left':'eog', 'EOG_right':'eog', 'GSR1':'gsr', 'GSR2':'gsr', 'Resp':'resp', 'Plet':'misc'}
raw3.set_channel_types(ch_types)

# Drop channels
drop = ['Erg1', 'Erg2', 'Temp']
raw3.drop_channels(drop, on_missing='raise')

#Set biosemi Montage
biosemi_montage = mne.channels.make_standard_montage("biosemi128")
raw3.set_montage(biosemi_montage)

#Selecting data for_psg_scoring
psg_info = ['C4','B22','A28', 'A15', 'D19','D4', 'M1', 'M2', 'EMG_Center', 'EMG_left', 'EMG_right', 'EOG_under', 'EOG_left', 'EOG_right', 'GSR1', 'GSR2', 'Resp', 'Plet']
data_4psg_score = raw3.pick_channels(ch_names=psg_info)

#Renaming EEG channels
mapping_eeg = {'C4':'F4','B22':'C4','A28':'O2', 'A15':'O1', 'D19':'C3','D4':'F3'}
data_4psg_score.rename_channels(mapping_eeg)

#Passing band_pass filter to EEG and EOG
data_4psg_score.filter(l_freq=None , h_freq=0.3, picks=['eeg','eog'], method="iir")
data_4psg_score.filter(l_freq=None, h_freq=10, picks=['M1','M2'], method="iir")

#Create bipolar references
ch_right = ['F4','C4','O2']
ch_left = ['F3','C3','O1','EOG_right','EOG_left']

mne.set_bipolar_reference(
    data_4psg_score,
    anode=ch_right,
    cathode=['M1']*len(ch_right),
    copy=False
)

mne.set_bipolar_reference(
    data_4psg_score,
    anode=ch_left,
    cathode=['M2']*len(ch_left),
    copy=False
)

#Getting epochs
epochs = mne.Epochs(data_4psg_score, events5, preload=True, tmin=-600, tmax=0, baseline=(None, 0))

#path2 = r"C:\\Users\\_s2030535\\Documents\\EMODEC_data_preprocess\\5_bandpass_filter\\"
path2 = "/work/SLEEP/fipanaque/nextcloud/EMODEC_data_preprocess/6_data_psg_scoring/"

for i in range(1,len(epochs)):
    data = epochs[i]
    pseudo_raw_data = np.hstack(data)
    pseudo_raw = mne.io.RawArray(pseudo_raw_data, epochs.info)
    file_name = path2 + "{0}".format(_subject) + "_sleep_s{0}".format(_session) + "_epoch{0}.edf".format(i)
    mne.export.export_raw(file_name, pseudo_raw, fmt='edf', overwrite=True, verbose=None)

#mne.export.export_raw(file_name, raw3, fmt='auto', physical_range='auto', add_ch_type=False, overwrite=True, verbose=None)

