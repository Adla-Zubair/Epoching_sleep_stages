# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 15:03:22 2023

@author: WS3
"""

#%%
# Importing modules
import glob
import mne
from mne.annotations import read_annotations

#%% Dataset ready !!


#selecting EEG file of session 1 of all subjects
# Parent path
root_dir_eegfile = r"/path/to/AllEEGFiles"

# Checking the files
for path in glob.glob(f'{root_dir_eegfile}/**/ses-1/*/*.edf', recursive=True):
    print("File Name: ", path.split('\\')[-1])
    print("Path: ", path)

# Files in a list
eegfiles = glob.glob(f'{root_dir_eegfile}/**/ses-1/*/*.edf')

#selecting hypnogram of session 1 of all subjects
# Parent path
root_dir_hypnogram = r"/path/to/derivatives"

# Checking the files
for path in glob.glob(f'{root_dir_hypnogram}/**/ses-1/*/*.edf', recursive=True):
    print("File Name: ", path.split('\\')[-1])
    print("Path: ", path)
     
# Files in a list
hypnograms = glob.glob(f'{root_dir_hypnogram}/**/ses-1/*/*.edf')


#%% For single subject


# selecting 1 file for testing
eegfile_s1 =  eegfiles[0]
hypnogram_s1 = hypnograms[0]

# Loading the raw EEG data
eeg = mne.io.read_raw_edf(eegfile_s1, preload= True)

# Loading the scored data
annotations = read_annotations(hypnogram_s1)

# Adding the annotations to the EEG file
eeg.set_annotations(annotations)

# Calling the annotations from EEG into a separate event file
events  = mne.events_from_annotations(eeg)

# Epoching the EEG data around the annotations as an example
epochs = mne.Epochs(eeg, events[0], tmin=-1, tmax=1)

#%%Same code as a loop for all subjects

# For concatenating all the epochs into one list
epoch_list = []


for i in range(len(hypnograms)):
    
    # Loading the raw EEG data
    eeg = mne.io.read_raw_edf(eegfiles[i], preload= True)
    
    # Loading the scored data
    annotations = read_annotations(hypnograms[i])
    
    # Adding the annotations to the EEG file
    eeg.set_annotations(annotations)
    
    # Calling the annotations from EEG into a separate event file
    events  = mne.events_from_annotations(eeg)
    
    # Epoching the EEG data around the annotations as an example
    epochs = mne.Epochs(eeg, events[0], tmin=-1, tmax=1)
    
    epoch_list.append(epochs)

#%%