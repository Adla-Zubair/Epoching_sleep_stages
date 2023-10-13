#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 14:17:12 2023


testing for sleep data non linear code !!!!!
"""

#%%
# Importing modules
import glob
import mne
import yasa
import ccstools
import pandas as pd
import numpy as np
import os 
from mne.annotations import read_annotations
from ccstools.yasafeatures import compute_psd_features,compute_irasa_features,compute_fooof_features,compute_nonlinear_features
from ccstools.yasafeatures import compute_nonlinear_features
from ccstools.yasafeatures import generate_single_epoch_features

#%% Dataset ready !!


#selecting EEG file of session 1 of all subjects
# Parent path
root_dir_eegfile = r"/serverdata/ccshome/ravindra/NAS/CCS_AdminFolders/Nphy_Sleep_Full_Data_SATYAM/Sleep_data/AllEEGFiles"

# Checking the files
for path in glob.glob(f'{root_dir_eegfile}/**/ses-1/*/*.edf', recursive=True):
    print("File Name: ", path.split('\\')[-1])
    print("Path: ", path)

# Files in a list
eegfiles = glob.glob(f'{root_dir_eegfile}/**/ses-1/*/*.edf')

#selecting hypnogram of session 1 of all subjects
# Parent path
root_dir_hypnogram = r"/serverdata/ccshome/ravindra/NAS/CCS_AdminFolders/Nphy_Sleep_Full_Data_SATYAM/Sleep_data/derivatives"

# Checking the files
for path in glob.glob(f'{root_dir_hypnogram}/**/ses-1/*/*.edf', recursive=True):
    print("File Name: ", path.split('\\')[-1])
    print("Path: ", path)
     
# Files in a list
hypnograms = glob.glob(f'{root_dir_hypnogram}/**/ses-1/*/*.edf')




#%% Hypnogram of all the subjects



for files in range(5):   # range(len(hypnograms))
    
    # Opening a UI asking to select scored EDF files
    hypno_data = mne.read_annotations(hypnograms[files])
    
    # onset column is start of an epoch and duration tells us how long
    hypnogram_annot = hypno_data.to_data_frame()
    
    # change the duration column into epochs count
    hypnogram_annot.duration = hypnogram_annot.duration/30
    
    # convert the onset column to epoch number
    timestamps = hypnogram_annot.onset.dt.strftime("%m/%d/%Y, %H:%M:%S")
    
    only_time = []
    for entries in timestamps:
        times = entries.split()[1]
        only_time.append(times.split(':'))
    
    # converting hour month and seconds as epoch number
    epochs_start = []
    for entries in only_time:
        hh = int(entries[0]) * 120
        mm = int(entries[1]) * 2
        ss = int(entries[2])/ 30
        epochs_start.append(int(hh+mm+ss))
    
    # replacing the onset column with start of epoch
    hypnogram_annot['onset'] = epochs_start
    
    # keep the description column neat
    just_labels = []
    
    # Building a check for any other marker in hypnogram file
    # Spotted movement marker in some files
    # Checking  there are movement markers and converting them to W marker
    
    # In the below for and if loop block, we can add any other anomalous marker
    # Currently we have acccounted only for Movement time into W
    # Use elif to add more criteria
    for entries in hypnogram_annot.description:
        if entries == 'Movement time':
            just_labels.append('W')
        else:
            just_labels.append(entries.split()[2]) # Some sleep files had Sleep Stage W, hence picked 2nd entry (third)
    
    # replacing the description column with just_labels
    hypnogram_annot['description'] = just_labels
    
    # we need only the duration column and description column to recreate hypnogram
    # just reapeat duration times the label in description column
    hypno_30s = []
    for stages in range(len(hypnogram_annot)):
        for repetitions in range(int(hypnogram_annot.duration[stages])):
            hypno_30s.append(hypnogram_annot.description[stages])
    
    # converting list to numpy array
    hypno_30s = np.asarray(hypno_30s)
    
    # converting string array into int array using yasa
    # hypno_30s = yasa.hypno_str_to_int(hypno_30s)
    
    fname = os.path.basename(hypnograms[files])[:6] + '.csv'
    
    # spit out the sleep stage sequences as a text file
    np.savetxt(fname,
               hypno_30s,
               delimiter = ',',
               fmt='%s') # you may change this to s for string
    
    # converting W and R into 0 and 4 | In case any renaming is needed
    hypno_30s = [s.replace('W', '0') for s in hypno_30s]
    hypno_30s = [s.replace('R', '4') for s in hypno_30s]
    hypno_30s = [s.replace('N1', '1') for s in hypno_30s]
    hypno_30s = [s.replace('N2', '2') for s in hypno_30s]
    hypno_30s = [s.replace('N3', '3') for s in hypno_30s]
    
    # Plot the hypnogram and save as image
    # plot hypnogram
    # yasa.plot_hypnogram(hypno_30s,lw = 1,figsize=(25, 2.5))
    # plt.tight_layout()
    # plt.savefig(os.path.basename(scored_files[files])[:-4] + '_hypno.png',dpi = 600)
    # plt.close()
    
    
    print(files)
    




#%%Epoching the dataset

# For concatenating all the epochs into one list
epoch_list = []
master_eventinfo = pd.DataFrame()
#for testing
eegfiles_test = eegfiles[0:5]    # selected for only 5 subjects
hypnograms_test = hypnograms[0:5]


for i in range(len(hypnograms_test)):                  # change to hypnograms for running entire dataset
    
    # Loading the raw EEG data
    eeg = mne.io.read_raw_edf(eegfiles[i], preload= True)
    
    # Dropping unused channels
    eeg.drop_channels(['X1','X2','X3',    'X4',    'X5',    'X6',    'X7',    'SpO2',    'EtCO2',    'DC03',    'DC04',    'DC05',    'DC06',    'Pulse',    'CO2Wave','EEG Mark1', 'EEG Mark2',    'Events/Markers'])
    # Loading the name of the subject
    fname = os.path.basename(hypnograms[i])[:6] + '.csv'
    
    # Loading the scored data
    annotations = read_annotations(hypnograms[i])

    # Adding the annotations to the EEG file
    eeg.set_annotations(annotations)
    
    # Calling the annotations from EEG into a separate event file
    events  = mne.events_from_annotations(eeg)
    
    # Epoching the EEG data around the annotations as an example
    epochs = mne.Epochs(eeg, events[0], tmin= 0, tmax= 29, baseline=(0, 0))
    epoch_list.append(epochs)
    
    # Creating a dataframe of event markers
    x = pd.DataFrame(events[0])
    x['subjname'] = fname[:6]

    # Creating event marker containing epoch time point epoch no sleep stage subject name
    master_eventinfo =  master_eventinfo.append(x)

    
# Concatenate the epochs
conc_epochs = mne.concatenate_epochs(epoch_list,add_offset=True,on_mismatch='ignore')

# Tweaking the master_eventinfo dataframe
master_eventinfo = master_eventinfo.reset_index()
master_eventinfo.rename(columns = {'index':'Epoch_subj'}, inplace = True)
master_eventinfo.rename(columns = {0:'Time_points'}, inplace = True)
master_eventinfo.rename(columns = {2:'Sleep_stage'}, inplace = True)
master_eventinfo.drop(1, axis=1, inplace=True)



#%%


# Defining bands for feature extraction
bands=[(1, 4, 'Delta'), (4, 8, 'Theta'), (6, 10, 'ThetaAlpha'),
       (8, 12, 'Alpha'), (12, 18, 'Beta1'),(18, 30, 'Beta2'), (30, 40, 'Gamma')]


chanlist = conc_epochs.ch_names
n_epochs,n_chans,_  = conc_epochs._data.shape
srate               = conc_epochs.info.get('sfreq')
winsize = 1
all_eegfeatures = pd.DataFrame()
# Generating multivariate features from smaller epochs
# Loop performed per epoch 

    
for epochno in range(3):
    try:
        epochdata = mne.io.RawArray(conc_epochs[epochno]._data[0], conc_epochs.info)
        data = epochdata._data*1e6
        print("epoch no :" ,epochno)
        # Generating the features per epoch per channel   
        for chan_no in range(n_chans):         
            #epochfeatures = generate_single_epoch_features(np.expand_dims(data[chan_no],axis=0),srate,winsize,bands=bands)
            epochfeatures = compute_nonlinear_features(np.expand_dims(data[chan_no],axis=0))
            # The event info also incorporated into the dataset
            epochfeatures["Channel"] = chanlist[chan_no]
            epochfeatures["subjname"] = master_eventinfo["subjname"][epochno]
            epochfeatures["Stage"] = master_eventinfo['Sleep_stage'][epochno]
            epochfeatures['epochno'] = epochno
            # Concatenating the feactures into a single dataset
            all_eegfeatures = all_eegfeatures.append(epochfeatures)
            print(chanlist[chan_no])
    except:
        print(" Error occured :(")

  


#%%


































