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