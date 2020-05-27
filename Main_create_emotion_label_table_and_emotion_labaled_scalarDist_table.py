# import models:
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
# other imports:
from pathlib import Path
from CordioESP import CordioESP_ToolBox
import os, glob
from progressbar import *
import sys
import warnings
from tkinter import Tk
from tkinter import filedialog
import easygui
import numpy as np

sys.path.insert(0, "D:\work\code\Python\CordioAlgorithmsPython\infrastructure")

from patientsInformation.CordioPatientClinicalInformation import CordioClinicalInformation
import CordioFile

warnings.filterwarnings('ignore')

# get current run information from user:
# -------------------------------------
root = Tk()
root.withdraw()  # use to hide tkinter window
currdir = os.getcwd()

msg = "Please Enter Your Preference"
title = "setup run"
fieldNames = ["setup defenition name(string):", 'sentence ID list(S0007,S0010)', "number of patients(1,2,...):", "add datetime to output(y/n):", "overwrite existing tables(y/n):"]
fieldValues = ['no_setup_defined', 'S0007', '1', 'y', 'n']  # we start with blanks for the values
fieldValues = easygui.multenterbox(msg, title, fieldNames)
setup_name = fieldValues[0]
fieldValues[1] = fieldValues[1].replace(" ", "")
sentence_list = fieldValues[1].split(',')
num_of_patients = int(fieldValues[2])
add_datetime = fieldValues[3] == 'y'
overwrite_tables = fieldValues[4] == 'y'

patientDir_paths = [np.nan] * num_of_patients
all_scalarDistTable = [np.nan] * num_of_patients
for i in range(num_of_patients):
    patientDir_paths[i] = filedialog.askdirectory(parent=root, initialdir=currdir,
                                           title='Please select the ' + str(i+1) + ' patient dir directories')
    curr_patient_ID = os.path.basename(os.path.normpath(patientDir_paths[i]))
    all_scalarDistTable[i] = filedialog.askopenfiles(parent=root, initialdir=currdir,
                                                    title='Please select a the scalar dist table for '+curr_patient_ID)[0]
output_dir = filedialog.askdirectory(parent=root, initialdir=currdir, title='Please select a the output directory')

all_scalarDistTable = [x.name for x in all_scalarDistTable]

# getting tools ready:
# -------------------
# from CordioESP import CordioESP_ToolBox
ESP_TB = CordioESP_ToolBox()
clinicalInformation = CordioClinicalInformation
fileHandle = CordioFile

# create tables:
# ----------------

# progress bar initialization:
widgets = [FormatLabel('<<<all patient process>>>'), ' ', Percentage(), ' ', Bar('#'), ' ', RotatingMarker()]
progressbar = ProgressBar(widgets=widgets, maxval=len(patientDir_paths))
progressbar.start()

# get all existing urls in output dir:
all_avalble_csvs = glob.glob(output_dir + "//**//*.csv", recursive=True)

for i, patientDir_path in zip(range(len(patientDir_paths)), patientDir_paths):
    # progress bar update:
    widgets[0] = FormatLabel('<filename-{0}>'.format(i))
    progressbar.update(i)
    # table per model:
    for model in ESP_TB.model_list:
        emotions_list = ESP_TB.model_emotion_dict[type(model).__name__]
        # set table path and name:
        all_wavs = glob.glob(os.path.join(patientDir_path, '*.wav'))
        p = Path(str(all_wavs[0]))
        patient_ID = fileHandle.CordioExtractPatient(p)
        save_url_path = output_dir + "\\" + setup_name + "\\" + patient_ID + "\\" + type(model).__name__
        for sentence in sentence_list:
            # save_file_name = ESPResults_model_version_patientId_SentanceId_date_time
            save_file_name = 'ESPResults_' + type(model).__name__ + '_V' + str(
                ESP_TB.version) + '_' + patient_ID + '_' + sentence
            # check for existing table:
            is_exist = [patient_prob_tables_urls for patient_prob_tables_urls in all_avalble_csvs if
                        save_file_name in patient_prob_tables_urls]
            if not ((is_exist != []) and (overwrite_tables == False)):
                # calculate emotion table:
                table = ESP_TB.create_ESP_patient_model_sentence_labeled_table(patientDir_path, model, sentence,
                                                                               clinicalInformation, fileHandle)
                # save table:
                ESP_TB.SaveTable(table, save_url_path, save_file_name, add_datetime=False)
                # update all existing urls in output dir:
                all_avalble_csvs = glob.glob(output_dir + "//**//*.csv", recursive=True)

            # calculating and saving filtered scalar_dist_tablels with emotion labeling:
            # -------------------------------------------------------------------------
            # getting scalar_dist_table_url:
            scalar_dist_table_url = list(filter(lambda x: 'Scalar' in x, all_scalarDistTable))
            scalar_dist_table_url = list(filter(lambda x: patient_ID in x, scalar_dist_table_url))
            scalar_dist_table_url = list(filter(lambda x: sentence in x, scalar_dist_table_url))[0]
            # getting emo_labeled_data_table_url:
            emo_labeled_data_table_url = list(filter(lambda x: 'ESPResults' in x, all_avalble_csvs))
            emo_labeled_data_table_url = list(filter(lambda x: patient_ID in x, emo_labeled_data_table_url))
            emo_labeled_data_table_url = list(filter(lambda x: type(model).__name__ in x, emo_labeled_data_table_url))
            emo_labeled_data_table_url = list(filter(lambda x: sentence in x, emo_labeled_data_table_url))[0]
            # create the relevant table:
            updated_scalar_distance_table = ESP_TB.manipulate_scalar_distance_table(scalar_dist_table_url,
                                                                                    emo_labeled_data_table_url)
            # save table:
            save_file_name = 'ESP_' + os.path.split(scalar_dist_table_url)[-1][0:-4] + '_' + type(model).__name__
            ESP_TB.SaveTable(updated_scalar_distance_table, save_url_path, save_file_name, add_datetime=True)

progressbar.finish()
