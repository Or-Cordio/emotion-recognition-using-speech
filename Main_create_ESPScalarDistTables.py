# # import models:
# from sklearn.svm import SVC
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
# # other imports:
# from pathlib import Path
# from CordioESP import CordioESP_ToolBox
# import os, glob
# from progressbar import *
# import sys
# import warnings
# from tkinter import Tk
# from tkinter import filedialog
# import easygui
# import numpy as np
# import pandas as pd
# from itertools import compress
#
# sys.path.insert(0, "D:\work\code\Python\CordioAlgorithmsPython\infrastructure")
#
# from patientsInformation.CordioPatientClinicalInformation import CordioClinicalInformation
# import CordioFile
#
# warnings.filterwarnings('ignore')
#
# # getting tools ready:
# # -------------------
# # from CordioESP import CordioESP_ToolBox
# ESP_TB = CordioESP_ToolBox()
# clinicalInformation = CordioClinicalInformation
# fileHandle = CordioFile
#
# # get current run information from user:
# # -------------------------------------
#
# message = "get input data from csv?"
# title = "Cordio ESP"
# input_data_from_csv = easygui.boolbox(message, title, ["Yes", "No (insert manually)"])
#
# if input_data_from_csv:
#     input_data_from_csv_url = easygui.fileopenbox('please select input csv file')
#     input_df = pd.read_csv(input_data_from_csv_url)
#     input_df = input_df[input_df["toRun"] == 1]  # filter data
#     db_url = easygui.diropenbox('please select database folder')
#     # all_db_csvs = glob.glob(db_url + "//**//*.csv", recursive=True)
#     for patientID, sentence in zip(input_df["patientID"], input_df["sentence"]):
#         # get most updated scalar dist table:
#         patientNsentence_scalar_csvs_urls = db_url + '\\' + patientID[0:3] + '\\' + patientID + \
#                                             '\\results\\CordioPythonDistanceCalculation'
#         patientNsentence_scalar_csvs = glob.glob(patientNsentence_scalar_csvs_urls + "//**//*.csv", recursive=False)
#         if patientNsentence_scalar_csvs == []: continue
#         patientNsentence_db_csvs = [csv_url for csv_url in patientNsentence_scalar_csvs_urls if patientID in csv_url]
#         if patientNsentence_db_csvs == []: continue
#         patientNsentence_db_csvs = [csv_url for csv_url in patientNsentence_scalar_csvs if sentence in csv_url]
#         if patientNsentence_db_csvs == []: continue
#         patientNsentence_db_csvs = [csv_url for csv_url in patientNsentence_scalar_csvs if 'Scalar' in csv_url]
#         if patientNsentence_db_csvs == []: continue
#         path_patientNsentence_db_csvs = [Path(patientNsentence_db_csv) for patientNsentence_db_csv in
#                                          patientNsentence_db_csvs]
#         # TODO: use fileHandle here
#         patientNsentence_db_csvs_file_names = [os.path.basename(path_patientNsentence_db_csv) for
#                                                path_patientNsentence_db_csv in path_patientNsentence_db_csvs]
#         # ->filter by version:
#         patientNsentence_db_csvs_versions = [patientNsentence_db_csvs_file_name[25] for
#                                              patientNsentence_db_csvs_file_name in patientNsentence_db_csvs_file_names]
#         latest_ver_idx = [patientNsentence_db_csvs_version == max(patientNsentence_db_csvs_versions) for
#                           patientNsentence_db_csvs_version in patientNsentence_db_csvs_versions]
#         path_patientNsentence_db_csvs = list(compress(path_patientNsentence_db_csvs, latest_ver_idx))
#         # -> filter by date and time:
#         patientNsentence_db_csvs_dateNtimes = [patientNsentence_db_csvs_file_name[-17:-4] for
#                                                patientNsentence_db_csvs_file_name in
#                                                patientNsentence_db_csvs_file_names]
#         patientNsentence_db_csvs_dateNtimes = pd.to_datetime(patientNsentence_db_csvs_dateNtimes,
#                                                              format='%y%m%d_%H%M%S')
#         latest_ver_idx = [patientNsentence_db_csvs_dateNtime == max(patientNsentence_db_csvs_dateNtimes) for
#                           patientNsentence_db_csvs_dateNtime in patientNsentence_db_csvs_dateNtimes]
#         path_patientNsentence_db_csv = list(compress(path_patientNsentence_db_csvs, latest_ver_idx))
#         path_patientNsentence_scalar_csv = path_patientNsentence_db_csv[
#             0]  # path_patientNsentence_db_csv should be single
# else:
#     root = Tk()
#     root.withdraw()  # use to hide tkinter window
#     currdir = os.getcwd()
#
#     msg = "Please Enter Your Preference"
#     title = "setup run"
#     fieldNames = ['sentence ID list(S0007,S0010)', "number of patients(1,2,...):", "add datetime to output(y/n):",
#                   "overwrite existing tables(y/n):", "get data from csv url(leave empty for manual picking)"]
#     fieldValues = ['no_setup_defined', 'S0007', '1', 'y', 'n']  # we start with blanks for the values
#     fieldValues = easygui.multenterbox(msg, title, fieldNames)
#     fieldValues[0] = fieldValues[0].replace(" ", "")
#     sentence_list = fieldValues[0].split(',')
#     num_of_patients = int(fieldValues[1])
#     add_datetime = fieldValues[2]
#     overwrite_tables = fieldValues[3]
#     inputRunTableUrl = fieldValues[4]
#
#     patientDir_paths = [np.nan] * num_of_patients
#     all_scalarDistTable = [np.nan] * num_of_patients
#     for i in range(num_of_patients):
#         patientDir_paths[i] = filedialog.askdirectory(parent=root, initialdir=currdir,
#                                                       title='Please select the ' + str(
#                                                           i + 1) + ' patient dir directories')
#         curr_patient_ID = os.path.basename(os.path.normpath(patientDir_paths[i]))
#         all_scalarDistTable[i] = filedialog.askopenfiles(parent=root, initialdir=currdir,
#                                                          title='Please select a the scalar dist table for ' + curr_patient_ID)[
#             0]
#     # output_dir = filedialog.askdirectory(parent=root, initialdir=currdir, title='Please select a the output directory')
#
#     all_scalarDistTable = [x.name for x in all_scalarDistTable]
#
# # create tables:
# # ----------------
#
# # progress bar initialization:
# widgets = [FormatLabel('<<<all patient process>>>'), ' ', Percentage(), ' ', Bar('#'), ' ', RotatingMarker()]
# progressbar = ProgressBar(widgets=widgets, maxval=len(patientDir_paths))
# progressbar.start()
#
# # set output paths:
# save_url_paths = [patientDir_path + '\\results\\ESP' for patientDir_path in patientDir_paths]
#
# for i, patientDir_path, save_url_path in zip(range(len(patientDir_paths)), patientDir_paths, save_url_paths):
#     # progress bar update:
#     widgets[0] = FormatLabel('<filename-{0}>'.format(i))
#     progressbar.update(i)
#     # get all existing urls in output dir:
#     all_avalble_csvs = glob.glob(save_url_path + "//**//*.csv", recursive=True)
#     # table per model:
#     for model in ESP_TB.model_list:
#         emotions_list = ESP_TB.model_emotion_dict[type(model).__name__]
#         # set table path and name:
#         all_wavs = glob.glob(os.path.join(patientDir_path, '*.wav'))
#         p = Path(str(all_wavs[0]))
#         patient_ID = fileHandle.CordioExtractPatient(p)
#
#         for sentence in sentence_list:
#             # save_file_name = ESPResults_model_version_patientId_SentanceId_date_time
#             save_file_name = 'ESPResults_' + type(model).__name__ + '_V' + str(
#                 ESP_TB.version) + '_' + patient_ID + '_' + sentence
#             # check for existing table:
#             is_exist = [patient_prob_tables_urls for patient_prob_tables_urls in all_avalble_csvs if
#                         save_file_name in patient_prob_tables_urls]
#             if not ((is_exist != []) and (overwrite_tables == False)):
#                 # calculate emotion table:
#                 table = ESP_TB.create_ESP_patient_model_sentence_labeled_table(patientDir_path, model, sentence,
#                                                                                clinicalInformation, fileHandle)
#                 # save table:
#                 ESP_TB.SaveTable(table, save_url_path, save_file_name, add_datetime=False)
#                 # update all existing urls in output dir:
#                 all_avalble_csvs = glob.glob(save_url_path + "//**//*.csv", recursive=True)
#
# progressbar.finish()

from CordioESP import CordioESP_ToolBox, compress
import os, glob
from progressbar import *
import sys
import warnings
import easygui
import pandas as pd
from tqdm import tqdm
from itertools import compress
import ntpath

sys.path.insert(0, "D:\work\code\Python\CordioAlgorithmsPython\infrastructure")
from patientsInformation.CordioPatientClinicalInformation import CordioClinicalInformation
import CordioFile


warnings.filterwarnings('ignore')

# TODO: Add version control mechanism --> do not compute for same version

# getting tools ready:
# -------------------
# from CordioESP import CordioESP_ToolBox
ESP_TB = CordioESP_ToolBox()
clinicalInformation = CordioClinicalInformation
fileHandle = CordioFile

# get current run information from user:
# -------------------------------------

input_data_from_csv_url = easygui.fileopenbox('please select input csv file')
input_df = pd.read_csv(input_data_from_csv_url)
input_df = input_df[input_df["toRun"] == 1] # filter data
db_url = easygui.diropenbox('please select database folder')

# create tables:
# ----------------

progressbar = tqdm(total = len(input_df), position=0, leave=False)
for i, patientID, sentence in zip(range(len(input_df)), input_df["patientID"], input_df["sentence"]):
    # progressbar:
    progressbar.set_description('procesing...'.format(i))
    progressbar.update(1)
    # setup paths:
    if '-' not in patientID: patientID = patientID[0:3]+'-'+patientID[3:]
    patientDir_path = db_url + '\\' + patientID[0:3] + '\\' + patientID
    save_url_path = patientDir_path + '\\results\\CordioPythonDistanceCalculation'
    emo_labeled_data_tables_url = patientDir_path + '\\results\\CordioESP'
    # get patient ESP CSVs:
    all_available_ESPScalar_csvs = glob.glob(save_url_path + "//**//*.csv", recursive=True)
    all_available_emo_labeled_csvs = glob.glob(emo_labeled_data_tables_url + "//**//*.csv", recursive=True)
    for model in ESP_TB.model_list:
        # save_file_name = 'ESPResults_' + type(model).__name__ + '_V' + str(ESP_TB.version) + '_' + patientID + \
        #                  '_' + sentence

        # get emotion labeled data table:
        # ------------------------------
        # check for existing table with the same version:
        all_available_model_emo_labeled_csvs = [model_emo_labeled_csv for model_emo_labeled_csv in all_available_model_emo_labeled_csvs if sentence in model_emo_labeled_csv]
        all_available_model_emo_labeled_csvs = [ESP_model_csv for ESP_model_csv in all_available_emo_labeled_csvs if type(model).__name__ in ESP_model_csv]
        all_available_model_model_emo_version = [float(model_model_emo_version[-36:-33]) for model_model_emo_version in all_available_model_emo_labeled_csvs]
        all_available_model_model_emo_lataset_ver_idx = [model_model_emo_version == ESP_TB.version for model_model_emo_version in all_available_model_model_emo_version]
        if not any(all_available_model_model_emo_lataset_ver_idx): continue
        latest_ver_table_urls = list(compress(all_available_model_emo_labeled_csvs, all_available_model_model_emo_lataset_ver_idx))
        emo_table_url = latest_ver_table_urls[-1]# same version -> time doesnt matter
        # get scalar dist labeled data table:
        # ------------------------------
        all_available_ESPScalar_csv_urls = [ESPScalar_csv_url for ESPScalar_csv_url in all_available_ESPScalar_csvs if 'ScalarAsrDtwDistSummary' in ESPScalar_csv_url]
        if all_available_ESPScalar_csv_urls == []: continue
        all_available_ESPScalar_csv_urls = [ESPScalar_csv_url for ESPScalar_csv_url in all_available_ESPScalar_csv_urls if sentence in ESPScalar_csv_url]
        if all_available_ESPScalar_csv_urls == []: continue
        # get version
        all_available_ESPScalar_csv_versions = [float(ntpath.basename(ESPScalar_csv_url)[25:28]) for ESPScalar_csv_url in all_available_ESPScalar_csv_urls]
        all_available_ESPScalar_csv_late_ver_idx = [ESPScalar_csv_version == max(all_available_ESPScalar_csv_versions) for ESPScalar_csv_version in all_available_ESPScalar_csv_versions ]
        all_available_ESPScalar_csv_urls = list(compress(all_available_ESPScalar_csv_urls, all_available_ESPScalar_csv_late_ver_idx))
        latest_ver_ESPScalar_url = all_available_ESPScalar_csv_urls[-1] # lataest version - date is not important
        # TODO:

        all_available_model_model_emo_version = [float(model_model_emo_version[-36:-33]) for model_model_emo_version in all_available_model_emo_labeled_csvs]
            # updating existing emotion tableL
            latest_ver_table_urls = list(compress(all_available_model_emo_labeled_csvs, all_available_model_model_emo_lataset_ver_idx))
            table = pd.read_csv(latest_ver_table_urls[-1])
            new_table = ESP_TB.append_ESP_patient_model_sentence_labeled_table(patientDir_path, table, model, sentence, clinicalInformation, fileHandle)
        else:
            # calculate new emotion table:
            # get all wavs:
            all_wavs = glob.glob(os.path.join(patientDir_path, '*.wav'))
            table = ESP_TB.create_ESP_patient_model_sentence_labeled_table(all_wavs, model, sentence,
                                                                           clinicalInformation, fileHandle)

        if ("new_table" in locals()):
            if(len(new_table)>len(table)):
                # if table existed and updated:
                os.remove(latest_ver_table_urls[-1]) # remove previes table
                new_table['FileIdx'] = []
                new_table = table.reset_index()
                new_table.index.name = 'FileIdx'
                new_table.index = new_table.index + 1
                ESP_TB.SaveTable(table, save_url_path, save_file_name, add_datetime=True)
            # if len(new_table)<=len(table) then keep the original table
        else:
            table.index.name = 'FileIdx'
            table.index = table.index + 1
            # save table:
            ESP_TB.SaveTable(table, save_url_path, save_file_name, add_datetime=True)

        if ("new_table" in locals()): del new_table

progressbar.close()