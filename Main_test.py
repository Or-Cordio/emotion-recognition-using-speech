from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from pathlib import Path
import os, glob
from progressbar import *

import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# sys.path.append("D:\work\code\Python\CordioAlgorithmsPython\infrastructure")
sys.path.insert(0, "D:\work\code\Python\CordioAlgorithmsPython\infrastructure")

from patientsInformation.CordioPatientClinicalInformation import CordioClinicalInformation
import CordioFile

warnings.filterwarnings('ignore')

# hard coded data for current run:
# -------------------------------

# model_list = [SVC(probability=True), AdaBoostClassifier(), RandomForestClassifier(), GradientBoostingClassifier(),
#               DecisionTreeClassifier(), KNeighborsClassifier(), MLPClassifier()]
model_list = [SVC(probability=True), AdaBoostClassifier(), RandomForestClassifier(), KNeighborsClassifier()]

# emotions_list = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'ps', 'boredom']
emotions_lists = [['happy', 'sad', 'disgust', 'ps', 'boredom'],
                  ['calm', 'happy', 'disgust', 'ps', 'boredom'],
                  ['calm', 'happy', 'fear', 'disgust', 'boredom'],
                  ['calm', 'happy', 'sad', 'disgust', 'ps']]
model_emotion_dict = {'SVC': ['happy', 'sad', 'disgust', 'ps', 'boredom'],
                      'AdaBoostClassifier': ['calm', 'happy', 'disgust', 'ps', 'boredom'],
                      'RandomForestClassifier': ['calm', 'happy', 'fear', 'disgust', 'boredom'],
                      'KNeighborsClassifier': ['calm', 'happy', 'sad', 'disgust', 'ps']}
patientDir_paths = ["D:\work\db\RAM\RAM-0071", "D:\work\db\BSV\BSV-0006", "D:\work\db\HYA\HYA-0055"]
same_class_per_all_models = False
# setup_name = "9emotions_7models_first_setup"
session_hour_range = 1
setup_name = "5emotions_4models_second_setup"
add_datetime = True

# getting tools ready:
from CordioESP import CordioESP_ToolBox
ESP_TB = CordioESP_ToolBox()
clinicalInformation = CordioClinicalInformation
fileHandle = CordioFile

# get probability tables:
# ----------------------
# 2 configurations defighned by same_class_fpr_all_models:
## one prob table per patient (for all models - same classes for all models)
## prob table for a model(for all models - different classes for each model)

# progress bar initialization:
widgets = [FormatLabel('<<<all patient process>>>'), ' ', Percentage(), ' ', Bar('#'), ' ', RotatingMarker()]
progressbar = ProgressBar(widgets=widgets, maxval=len(patientDir_paths))
progressbar.start()

# create patient_prob_tables_urls:
all_csv_root_path = "D:\\work\\code\\Python\\ESP\\noPaperProject\\results_tablesOfProb\\" + setup_name
patient_prob_tables_urls = glob.glob(all_csv_root_path+"//**//*.csv", recursive=True)

for i, patientDir_path, emotions_list in zip(range(len(patientDir_paths)), patientDir_paths, emotions_lists):
    # progress bar update:
    widgets[0] = FormatLabel('<filename-{0}>'.format(i))
    progressbar.update(i)
    if same_class_per_all_models:
        # table per patient:
        # set table path and name:
        all_wavs = glob.glob(os.path.join(patientDir_path, '*.wav'))
        p = Path(str(all_wavs[0]))
        patient_ID = fileHandle.CordioExtractPatient(p)
        save_url_path = "results_tablesOfProb\\" + setup_name + "\\" + patient_ID
        save_file_name = 'tablesOfProb_' + patient_ID + '_' + str(
            len(emotions_list) + 'classes_' + str(len(model_list)) + 'models')
        # check for existing table:
        is_exist = [patient_prob_tables_urls for patient_prob_tables_urls in patient_prob_tables_urls if
                    save_file_name in patient_prob_tables_urls]
        if is_exist != []:
            continue
        # calculate table:
        table = ESP_TB.predict_all_proba_for_patient(patientDir_path, clinicalInformation, fileHandle, model_list,
                                                     emotions_list)
        # save patient table:
        ESP_TB.SaveTable(table, save_url_path, save_file_name, add_datetime=True)

    else:
        # table per model:
        for model, emotions_list in zip(model_list, emotions_lists):
            # set table path and name:
            all_wavs = glob.glob(os.path.join(patientDir_path, '*.wav'))
            p = Path(str(all_wavs[0]))
            patient_ID = fileHandle.CordioExtractPatient(p)
            save_url_path = "results_tablesOfProb\\" + setup_name + "\\" + patient_ID
            save_file_name = 'tablesOfProb_' + patient_ID + '_' + str(len(emotions_list)) + 'classes_' + type(
                model).__name__
            # check for existing table:
            is_exist = [patient_prob_tables_urls for patient_prob_tables_urls in patient_prob_tables_urls if save_file_name in patient_prob_tables_urls]
            if is_exist != []:
                continue
            # calculate table:
            table = ESP_TB.predict_all_proba_for_patient(patientDir_path, clinicalInformation, fileHandle, [model],
                                                         emotions_list)
            # save table:
            ESP_TB.SaveTable(table, save_url_path, save_file_name, add_datetime=True)

progressbar.finish()

# patient_prob_tables_urls = [
#     "D:\\work\\code\\Python\\ESP\\noPaperProject\\results_tablesOfProb\\tablesOfProb_RAM-0071_9classes_7models.csv",
#     "D:\\work\\code\\Python\\ESP\\noPaperProject\\results_tablesOfProb\\tablesOfProb_HYA-0055_9classes_7models.csv",
#     "D:\\work\\code\\Python\\ESP\\noPaperProject\\results_tablesOfProb\\tablesOfProb_BSV-0006_9classes_7models.csv"]
# patient_prob_tables_urls = ["D:\\work\\code\\Python\\ESP\\noPaperProject\\results_tablesOfProb\\tablesOfProb_RAM-0071_9classes_7models.csv"]

# plots:
# -----
# create patient_prob_tables_urls:
all_csv_root_path = "D:\\work\\code\\Python\\ESP\\noPaperProject\\results_tablesOfProb\\" + setup_name
patient_prob_tables_urls = glob.glob(all_csv_root_path+"//**//*.csv", recursive=True)
# plotting:
if not same_class_per_all_models:
    # progress bar initialization:
    widgets = [FormatLabel('<<<all prob tables process>>>'), ' ', Percentage(), ' ', Bar('#'), ' ', RotatingMarker()]
    progressbar = ProgressBar(widgets=widgets, maxval=len(patient_prob_tables_urls))
    progressbar.start()
    for i, patient_prob_tables_url in zip(range(len(patient_prob_tables_urls)), patient_prob_tables_urls):
        # progress bar update:
        widgets[0] = FormatLabel('<filename-{0}>'.format(i))
        progressbar.update(i)
        # get current data:
        model = patient_prob_tables_url.split('\\')[-1].split('_')[-3]
        emotion_list = model_emotion_dict[model]
        # call plot functions:
        ESP_TB.patient_plotNsave_mean_prob_session_emotion_3d([patient_prob_tables_url], model, emotion_list,
                                                              session_hour_range, setup_name)

        ESP_TB.patient_plotNsave_count_hard_decision_histogram([patient_prob_tables_url], model, emotion_list,
                                                               session_hour_range, setup_name)

        ESP_TB.patient_plotNsave_sum_histogram([patient_prob_tables_url], model, emotion_list, session_hour_range,
                                               setup_name)
    progressbar.finish()


ESP_TB.get_model_variance_per_patientNmodel_heatmap(patient_prob_tables_urls, model_list, emotions_list,
                                                    session_hour_range, setup_name)

ESP_TB.get_model_variance_per_patient_all_models_multiple_graph(patient_prob_tables_urls, model_list, emotions_list,
                                                                session_hour_range, setup_name)

ESP_TB.get_model_variance_per_patient_all_models(patient_prob_tables_urls, model_list, emotions_list,
                                                 session_hour_range, setup_name)

ESP_TB.patient_plotNsave_emotion_over_time_summerize_for_model_one_plot(patient_prob_tables_urls, model_list,
                                                                        emotions_list, session_hour_range, setup_name)

ESP_TB.patient_plotNsave_mean_prob_session_emotion_3d(patient_prob_tables_urls, model_list, emotions_list,
                                                      session_hour_range, setup_name)

ESP_TB.patient_plotNsave_emotion_over_time_summerize_for_model_subplots(patient_prob_tables_urls, model_list,
                                                                        emotions_list, session_hour_range, setup_name)

ESP_TB.patient_plotNsave_count_hard_decision_histogram(patient_prob_tables_urls, model_list, emotions_list,
                                                       session_hour_range, setup_name)

ESP_TB.patient_plotNsave_sum_histogram(patient_prob_tables_urls, model_list, emotions_list, session_hour_range,
                                       setup_name)

ESP_TB.patient_plotNsave_emotion_over_time(patient_prob_tables_urls, [], emotions_list, session_hour_range, setup_name)

for patient_prob_table in patient_prob_tables:
    prob_df = pd.read_csv(patient_prob_table)
    graphs_df = ESP_TB.get_table_by_session(patient_prob_table, 1)
    ESP_TB.patient_plotNsave_emotion_over_time(patient_prob_tables_urls, model_list, emotions_list)
