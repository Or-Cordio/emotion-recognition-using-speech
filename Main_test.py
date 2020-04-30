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

# sys.path.append("D:\work\code\Python\CordioAlgorithmsPython\infrastructure")
sys.path.insert(0, "D:\work\code\Python\CordioAlgorithmsPython\infrastructure")

from patientsInformation.CordioPatientClinicalInformation import CordioClinicalInformation
import CordioFile

model_list = [SVC(probability=True), AdaBoostClassifier(), RandomForestClassifier(), GradientBoostingClassifier(),
              DecisionTreeClassifier(), KNeighborsClassifier(), MLPClassifier()]

warnings.filterwarnings('ignore')

# model_list = [SVC(probability=True), GradientBoostingClassifier()]

emotions_list = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'ps', 'boredom']

# patientDir_path = "D:\work\db\RAM\RAM-0071"
patientDir_paths = ["D:\work\db\BSV\BSV-0006", "D:\work\db\HYA\HYA-0055"]
from CordioESP import CordioESP_ToolBox

# getting tools ready:
ESP_TB = CordioESP_ToolBox()
clinicalInformation = CordioClinicalInformation
fileHandle = CordioFile

import pandas as pd
patient_prob_tables_urls = ["D:\\work\\code\\Python\\ESP\\noPaperProject\\results_tablesOfProb\\tablesOfProb_RAM-0071_9classes_7models.csv",
                       "D:\\work\\code\\Python\\ESP\\noPaperProject\\results_tablesOfProb\\tablesOfProb_HYA-0055_9classes_7models.csv",
                       "D:\\work\\code\\Python\\ESP\\noPaperProject\\results_tablesOfProb\\tablesOfProb_BSV-0006_9classes_7models.csv"]
# patient_prob_tables_urls = ["D:\\work\\code\\Python\\ESP\\noPaperProject\\results_tablesOfProb\\tablesOfProb_RAM-0071_9classes_7models.csv"]
session_hour_range = 1
setup_name = "9emotions_7models_first_setup"

ESP_TB.get_model_variance_per_patient_all_models_multiple_graph(patient_prob_tables_urls, model_list, emotions_list, session_hour_range, setup_name)

ESP_TB.get_model_variance_per_patient_all_models_multiple_graph(patient_prob_tables_urls, model_list, emotions_list, session_hour_range, setup_name)

ESP_TB.get_model_variance_per_patient_all_models(patient_prob_tables_urls, model_list, emotions_list, session_hour_range, setup_name)

ESP_TB.get_model_variance_per_patientNmodel_heatmap(patient_prob_tables_urls, model_list, emotions_list, session_hour_range, setup_name)

ESP_TB.patient_plotNsave_emotion_over_time_summerize_for_model_one_plot(patient_prob_tables_urls, model_list, emotions_list, session_hour_range, setup_name)

ESP_TB.patient_plotNsave_mean_prob_session_emotion_3d(patient_prob_tables_urls, model_list, emotions_list, session_hour_range, setup_name)

ESP_TB.patient_plotNsave_emotion_over_time_summerize_for_model_subplots(patient_prob_tables_urls, model_list, emotions_list, session_hour_range, setup_name)

ESP_TB.patient_plotNsave_count_hard_decision_histogram(patient_prob_tables_urls, model_list, emotions_list, session_hour_range,setup_name)

ESP_TB.patient_plotNsave_sum_histogram(patient_prob_tables_urls, model_list, emotions_list, session_hour_range, setup_name)

ESP_TB.patient_plotNsave_emotion_over_time(patient_prob_tables_urls, [], emotions_list, session_hour_range, setup_name)

for patient_prob_table in patient_prob_tables:
    prob_df = pd.read_csv(patient_prob_table)
    graphs_df = ESP_TB.get_table_by_session(patient_prob_table, 1)
    ESP_TB.patient_plotNsave_emotion_over_time(patient_prob_tables_urls, model_list, emotions_list)

# progress bar initialization:
widgets = [FormatLabel('<<<all patient process>>>'), ' ', Percentage(), ' ', Bar('#'), ' ', RotatingMarker()]
progressbar = ProgressBar(widgets=widgets, maxval=len(patientDir_paths))
progressbar.start()
for i, patientDir_path in zip(range(len(patientDir_paths)), patientDir_paths):
    # progress bar update:
    widgets[0] = FormatLabel('<filename-{0}>'.format(i))
    progressbar.update(i)

    table = ESP_TB.predict_all_proba_for_patient(patientDir_path, clinicalInformation, fileHandle, model_list, emotions_list)
    # save patient table:
    emo_naum = len(emotions_list)
    model_num = len(model_list)
    all_wavs = glob.glob(os.path.join(patientDir_path, '*.wav'))
    p = Path(str(all_wavs[0]))
    patient_ID = fileHandle.CordioExtractPatient(p)
    Path("results_tablesOfProb").mkdir(parents=True, exist_ok=True)
    table.to_csv('results_tablesOfProb\\tablesOfProb_' + patient_ID + '_' + str(emo_naum) + 'classes_' + str(
        model_num) + 'models.csv')
progressbar.finish()

