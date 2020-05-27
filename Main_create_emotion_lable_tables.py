from CordioESP import CordioESP_ToolBox, compress
import os, glob
from progressbar import *
import sys
import warnings
import easygui
import pandas as pd
from tqdm import tqdm
from itertools import compress

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
    save_url_path = patientDir_path + '\\results\\CordioESP'
    # get patient ESP CSVs:
    all_available_ESP_csvs = glob.glob(save_url_path + "//**//*.csv", recursive=True)
    for model in ESP_TB.model_list:
        save_file_name = 'ESPResults_' + type(model).__name__ + '_V' + str(ESP_TB.version) + '_' + patientID + \
                         '_' + sentence

        # check for existing table with the same version:
        all_available_model_ESP_csvs = [ESP_model_csv for ESP_model_csv in all_available_ESP_csvs if type(model).__name__ in ESP_model_csv]
        all_available_model_ESP_version = [float(model_ESP_version[-36:-33]) for model_ESP_version in all_available_model_ESP_csvs]
        all_available_model_ESP_lataset_ver_idx = [model_ESP_version == ESP_TB.version for model_ESP_version in all_available_model_ESP_version]
        if any(all_available_model_ESP_lataset_ver_idx):
            # updating existing emotion tableL
            latest_ver_table_urls = list(compress(all_available_model_ESP_csvs, all_available_model_ESP_lataset_ver_idx))
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
