import os, glob
import pandas as pd
import numpy as np
import datetime
from datetime import datetime as dt
# import progressbar
from matplotlib.pyplot import subplots_adjust
from progressbar import *
from pathlib import Path
from emotion_recognition import EmotionRecognizer
from progressbar import *
import matplotlib.pyplot as plt
from pylab import *
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


# import sys
# # sys.path.append("D:\work\code\Python\CordioAlgorithmsPython\infrastructure")
# sys.path.insert(0, "D:\work\code\Python\CordioAlgorithmsPython\infrastructure")

# from patientsInformation.CordioPatientClinicalInformation import CordioClinicalInformation
# import CordioFile

# model_list = [SVC(probability=True), AdaBoostClassifier(), RandomForestClassifier(), GradientBoostingClassifier(),
#               DecisionTreeClassifier(), KNeighborsClassifier(), MLPClassifier()]
# emotions_list = ['sad', 'neutral', 'happy']
#
# patientDir_path = "D:\work\db\RAM\RAM-0071"
# p = "D:\work\db\BSV\BSV-0006\BSV-0006_190520_094113_S0010_he_1.25_SM-J400F_Android26.wav"

class CordioESP_ToolBox:
    ''' Expressive Speech Processing '''

    def __init__(self):
        self.suported_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'ps', 'boredom']
        self.supported_models = ['SVC', 'AdaBoostClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier',
                                 'DecisionTreeClassifier', 'KNeighborsClassifier', 'MLPClassifier']

    def modelTrain(self, model, emotions_list):
        # my_model probability attribute needs to be Truth!

        # pass my model to EmotionRecognizer instance
        # and balance the dataset
        rec = EmotionRecognizer(model=model, emotions=emotions_list, balance=True, verbose=0, probability=True)
        # train the model
        rec.train()

        return rec
        # check the test accuracy for that model
        # print(type(my_model).__name__+" Test score:", rec.test_score())
        # # check the train accuracy for that model
        # print("Train score:", rec.train_score())

    def modelPredict(self, rec, wav_url):
        try:
            out = rec.predict_proba(wav_url)
        except ValueError:
            out = type(rec.model).__name__ + " doesnt support in predict_proba"
        return out

    def predict_all_proba_for_patientNmodel(self, model, fileHandle, clinicalInformation, patient_info_column_names,
                                            emotions_list, all_wavs):

        df_len = len(all_wavs)
        patientNmodel_df = pd.DataFrame(index=np.arange(df_len),
                                        columns=patient_info_column_names + ['Model'] + emotions_list)

        rec = self.modelTrain(self, model, emotions_list)

        # progress bar initialization:
        p = Path(str(all_wavs[0]))
        # fileHandle = CordioFile
        patient_ID = fileHandle.CordioExtractPatient(p)
        patient_model = type(model).__name__
        widgets = [FormatLabel('<patient: ' + patient_ID + '; model: ' + patient_model + '>'), ' ', Percentage(), ' ',
                   Bar('#'), ' ', RotatingMarker()]
        progressbar = ProgressBar(widgets=widgets, maxval=df_len)
        progressbar.start()

        # fill df:
        for (i, wav) in zip(range(df_len), all_wavs):
            # progress bar update:
            widgets[0] = FormatLabel('<filename-{0}>'.format(i))
            progressbar.update(i)

            # add  soft decision score for each emotion
            patientNmodel_df.loc[i] = self.modelPredict(self, rec, wav)

            # insert basic information:
            p = Path(str(wav))
            # fileHandle = CordioFile
            patientNmodel_df.at[i, "PatientName"] = fileHandle.CordioExtractPatient(p)
            patientNmodel_df.at[i, "Date"] = fileHandle.CordioExtractRecordingDateTime(p).strftime("%d/%m/%Y")
            patientNmodel_df.at[i, "Time"] = fileHandle.CordioExtractRecordingDateTime(p).strftime("%H:%M:%S")
            patientNmodel_df.at[i, "sentence"] = fileHandle.CordioExtractSentence(p)
            patientNmodel_df.at[i, "Language"] = fileHandle.CordioExtractLanguage(p)
            # TODO: add App version, Device identifier and OS version columns

            # setting clinical status:
            clinicalStatus = self.get_clinical_info(self, clinicalInformation, fileHandle.CordioExtractRecordingDateTime(p),
                                               patientNmodel_df.at[i, "PatientName"])
            # clinicalInfo = clinicalInformation(patientNmodel_df.at[i, "PatientName"], '')
            # clinicalStatusCode = clinicalInfo(fileHandle.CordioExtractRecordingDateTime(p))
            # clinicalStatus = "dry"
            # if clinicalStatusCode == -1:
            #     # recording is not valid (before patient registration)
            #     clinicalStatus = 'recording is not valid (before patient registration)'
            # elif clinicalStatusCode == clinicalInfo.CLINICAL_STATUS_UNKNOWN:
            #     clinicalStatus = "unknown"
            # elif clinicalStatusCode == clinicalInfo.CLINICAL_STATUS_WET:
            #     clinicalStatus = "wet"
            # patientNmodel_df.at[i, "ClinicalStatus"] = clinicalStatus

            # setting model:
            patientNmodel_df.at[i, "Model"] = type(model).__name__
        progressbar.finish()

        return patientNmodel_df

    def predict_all_proba_for_patient(self, patientDir_path, clinicalInformation, fileHandle, model_list,
                                      emotions_list):
        # get all wavs:
        all_wavs = glob.glob(os.path.join(patientDir_path, '*.wav'))
        num_of_wav = len(all_wavs) * len(model_list)

        # create basic information table for patient:
        patient_info_column_names = ["PatientName", "Date", "Time", "sentence", "Language", "ClinicalStatus"]
        patient_df = pd.DataFrame(columns=patient_info_column_names + ['Model'] + emotions_list)

        # progress bar initialization:
        p = Path(str(all_wavs[0]))
        # fileHandle = CordioFile
        patient_ID = fileHandle.CordioExtractPatient(p)
        widgets = [FormatLabel('<<patient: ' + patient_ID + '; all models process>>'), ' ', Percentage(), ' ',
                   Bar('#'), ' ', RotatingMarker()]
        progressbar = ProgressBar(widgets=widgets, maxval=len(model_list))
        progressbar.start()

        # calculating for all models:
        for i, model in zip(range(len(model_list)), model_list):
            # progress bar update:
            widgets[0] = FormatLabel('<filename-{0}>'.format(i))
            progressbar.update(i)

            tmp = self.predict_all_proba_for_patientNmodel(self, model, fileHandle, clinicalInformation,
                                                           patient_info_column_names, emotions_list, all_wavs)
            patient_df = patient_df.append(tmp)
        progressbar.finish()

        return patient_df

    def get_clinical_info(self, clinicalInformation, recording_datetime, patient_id):
        clinicalInfo = clinicalInformation(patient_id, '')
        clinicalStatusCode = clinicalInfo(recording_datetime)
        clinicalStatus = "dry"
        if clinicalStatusCode == -1:
            # recording is not valid (before patient registration)
            clinicalStatus = 'recording is not valid (before patient registration)'
        elif clinicalStatusCode == clinicalInfo.CLINICAL_STATUS_UNKNOWN:
            clinicalStatus = "unknown"
        elif clinicalStatusCode == clinicalInfo.CLINICAL_STATUS_WET:
            clinicalStatus = "wet"

        return clinicalStatus

    def plot_emotion_over_time(self, prob_table, session_hour_range):

        # TODO: add description

        # prob_table check: check necessary columns existence
        prob_table_col_names = list(prob_table.columns)
        if 'Unnamed: 0' in prob_table_col_names:
            prob_table.drop('Unnamed: 0', axis=1)
        prob_table['Date'] = pd.to_datetime(prob_table['Date'], format="%d/%m/%Y")
        prob_table['Time'] = pd.to_datetime(prob_table['Time'], format="%H:%M:%S")

        # initial graphs df:
        emotions_in_prob_table_idx = [idx for idx, val in enumerate(self.suported_emotions) if val in prob_table_col_names]
        emotions_in_prob_table = [self.suported_emotions[i] for i in emotions_in_prob_table_idx]
        graphs_df_col_names = ['Patient_id', 'SessionIdx', 'Date', 'FirstSessionRecTime',
                               'LastSessionRecTime', 'Model', 'IsWet'] + emotions_in_prob_table
        graphs_df = pd.DataFrame(columns=graphs_df_col_names)

        # fill graphs_df:
        unique_dates = prob_table.Date.dt.strftime("%d/%m/%Y").unique()
        unique_dates = [x for x in unique_dates if str(x) != 'nan']     # remove nans
        prob_table = prob_table.sort_values(['Date', 'Time'], ascending=[True, True])
        session_idx = 0
        for date in unique_dates:
            print(date)
            # get current date sub-df
            dt_date = dt.strptime(date, "%d/%m/%Y")

            # mean probabilities for each model type:
            unique_model_types = prob_table.Model.unique()
            # remove unsapported models:
            unique_model_types = [val for idx, val in enumerate(self.supported_models) if val in unique_model_types]
            for model in unique_model_types:
                prob_table_dateNmodel_sub_df = prob_table[(prob_table['Model'] == model) & (prob_table['Date'] == dt_date)]

                curr_time_idx = prob_table_dateNmodel_sub_df.index.values[0] # first index of prob_table_dateNmodel_sub_df
                curr_time = pd.to_datetime(prob_table_dateNmodel_sub_df['Time'].loc[curr_time_idx], format="%H:%M:%S")
                last_dateNmodel_idx = prob_table_dateNmodel_sub_df.index[-1]
                while curr_time_idx <= last_dateNmodel_idx:
                    session_mask = (prob_table_dateNmodel_sub_df['Time'] >= curr_time) & (prob_table_dateNmodel_sub_df['Time'] < curr_time + datetime.timedelta(hours=session_hour_range))
                    prob_table_dateNmodel_seassion_sub_df = prob_table_dateNmodel_sub_df[session_mask]
                    mean_prob_row = prob_table_dateNmodel_seassion_sub_df.mean(axis=0, numeric_only=True)

                    basic_info_dict = {'Patient_id': [prob_table_dateNmodel_seassion_sub_df.iloc[0]['PatientName']],
                                       'SessionIdx': [session_idx],
                                       'Date': [prob_table_dateNmodel_seassion_sub_df.iloc[0]['Date']],
                                       'FirstSessionRecTime': [prob_table_dateNmodel_seassion_sub_df.iloc[0]['Time'].strftime("%H:%M:%S")],
                                       'LastSessionRecTime': [prob_table_dateNmodel_seassion_sub_df.iloc[-1]['Time'].strftime("%H:%M:%S")],
                                       'Model': model,
                                       'IsWet': (prob_table_dateNmodel_seassion_sub_df['ClinicalStatus'] == 'wet').any()}
                    basic_info_dict.update(mean_prob_row.to_dict())
                    full_info_dict = basic_info_dict
                    # remove bad entries:
                    full_info_dict = {k: full_info_dict[k] for k in graphs_df_col_names}
                    # insert new row
                    graphs_df = graphs_df.append(pd.DataFrame(full_info_dict))
                    session_idx = session_idx + 1

                    # iterate to next time value:
                    last_true_value_idx = session_mask[::-1].idxmax()
                    curr_time_idx = last_true_value_idx + 1
                    if curr_time_idx <= last_dateNmodel_idx:
                        curr_time = pd.to_datetime(prob_table_dateNmodel_sub_df['Time'].loc[curr_time_idx], format="%H:%M:%S")

        return graphs_df

    def patient_plotNsave_emotion_over_time(self, patient_prob_tables_urls, model_list, emotion_list):

        for patient_prob_table_url in patient_prob_tables_urls:
            # loading data if available:
            try:
                graph_df = pd.read_csv(patient_prob_table_url)
            except:
                print("File not avalble in: "+patient_prob_table_url)
            patient_id = graph_df['Patient_id'][0]
            # remove unsapported models:
            model_list = [val for idx, val in enumerate(self.supported_models) if val in model_list]
            emotion_list = [val for idx, val in enumerate(self.suported_emotions) if val in emotion_list]

            for model in model_list:
                model_graphs_df = graph_df[graph_df['Model'] == model]
                for emotion in emotion_list:
                    # plot:
                    fig, ax = plt.subplots()
                    x = model_graphs_df['Date']
                    y = model_graphs_df[emotion]
                    ax.plot(x, y, linestyle='--', marker='o', color='black')
                    ax.fill_between(x, 0, 1, where=model_graphs_df['IsWet'],
                                    color='aqua', alpha=0.4, transform=ax.get_xaxis_transform())
                    ax.legend(["mean sessions probabilty for emotion", "wet sessions"])
                    fig.suptitle('Mean Sessions Probability\n'+
                                 'trained with '+str(len(emotion_list))+' classes\n'+
                                 'Patient: '+patient_id+
                                 ', Model: '+model+
                                 ', Emotion: '+emotion, fontsize=16)
                    plt.xlabel('Date\n(may be multiple sessions in one dates - different hours)')
                    ax.xaxis.set_major_locator(plt.MaxNLocator(30))     # reducing number of plot ticks
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)   # rotate plot tics
                    plt.grid()
                    plt.ylabel(emotion+' mean session probability')

                    # save fig:
                    save_url_path = "results_tablesOfProb\\"+str(len(emotion_list))+"emotionsTraining\\"+patient_id+"\\"+model
                    Path(save_url_path).mkdir(parents=True, exist_ok=True)
                    save_file_name = 'MeanSessionsProbability'+'_trainedWith'+str(len(emotion_list))+'Classes'+'_'+patient_id+'_'+model+'_'+emotion
                    manager = plt.get_current_fig_manager()
                    manager.window.showMaximized()
                    fig.savefig(save_url_path+"\\"+save_file_name+".png")