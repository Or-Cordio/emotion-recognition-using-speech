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
from collections import Counter
import seaborn as sn
from progressbar import *
import matplotlib.dates as mdates

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
    """
    Expressive Speech Processing

    # TODO: add class description

       Cordio Medical - Confidential
         Version: 0.1    2020-04-27

         Revision History:
         |   Ver    | Author    | Date           | Change Description
         |----------|-----------|----------------|--------------------
         |   0.1    | Or        | 2020-04-27     | Initial
         |   x.x    | xxxx      | xxxx-xx-xx     | x
     """

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

        rec = self.modelTrain(model, emotions_list)

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
            patientNmodel_df.loc[i] = self.modelPredict(rec, wav)

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
            clinicalStatus = self.get_clinical_info(clinicalInformation, fileHandle.CordioExtractRecordingDateTime(p),
                                               patientNmodel_df.at[i, "PatientName"])

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

            tmp = self.predict_all_proba_for_patientNmodel(model, fileHandle, clinicalInformation,
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

    def SaveFig(self, fig, save_url_path, save_file_name, add_datetime, close_fig):
        # from pathlib import Path
        # create folders in path if not exist:
        Path(save_url_path).mkdir(parents=True, exist_ok=True)
        # remove old file with the same name if exist:
        if os.path.isfile(save_url_path + "\\" + save_file_name + ".png"):
            os.remove(save_url_path + "\\" + save_file_name + ".png")
        plt.ioff()
        # save file:
        # datetime object containing current date and time
        now = dt.now()
        if (add_datetime == []) or (add_datetime == True):
            dt_string = now.strftime("%d%m%y_%H%M%S")
            fig.savefig(save_url_path + "\\" + save_file_name + dt_string + ".png", bbox_inches='tight')
        else:
            fig.savefig(save_url_path + "\\" + save_file_name + ".png", bbox_inches='tight')
        if close_fig:
            plt.close(fig)

    def SaveTable(self, table, save_url_path, save_file_name, add_datetime):
        # from pathlib import Path
        # create folders in path if not exist:
        Path(save_url_path).mkdir(parents=True, exist_ok=True)
        # remove old file with the same name if exist:
        if os.path.isfile(save_url_path + "\\" + save_file_name + ".png"):
            os.remove(save_url_path + "\\" + save_file_name + ".png")
        plt.ioff()
        # save file:
        # datetime object containing current date and time
        now = dt.now()
        if (add_datetime == []) or (add_datetime == True):
            dt_string = now.strftime("%d%m%y_%H%M%S")
            table.to_csv(save_url_path + "\\" + save_file_name + "_" + dt_string + '.csv')
        else:
            table.to_csv(save_url_path + "\\" + save_file_name + '.csv')

    def get_table_by_session(self, prob_table, session_hour_range, session_action, emotions_list):

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
                    # mean_prob_row = prob_table_dateNmodel_seassion_sub_df.mean(axis=0, numeric_only=True, skipna=True)
                    # mean_prob_row = getattr(prob_table_dateNmodel_seassion_sub_df.mean(axis=0, numeric_only=True, skipna=True).modules[__name__], str)

                    mean_prob_row = getattr(prob_table_dateNmodel_seassion_sub_df[emotions_list], session_action)(axis=0, numeric_only=True, skipna=True)
                    # if session_action = 'std' and prob_table_dateNmodel_seassion_sub_df is one line mean_prob_row will be nan

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
                    # for debug:
                    if pd.DataFrame(full_info_dict).isnull().values.any():
                        print(str(date)+' '+model+' '+str(curr_time_idx))
                    # insert new row
                    graphs_df = graphs_df.append(pd.DataFrame(full_info_dict))
                    session_idx = session_idx + 1

                    # iterate to next time value:
                    last_true_value_idx = session_mask[::-1].idxmax()
                    curr_time_idx = last_true_value_idx + 1
                    if curr_time_idx <= last_dateNmodel_idx:
                        try:
                            curr_time = pd.to_datetime(prob_table_dateNmodel_sub_df['Time'].loc[curr_time_idx], format="%H:%M:%S")
                        except:
                            print(str(date)+' '+model+' '+str(curr_time_idx))


        return graphs_df

    # plot and save methods:
    # ---------------------

    def patient_plotNsave_emotion_over_time(self, patient_prob_tables_urls, model_list, emotion_list, session_hour_range, setup_name):
        # TODO: add documentation
        for patient_prob_table_url in patient_prob_tables_urls:
            # loading data if available:
            try:
                prob_table = pd.read_csv(patient_prob_table_url)
            except:
                print("File not avalble in: "+patient_prob_table_url)
            # fix numbers loaded as str:
            for emotion in emotion_list:
                if(type(prob_table[emotion][0]) == str):
                    prob_table[emotion] = prob_table[emotion].apply(pd.to_numeric, errors='coerce')


            # get graph_df:
            if (session_hour_range == []) or (type(session_hour_range) != int):
                session_hour_range = 1
                print("set session_hour_range to default value of 1 hour")
            graph_df = self.get_table_by_session(prob_table, session_hour_range, session_action='mean')

            patient_id = graph_df["Patient_id"].iloc[0]
            # ensure data in the right format:
            if (model_list == []) or (type(model_list[0]) != str):
                model_list = graph_df.Model.unique()
                # sys.warning("using all available models")


            # remove unsupported models:
            model_list = [val for idx, val in enumerate(self.supported_models) if val in model_list]
            emotion_list = [val for idx, val in enumerate(self.suported_emotions) if val in emotion_list]

            for model in model_list:
                model_graphs_df = graph_df[graph_df['Model'] == model]
                for emotion in emotion_list:
                    # plot:
                    fig, ax = plt.subplots(num=None, figsize=(20, 10), dpi=200, facecolor='w', edgecolor='k')
                    x = model_graphs_df['Date']
                    y = model_graphs_df[emotion]
                    ax.plot(x, y, linestyle='--', marker='o', color='black')
                    ax.fill_between(x, 0, 1, where=model_graphs_df['IsWet'],
                                    color='aqua', alpha=0.4, transform=ax.get_xaxis_transform())
                    ax.legend(["mean sessions probabilty for emotion", "wet sessions"])
                    fig.suptitle('Mean Sessions Probability\n'+'trained with '+str(len(emotion_list))+' classes\n'+'Patient: '+patient_id+', Model: '+model+', Emotion: '+emotion, fontsize=16)
                    plt.xlabel('Date\n(may be multiple sessions in one dates - different hours)')
                    ax.xaxis.set_major_locator(plt.MaxNLocator(30))     # reducing number of plot ticks
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)   # rotate plot tics
                    plt.grid()
                    plt.ylabel(emotion+' mean session probability')

                    # save fig:
                    save_url_path = "results_tablesOfProb\\"+setup_name+"\\"+patient_id+"\\"+model
                    Path(save_url_path).mkdir(parents=True, exist_ok=True)
                    save_file_name = 'MeanSessionsProbability'+'_trainedWith'+str(len(emotion_list))+'Classes'+'_'+patient_id+'_'+model+'_'+emotion
                    # manager = plt.get_current_fig_manager()
                    # manager.window.showMaximized()
                    if os.path.isfile(save_url_path+"\\"+save_file_name+".png"):
                        os.remove(save_url_path+"\\"+save_file_name+".png")
                    plt.ioff()
                    fig.savefig(save_url_path+"\\"+save_file_name+".png", bbox_inches='tight')
                    plt.close(fig)

    def patient_plotNsave_sum_histogram(self, patient_prob_tables_urls, model_list, emotion_list, session_hour_range,setup_name):
        #TODO: add a description

        for patient_prob_table_url in patient_prob_tables_urls:
            # loading data if available:
            try:
                prob_table = pd.read_csv(patient_prob_table_url)
            except:
                print("File not avalble in: " + patient_prob_table_url)
            # fix numbers loaded as str:
            for emotion in emotion_list:
                if (type(prob_table[emotion][0]) == str):
                    prob_table[emotion] = prob_table[emotion].apply(pd.to_numeric, errors='coerce')

            # get graph_df:
            if (session_hour_range == []) or (type(session_hour_range) != int):
                session_hour_range = 1
                print("set session_hour_range to default value of 1 hour")
            graph_df = self.get_table_by_session(prob_table, session_hour_range, session_action='mean', emotions_list=emotion_list)

            patient_id = graph_df["Patient_id"].iloc[0]
            # ensure data in the right format:
            if (model_list == []) or (type(model_list[0]) != str):
                model_list = graph_df.Model.unique()
                # print("using all available models")

            # remove unsupported models:
            model_list = [val for idx, val in enumerate(self.supported_models) if val in model_list]
            emotion_list = [val for idx, val in enumerate(self.suported_emotions) if val in emotion_list]

            for model in model_list:
                model_graphs_df = graph_df[graph_df['Model'] == model]
                model_graphs_df_summed_row = model_graphs_df.sum(axis=0, numeric_only=True, skipna=True)
                model_graphs_df_std_row = model_graphs_df.std(axis=0, numeric_only=True, skipna=True)

                #plot:
                model_graphs_df_summed_row.hist()
                fig, ax = plt.subplots(num=None, figsize=(20, 10), dpi=200, facecolor='w', edgecolor='k')
                x = model_graphs_df[emotion_list].columns
                y = model_graphs_df_summed_row
                ax.bar(x, y, yerr=model_graphs_df_std_row)
                # ax.legend(["mean sessions probabilty for emotion", "wet sessions"])
                fig.suptitle('Sum of all soft decision scores for all recordings per emotion\n'+'trained with ' + str(len(emotion_list)) + ' classes\n'+ 'Patient: ' + patient_id + ', Model: ' + model,fontsize=16)
                plt.xlabel('Emotions/Classes')
                plt.grid()
                plt.ylabel('Mean probability for all emotions')

                # save fig:
                save_url_path = "results_tablesOfProb\\" + setup_name + "\\" + patient_id + "\\" + model
                Path(save_url_path).mkdir(parents=True, exist_ok=True)
                save_file_name = 'SumOfAllSoftDecisionScoresForAllRecordingsPerEmotion' + '_trainedWith' + str(
                    len(emotion_list)) + 'Classes' + '_' + patient_id + '_' + model
                # manager = plt.get_current_fig_manager()
                # manager.window.showMaximized()
                if os.path.isfile(save_url_path + "\\" + save_file_name + ".png"):
                    os.remove(save_url_path + "\\" + save_file_name + ".png")
                plt.ioff()
                fig.savefig(save_url_path + "\\" + save_file_name + ".png", bbox_inches='tight')

                plt.close(fig)

    def patient_plotNsave_count_hard_decision_histogram(self, patient_prob_tables_urls, model_list, emotion_list, session_hour_range,setup_name):
        #TODO: add a description

        for patient_prob_table_url in patient_prob_tables_urls:
            # loading data if available:
            try:
                prob_table = pd.read_csv(patient_prob_table_url)
            except:
                print("File not avalble in: " + patient_prob_table_url)
            # fix numbers loaded as str:
            for emotion in emotion_list:
                if (type(prob_table[emotion][0]) == str):
                    prob_table[emotion] = prob_table[emotion].apply(pd.to_numeric, errors='coerce')

            # get graph_df:
            if (session_hour_range == []) or (type(session_hour_range) != int):
                session_hour_range = 1
                print("set session_hour_range to default value of 1 hour")
            graph_df = self.get_table_by_session(prob_table, session_hour_range, session_action='mean', emotions_list=emotion_list)

            patient_id = graph_df["Patient_id"].iloc[0]
            # ensure data in the right format:
            if (model_list == []) or (type(model_list[0]) != str):
                model_list = graph_df.Model.unique()
                # print("using all available models")

            # remove unsupported models:
            model_list = [val for idx, val in enumerate(self.supported_models) if val in model_list]
            emotion_list = [val for idx, val in enumerate(self.suported_emotions) if val in emotion_list]

            for model in model_list:
                model_graphs_df = graph_df[graph_df['Model'] == model]
                model_graphs_df_std_row = model_graphs_df[emotion_list].idxmax(axis=1, skipna=True)
                from collections import Counter
                histogram_dict = Counter(model_graphs_df_std_row)
                #plot:
                fig, ax = plt.subplots(num=None, figsize=(20, 10), dpi=200, facecolor='w', edgecolor='k')
                x = emotion_list
                y=list()
                for emo in emotion_list: y.append(histogram_dict[emo])
                ax.bar(x, y)
                # ax.legend(["mean sessions probabilty for emotion", "wet sessions"])
                fig.suptitle('Count of hard decision for all recordings per emotion\n'+'trained with ' + str(len(emotion_list)) + ' classes\n'+ 'Patient: ' + patient_id + ', Model: ' + model,fontsize=16)
                plt.xlabel('Emotions/Classes')
                plt.grid()
                plt.ylabel('Count of hard decision for each emotions')

                # save fig:
                save_url_path = "results_tablesOfProb\\" + setup_name + "\\" + patient_id + "\\" + model
                Path(save_url_path).mkdir(parents=True, exist_ok=True)
                save_file_name = 'CountOfHardDecisionForAllRecordingsPerEmotion' + '_trainedWith' + str(
                    len(emotion_list)) + 'Classes' + '_' + patient_id + '_' + model
                # manager = plt.get_current_fig_manager()
                # manager.window.showMaximized()
                if os.path.isfile(save_url_path + "\\" + save_file_name + ".png"):
                    os.remove(save_url_path + "\\" + save_file_name + ".png")
                plt.ioff()
                fig.savefig(save_url_path + "\\" + save_file_name + ".png", bbox_inches='tight')

                plt.close(fig)

    def patient_plotNsave_mean_prob_session_emotion_3d(self, patient_prob_tables_urls, model_list, emotion_list, session_hour_range, setup_name):
        #TODO: add description

        for patient_prob_table_url in patient_prob_tables_urls:
            # loading data if available:
            try:
                prob_table = pd.read_csv(patient_prob_table_url)
            except:
                print("File not avalble in: "+patient_prob_table_url)
            # fix numbers loaded as str:
            for emotion in emotion_list:
                if(type(prob_table[emotion][0]) == str):
                    prob_table[emotion] = prob_table[emotion].apply(pd.to_numeric, errors='coerce')


            # get graph_df:
            if (session_hour_range == []) or (type(session_hour_range) != int):
                session_hour_range = 1
                print("set session_hour_range to default value of 1 hour")
            # graph_df = self.plot_emotion_over_time(prob_table, session_hour_range)

            patient_id = prob_table["PatientName"].iloc[0]
            # ensure data in the right format:
            if (model_list == []) or (type(model_list[0]) != str):
                model_list = prob_table.Model.unique()
                # print("using all available models")


            # remove unsupported models:
            model_list = [val for idx, val in enumerate(self.supported_models) if val in model_list]
            emotion_list = [val for idx, val in enumerate(self.suported_emotions) if val in emotion_list]

            for model in model_list:
                model_graphs_df = prob_table[prob_table['Model'] == model]
                model_graphs_df['Date'] = pd.to_datetime(model_graphs_df['Date'], format="%d/%m/%Y")
                model_graphs_mean_by_date_df = model_graphs_df.resample('d', on='Date').mean().dropna(how='all')
                # add IsWet to model_graphs_mean_by_date_df
                model_graphs_mean_by_date_df['IsWet'] = ""
                for date in model_graphs_mean_by_date_df.index.values:
                    model_graphs_mean_by_date_df['IsWet'][date] = (
                                model_graphs_df['ClinicalStatus'][model_graphs_df['Date'] == date] == 'wet').any()
                emotion_list_with_clinical_status = emotion_list.copy()
                emotion_list_with_clinical_status.append('IsWet')
                model_graphs_mean_by_date_df_only_emotions = model_graphs_mean_by_date_df[emotion_list_with_clinical_status]
                model_graphs_mean_by_date_df_only_emotions['IsWet'] = model_graphs_mean_by_date_df_only_emotions['IsWet']*model_graphs_mean_by_date_df[emotion_list].values.max()
                model_graphs_mean_by_date_df_only_emotions = model_graphs_mean_by_date_df_only_emotions.astype(float)
                model_graphs_mean_by_date_df_only_emotions = model_graphs_mean_by_date_df_only_emotions.transpose()

                # plot
                fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=False, figsize=(20, 10), dpi=200,
                                       facecolor='w', edgecolor='k')
                ax = sn.heatmap(model_graphs_mean_by_date_df_only_emotions, annot=False)
                ax.xaxis.set_major_locator(plt.MaxNLocator(30))  # reducing number of plot ticks
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)  # rotate plot tics
                # rewrite x labels text:
                labels = [item.get_text() for item in ax.get_xticklabels()]
                for i in range(len(labels)):
                    labels[i] = labels[i][0:10]
                ax.set_xticklabels(labels)
                # color set wet index
                # TODO: color set wet index
                # for i, date in zip(range(len(ax.get_xticklabels())), ax.get_xticklabels()):
                #     # date_dt =
                #     if model_graphs_mean_by_date_df['IsWet'][date]:
                #         print(ax.get_xticklabels()[i])
                #         ax.get_xticklabels()[i].set_color("aqua")
                # ax.fill_between(model_graphs_mean_by_date_df_only_emotions.columns.values, 0, 1, where=model_graphs_mean_by_date_df['IsWet'],
                #                 color='aqua', alpha=0.9, transform=ax.get_xaxis_transform())
                ax.yaxis.set_major_locator(plt.MaxNLocator(len(emotion_list_with_clinical_status)))

                fig.suptitle('Mean Date Probability\n' + 'trained with ' + str(len(
                    emotion_list)) + ' classes\n' + 'Patient: ' + patient_id + ', Model: ' + model)
                plt.xlabel('Date\n(may be multiple sessions in one dates - different hours)')
                ax.set_ylabel('Mean Date Probability')

                # save fig:
                save_url_path = "results_tablesOfProb\\" + setup_name + "\\" + patient_id + "\\" + model
                Path(save_url_path).mkdir(parents=True, exist_ok=True)
                save_file_name = 'MeanSessionsProbabilityAllEmotionsInOneGraphHeatMap' + '_trainedWith' + str(
                    len(emotion_list)) + 'Classes' + '_' + patient_id + '_' + model
                # manager = plt.get_current_fig_manager()
                # manager.window.showMaximized()
                if os.path.isfile(save_url_path + "\\" + save_file_name + ".png"):
                    os.remove(save_url_path + "\\" + save_file_name + ".png")
                plt.ioff()
                fig.savefig(save_url_path + "\\" + save_file_name + ".png", bbox_inches='tight')
                plt.close(fig)

    def patient_plotNsave_emotion_over_time_summerize_for_model_one_plot(self, patient_prob_tables_urls, model_list, emotion_list, session_hour_range, setup_name):
        # TODO: add documentation
        for patient_prob_table_url in patient_prob_tables_urls:
            # loading data if available:
            try:
                prob_table = pd.read_csv(patient_prob_table_url)
            except:
                print("File not avalble in: "+patient_prob_table_url)
            # fix numbers loaded as str:
            for emotion in emotion_list:
                if(type(prob_table[emotion][0]) == str):
                    prob_table[emotion] = prob_table[emotion].apply(pd.to_numeric, errors='coerce')
            patient_id = prob_table["PatientName"].iloc[0]
            # ensure data in the right format:
            if (model_list == []) or (type(model_list[0]) != str):
                model_list = prob_table.Model.unique()
                # print("using all available models")
            # remove unsupported models:
            model_list = [val for idx, val in enumerate(self.supported_models) if val in model_list]
            emotion_list = [val for idx, val in enumerate(self.suported_emotions) if val in emotion_list]
            # add IsWet column:


            for model in model_list:
                model_graphs_df = prob_table[prob_table['Model'] == model] # filter by emotion
                fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=False,  figsize=(20, 10), dpi=200, facecolor='w',
                                       edgecolor='k')
                for emotion in emotion_list:
                    model_graphs_df['Date'] = pd.to_datetime(model_graphs_df['Date'], format="%d/%m/%Y")
                    model_graphs_mean_by_date_df = model_graphs_df.resample('d', on='Date').mean().dropna(how='all')
                    # add IsWet to model_graphs_mean_by_date_df
                    model_graphs_mean_by_date_df['IsWet'] = ""
                    for date in model_graphs_mean_by_date_df.index.values:
                        model_graphs_mean_by_date_df['IsWet'][date] = (model_graphs_df['ClinicalStatus'][model_graphs_df['Date']==date]=='wet').any()
                    # plot:
                    x = model_graphs_mean_by_date_df.index.values
                    y = model_graphs_mean_by_date_df[emotion]
                    ax.plot(x, y, linestyle='--', marker='o', label=emotion)

                    ax.fill_between(x, 0, 1, where=model_graphs_mean_by_date_df['IsWet'],
                                    color='aqua', alpha=0.4, transform=ax.get_xaxis_transform())# plt.xlabel('Date\n(may be multiple sessions in one dates - different hours)')
                    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
                    ax.xaxis.set_major_locator(plt.MaxNLocator(30))     # reducing number of plot ticks
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)   # rotate plot tics

                    # plt.ylabel(emotion+' mean session probability')
                ax.grid()
                fig.legend(loc='best')
                fig.suptitle('Mean Sessions Probability\n' + 'trained with ' + str(len(
                    emotion_list)) + ' classes\n' + 'Patient: ' + patient_id + ', Model: ' + model)
                plt.xlabel('Date\n(may be multiple sessions in one dates - different hours)')
                ax.set_ylabel('Mean Date Probability')

                # save fig:
                save_url_path = "results_tablesOfProb\\"+setup_name+"\\"+patient_id+"\\"+model
                Path(save_url_path).mkdir(parents=True, exist_ok=True)
                save_file_name = 'MeanSessionsProbabilityAllEmotionsInOneGraph'+'_trainedWith'+str(len(emotion_list))+'Classes'+'_'+patient_id+'_'+model
                # manager = plt.get_current_fig_manager()
                # manager.window.showMaximized()
                if os.path.isfile(save_url_path+"\\"+save_file_name+".png"):
                    os.remove(save_url_path+"\\"+save_file_name+".png")
                plt.ioff()
                fig.savefig(save_url_path+"\\"+save_file_name+".png", bbox_inches='tight')
                plt.close(fig)

    # def get_model_variance_per_patient_notGood(self, patient_prob_tables_urls, model_list, emotion_list, session_hour_range, setup_name, multi_graph):
    #     """
    #     Description: the function gets number of patients, models and emotion classes. the function calculate the
    #     mean variance for each session and plot this value over time(date).
    #
    #     Input:
    #         :parm patient_prob_tables_urls: list of urls to each patiant .wav files
    #         :type patient_prob_tables_urls:
    #         :parm model_list: list of scikit learn models. should be set to output probabilities
    #         :type model_list:
    #         :parm emotion_list: list of emotion classes
    #         :type emotion_list:
    #         :param setup_name: string that describes the current setup under which the models where trained by
    #         :type setup_name:
    #     """
    #     # TODO: complete documentation
    #     # TODO: finish function
    #     for patient_prob_table_url in patient_prob_tables_urls:
    #         # loading data if available:
    #         try:
    #             prob_table = pd.read_csv(patient_prob_table_url)
    #         except:
    #             print("File not avalble in: "+patient_prob_table_url)
    #         # fix numbers loaded as str:
    #         for emotion in emotion_list:
    #             if(type(prob_table[emotion][0]) == str):
    #                 prob_table[emotion] = prob_table[emotion].apply(pd.to_numeric, errors='coerce')
    #         patient_id = prob_table["PatientName"].iloc[0]
    #         # ensure data in the right format:
    #         if (model_list == []) or (type(model_list[0]) != str):
    #             model_list = prob_table.Model.unique()
    #             print("using all available models")
    #         # remove unsupported models:
    #         model_list = [val for idx, val in enumerate(self.supported_models) if val in model_list]
    #         emotion_list = [val for idx, val in enumerate(self.suported_emotions) if val in emotion_list]
    #         # add IsWet column:
    #
    #         # get graph_df:
    #         if (session_hour_range == []) or (type(session_hour_range) != int):
    #             session_hour_range = 1
    #             print("set session_hour_range to default value of 1 hour")
    #         prob_table_by_session_df = self.get_table_by_session(prob_table, session_hour_range, session_action='mean')
    #
    #         for model in model_list:
    #             prob_table_by_sessionNmodel_df = prob_table_by_session_df[prob_table_by_session_df['Model'] == model] # filter by model
    #             prob_table_by_sessionNmodel_df['std'] = prob_table_by_sessionNmodel_df.std(axis=1, numeric_only=True, skipna=True)
    #             fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=False,  figsize=(20, 10), dpi=200, facecolor='w',
    #                                    edgecolor='k')
    #             prob_table_by_sessionNmodel_df = prob_table_by_sessionNmodel_df.reset_index()
    #             x = prob_table_by_sessionNmodel_df['Date']
    #             y = prob_table_by_sessionNmodel_df['std']
    #             ax.plot(x, y, linestyle='--', marker='o', label=emotion)
    #             ax.fill_between(x, 0, 1, where=prob_table_by_sessionNmodel_df['IsWet'],
    #                             color='aqua', alpha=0.4,
    #                             transform=ax.get_xaxis_transform())  # plt.xlabel('Date\n(may be multiple sessions in one dates - different hours)')
    #             ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    #             ax.xaxis.set_major_locator(plt.MaxNLocator(30))  # reducing number of plot ticks
    #             plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)  # rotate plot tics
    #             ax.grid()
    #             fig.suptitle('Mean Sessions STD Over All Emotions\Classes\n' + 'trained with ' + str(len(
    #                 emotion_list)) + ' classes\n' + 'Patient: ' + patient_id + ', Model: ' + model)
    #             plt.xlabel('Date\n(may be multiple sessions in one dates - different hours)')
    #             ax.set_ylabel('Mean Sessions STD')
    #
    #
    #         # sublot implamentation:
    #         number_of_subplots = len(model_list)
    #         subplots_columns = 4
    #         number_of_rows = number_of_subplots // subplots_columns
    #         number_of_rows += (number_of_subplots % subplots_columns)>0
    #         Position = range(1, number_of_subplots + 1)
    #
    #         Position = range(1, number_of_subplots + 1)
    #         # fig = plt.figure(figsize=(20, 10), dpi=200, facecolor='w', edgecolor='k')
    #         # fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
    #         fig, ax= plt.subplots(nrows=number_of_rows, ncols=subplots_columns, sharex=True, sharey=False, figsize=(20, 10), dpi=200, facecolor='w',
    #                      edgecolor='k')
    #         model_idx=0
    #         for row in range(number_of_rows):
    #             for col in range(subplots_columns):
    #                 if model_idx > len(model_list)-1:
    #                     break
    #                 model = model_list[model_idx]
    #                 prob_table_by_sessionNmodel_df = prob_table_by_session_df[
    #                     prob_table_by_session_df['Model'] == model]  # filter by model
    #                 prob_table_by_sessionNmodel_df['std'] = prob_table_by_sessionNmodel_df.std(axis=1,
    #                                                                                            numeric_only=True,
    #                                                                                            skipna=True)
    #
    #                 prob_table_by_sessionNmodel_df = prob_table_by_sessionNmodel_df.reset_index()
    #                 x = prob_table_by_sessionNmodel_df['Date']
    #                 y = prob_table_by_sessionNmodel_df['std']
    #                 ax[row, col].plot(x, y, linestyle='--', marker='o', label=emotion)
    #                 ax[row, col].fill_between(x, 0, 1, where=prob_table_by_sessionNmodel_df['IsWet'],
    #                                 color='aqua', alpha=0.4,
    #                                 transform=ax[row, col].get_xaxis_transform())  # plt.xlabel('Date\n(may be multiple sessions in one dates - different hours)')
    #                 ax[row, col].yaxis.set_major_locator(plt.MaxNLocator(4))
    #                 ax[row, col].xaxis.set_major_locator(plt.MaxNLocator(20))  # reducing number of plot ticks
    #                 plt.setp(ax[row, col].xaxis.get_majorticklabels(), rotation=30)  # rotate plot tics
    #                 ax[row, col].grid()
    #                 plt.xlabel('Date\n(may be multiple sessions in one dates - different hours)')
    #                 ax[row, col].set_ylabel('Mean Sessions STD')
    #                 ax[row, col].title.set_text('Mean Sessions STD Over All Emotions\Classes\n' + 'trained with ' + str(len(
    #                     emotion_list)) + ' classes\n' + 'Patient: ' + patient_id + ', Model: ' + model)
    #
    #                 model_idx = model_idx + 1
    #         fig.tight_layout(pad=4.0)
    #
    #
    #             # save fig:
    #             save_url_path = "results_tablesOfProb\\" + setup_name + "\\" + patient_id + "\\" + model
    #             save_file_name = 'MeanSessionsSTDOverAllEmotions' + '_trainedWith' + str(
    #                 len(emotion_list)) + 'Classes' + '_' + patient_id + '_' + model
    #             self.SaveFig(fig, save_url_path, save_file_name, add_datetime=False, close_fig=True)


    def get_model_variance_per_patientNmodel_heatmap(self, patient_prob_tables_urls, model_list, emotion_list, session_hour_range, setup_name):
        """
        Description: the function gets number of patients, models and emotion classes. the function calculate the
        mean variance for each session and plot this value over time(date).

        Input:
            :parm patient_prob_tables_urls: list of urls to each patiant .wav files
            :type patient_prob_tables_urls:
            :parm model_list: list of scikit learn models. should be set to output probabilities
            :type model_list:
            :parm emotion_list: list of emotion classes
            :type emotion_list:
            :param setup_name: string that describes the current setup under which the models where trained by
            :type setup_name:
        """
        # TODO: complete documentation
        # TODO: finish function
        for patient_prob_table_url in patient_prob_tables_urls:
            # loading data if available:
            try:
                prob_table = pd.read_csv(patient_prob_table_url)
                # remove rows with nan:
                prob_table = prob_table[prob_table[emotion_list[0]].notna()]
                # remap df index:
                prob_table = prob_table.reset_index()
            except:
                print("File not avalble in: " + patient_prob_table_url)
            # fix numbers loaded as str:
            for emotion in emotion_list:
                if (type(prob_table[emotion][0]) == str):
                    prob_table[emotion] = prob_table[emotion].apply(pd.to_numeric, errors='coerce')
            # remove rows with nan:
            prob_table = prob_table[prob_table[emotion_list[0]].notna()]
            # remap df index:
            prob_table = prob_table.reset_index()
            # get patient id
            patient_id = prob_table["PatientName"].iloc[0]
            # ensure data in the right format:
            if (model_list == []) or (type(model_list[0]) != str):
                model_list = prob_table.Model.unique()
                # print("using all available models")
            # remove unsupported models:
            model_list = [val for idx, val in enumerate(self.supported_models) if val in model_list]
            emotion_list = [val for idx, val in enumerate(self.suported_emotions) if val in emotion_list]
            # add IsWet column:

            # get graph_df:
            if (session_hour_range == []) or (type(session_hour_range) != int):
                session_hour_range = 1
                print("set session_hour_range to default value of 1 hour")
            prob_table_by_session_df = self.get_table_by_session(prob_table, session_hour_range, session_action='std', emotions_list=emotion_list)

            for model in model_list:
                # cut model graph:
                prob_table_by_sessionNmodel_df = prob_table_by_session_df[prob_table_by_session_df['Model']==model]
                prob_table_by_sessionNmodel_df = prob_table_by_sessionNmodel_df.reset_index()
                prob_table_by_sessionNmodel_df['Date'] = pd.to_datetime(prob_table_by_sessionNmodel_df['Date'], format="%d/%m/%Y")
                prob_table_by_sessionNmodel_df = prob_table_by_sessionNmodel_df.set_index('Date')
                emotion_list_with_clinical_status = emotion_list.copy()
                emotion_list_with_clinical_status.append('IsWet')
                prob_table_by_sessionNmodel_df = prob_table_by_sessionNmodel_df[emotion_list_with_clinical_status]
                prob_table_by_sessionNmodel_df = prob_table_by_sessionNmodel_df.astype(float)
                prob_table_by_sessionNmodel_df = prob_table_by_sessionNmodel_df.transpose()

                # plot
                fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=False, figsize=(20, 10), dpi=200,
                                       facecolor='w', edgecolor='k')
                ax = sn.heatmap(prob_table_by_sessionNmodel_df, annot=False)
                ax.xaxis.set_major_locator(plt.MaxNLocator(30))  # reducing number of plot ticks
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)  # rotate plot tics
                # rewrite x labels text:
                labels = [item.get_text() for item in ax.get_xticklabels()]
                for i in range(len(labels)):
                    labels[i] = labels[i][0:10]
                ax.set_xticklabels(labels)
                # color set wet index
                # TODO: color set wet index
                # for i, date in zip(range(len(ax.get_xticklabels())), ax.get_xticklabels()):
                #     # date_dt =
                #     if model_graphs_mean_by_date_df['IsWet'][date]:
                #         print(ax.get_xticklabels()[i])
                #         ax.get_xticklabels()[i].set_color("aqua")
                # ax.fill_between(model_graphs_mean_by_date_df_only_emotions.columns.values, 0, 1, where=model_graphs_mean_by_date_df['IsWet'],
                #                 color='aqua', alpha=0.9, transform=ax.get_xaxis_transform())
                ax.yaxis.set_major_locator(plt.MaxNLocator(len(emotion_list_with_clinical_status)))

                fig.suptitle('Session STD per Emotion\Class\n' + 'trained with ' + str(len(
                    emotion_list)) + ' classes\n' + 'Patient: ' + patient_id + ', Model: ' + model)
                plt.xlabel('Date\n(may be multiple sessions in one dates - different hours)')
                ax.set_ylabel('STD per Emotion')

                # save fig:
                save_url_path = "results_tablesOfProb\\" + setup_name + "\\" + patient_id + "\\" + model
                save_file_name = 'MeanSessionsStdOverAllEmotionsHeatMap' + '_trainedWith' + str(
                    len(emotion_list)) + 'Classes' + '_' + patient_id + '_' + model
                self.SaveFig(fig, save_url_path, save_file_name, add_datetime=False, close_fig=True)

    def get_model_variance_per_patient_all_models_one_graph(self, patient_prob_tables_urls, model_list, emotion_list, session_hour_range, setup_name):
        for patient_prob_table_url in patient_prob_tables_urls:
            # loading data if available:
            try:
                prob_table = pd.read_csv(patient_prob_table_url)
            except:
                print("File not avalble in: " + patient_prob_table_url)
            # fix numbers loaded as str:
            for emotion in emotion_list:
                if (type(prob_table[emotion][0]) == str):
                    prob_table[emotion] = prob_table[emotion].apply(pd.to_numeric, errors='coerce')
            patient_id = prob_table["PatientName"].iloc[0]
            # ensure data in the right format:
            if (model_list == []) or (type(model_list[0]) != str):
                model_list = prob_table.Model.unique()
                # print("using all available models")
            # remove unsupported models:
            model_list = [val for idx, val in enumerate(self.supported_models) if val in model_list]
            emotion_list = [val for idx, val in enumerate(self.suported_emotions) if val in emotion_list]
            # add IsWet column:

            # get graph_df:
            if (session_hour_range == []) or (type(session_hour_range) != int):
                session_hour_range = 1
                print("set session_hour_range to default value of 1 hour")
            prob_table_by_session_df = self.get_table_by_session(prob_table, session_hour_range, session_action='std')

            fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=False, figsize=(20, 10), dpi=200,
                                   facecolor='w', edgecolor='k')

            for model in model_list:
                # cut model graph:
                prob_table_by_sessionNmodel_df = prob_table_by_session_df[prob_table_by_session_df['Model'] == model]
                prob_table_by_sessionNmodel_df = prob_table_by_sessionNmodel_df.reset_index()
                prob_table_by_sessionNmodel_df['Date'] = pd.to_datetime(prob_table_by_sessionNmodel_df['Date'],
                                                                        format="%d/%m/%Y")

                #mean of std's over all emotions:
                prob_table_by_sessionNmodel_df['MeanStd'] = prob_table_by_sessionNmodel_df.std(axis=1, numeric_only=True, skipna=True)

                x = prob_table_by_sessionNmodel_df['Date']
                y = prob_table_by_sessionNmodel_df['MeanStd']
                ax.plot(x, y, linestyle='--', marker='o', label=model)
                ax.fill_between(x, 0, 1, where=prob_table_by_sessionNmodel_df['IsWet'],
                                color='aqua', alpha=0.4,
                                transform=ax.get_xaxis_transform())  # plt.xlabel('Date\n(may be multiple sessions in one dates - different hours)')
                ax.yaxis.set_major_locator(plt.MaxNLocator(4))
                ax.xaxis.set_major_locator(plt.MaxNLocator(30))  # reducing number of plot ticks
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)  # rotate plot tics
                ax.grid()
                fig.suptitle('Mean Sessions STD Over All Emotions\Classes\n' + 'trained with ' + str(len(
                    emotion_list)) + ' classes\n' + 'Patient: ' + patient_id + ', Model: ' + model)
                plt.xlabel('Date\n(may be multiple sessions in one dates - different hours)')
                ax.set_ylabel('Mean Sessions STD')
            fig.legend()

            # save fig:
            save_url_path = "results_tablesOfProb\\" + setup_name + "\\" + patient_id
            save_file_name = 'MeanSessionsStdOverAllEmotionsAllModels' + '_trainedWith' + str(
                len(emotion_list)) + 'Classes' + '_' + patient_id + '_' + model
            self.SaveFig(fig, save_url_path, save_file_name, add_datetime=False, close_fig=True)

    def get_model_variance_per_patient_all_models_multiple_graph(self, patient_prob_tables_urls, model_list, emotion_list, session_hour_range, setup_name):
        # progress bar initialization:
        widgets = [FormatLabel('<<<all patient process>>>'), ' ', Percentage(), ' ', Bar('#'), ' ', RotatingMarker()]
        progressbar = ProgressBar(widgets=widgets, maxval=len(patient_prob_tables_urls))
        progressbar.start()
        for i_pb, patient_prob_table_url in zip(range(len(patient_prob_tables_urls)), patient_prob_tables_urls):
            # progress bar update:
            widgets[0] = FormatLabel('<filename-{0}>'.format(i_pb))
            progressbar.update(i_pb)

            # loading data if available:
            try:
                prob_table = pd.read_csv(patient_prob_table_url)
            except:
                print("File not avalble in: " + patient_prob_table_url)
            # fix numbers loaded as str:
            for emotion in emotion_list:
                if (type(prob_table[emotion][0]) == str):
                    prob_table[emotion] = prob_table[emotion].apply(pd.to_numeric, errors='coerce')
            patient_id = prob_table["PatientName"].iloc[0]
            # ensure data in the right format:
            if (model_list == []) or (type(model_list[0]) != str):
                model_list = prob_table.Model.unique()
                # print("using all available models")
            # remove unsupported models:
            model_list = [val for idx, val in enumerate(self.supported_models) if val in model_list]
            emotion_list = [val for idx, val in enumerate(self.suported_emotions) if val in emotion_list]
            # add IsWet column:

            # get graph_df:
            if (session_hour_range == []) or (type(session_hour_range) != int):
                session_hour_range = 1
                print("set session_hour_range to default value of 1 hour")
            prob_table_by_session_df = self.get_table_by_session(prob_table, session_hour_range, session_action='std')

            fig, ax = plt.subplots(nrows=len(model_list), ncols=1, sharex=True, sharey=False, figsize=(20, 10), dpi=200,
                                   facecolor='w', edgecolor='k')

            for model, i in zip(model_list, range(len(model_list))):
                # cut model graph:
                prob_table_by_sessionNmodel_df = prob_table_by_session_df[prob_table_by_session_df['Model'] == model]
                prob_table_by_sessionNmodel_df = prob_table_by_sessionNmodel_df.reset_index()
                prob_table_by_sessionNmodel_df['Date'] = pd.to_datetime(prob_table_by_sessionNmodel_df['Date'],
                                                                        format="%d/%m/%Y")

                # mean of std's over all emotions:
                prob_table_by_sessionNmodel_df['MeanStd'] = prob_table_by_sessionNmodel_df.std(axis=1,
                                                                                               numeric_only=True,
                                                                                               skipna=True)

                x = prob_table_by_sessionNmodel_df['Date']
                y = prob_table_by_sessionNmodel_df['MeanStd']
                ax[i].plot(x, y, linestyle='--', marker='x', label=model)
                ax[i].fill_between(x, 0, 1, where=prob_table_by_sessionNmodel_df['IsWet'],
                                color='aqua', alpha=0.4,
                                transform=ax[i].get_xaxis_transform())  # plt.xlabel('Date\n(may be multiple sessions in one dates - different hours)')
                ax[i].yaxis.set_major_locator(plt.MaxNLocator(4))
                ax[i].xaxis.set_major_locator(plt.MaxNLocator(30))  # reducing number of plot ticks
                plt.setp(ax[i].xaxis.get_majorticklabels(), rotation=30)  # rotate plot tics
                ax[i].grid()
                fig.suptitle('Mean Sessions STD Over All Emotions\Classes\n' + 'trained with ' + str(len(
                    emotion_list)) + ' classes\n' + 'Patient: ' + patient_id + ', Model: ' + model)
                plt.xlabel('Date\n(may be multiple sessions in one dates - different hours)')
                ax[i].set_ylabel(model, rotation=83, color='royalblue')
            subplots_adjust(hspace=0.000)
            # save fig:
            save_url_path = "results_tablesOfProb\\" + setup_name + "\\" + patient_id
            save_file_name = 'MeanSessionsStdOverAllEmotionsAllModelsMultipleGraph' + '_trainedWith' + str(
                len(emotion_list)) + 'Classes' + '_' + patient_id + '_' + model
            self.SaveFig(fig, save_url_path, save_file_name, add_datetime=False, close_fig=True)
        progressbar.finish()