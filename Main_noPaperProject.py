import sys

sys.path.append("D:\work\code\Python\CordioAlgorithmsPython\infrastructure")
# sys.path.insert(0, "D:\work\code\Python\CordioAlgorithmsPython\infrastructure")
p = "D:\work\db\BSV\BSV-0006\BSV-0006_190520_094113_S0010_he_1.25_SM-J400F_Android26.wav"

import os, glob, sys
import CordioFile
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from emotion_recognition import EmotionRecognizer

model_list = [SVC(probability=True), AdaBoostClassifier(), RandomForestClassifier(), GradientBoostingClassifier(),
              DecisionTreeClassifier(), KNeighborsClassifier(), MLPClassifier()]
emotions_list = ['sad', 'neutral', 'happy']

patientDir_path = "D:\work\db\RAM\RAM-0071"

class CordioESP:
    ''' Expressive Speech Processing '''

    # init CordioESP
    super().__init__(None, **kwargs)

    self.n_rnn_layers = kwargs.get("n_rnn_layers", 2)
    self.n_dense_layers = kwargs.get("n_dense_layers", 2)
    self.rnn_units = kwargs.get("rnn_units", 128)
    self.dense_units = kwargs.get("dense_units", 128)
    self.cell = kwargs.get("cell", LSTM)

    def predict_all_proba_for_patient(patientDir_path, model_list, emotions_list):
        # get all wavs:
        all_wavs = glob.glob(os.path.join(patientDir_path, '*.wav'))
        num_of_wav = len(all_wavs) * len(model_list)

        # create basic information table for patient:
        patient_info_column_names = ["PatientName", "Date", "Time", "sentence", "Language"]
        patient_df = pd.DataFrame(columns=patient_info_column_names+['Model'] + emotions_list)

        for model in model_list:
            patient_df.append(predict_all_proba_for_patientNmodel(model, emotions_list, all_wavs))

    def modelTrain(model, emotions_list):
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

    def modelPredict(rec, wav_url):
        try:
            out = rec.predict_proba(wav_url)
        except ValueError:
            out = type(rec.model).__name__ + " doesnt support in predict_proba"
        return out

    def predict_all_proba_for_patientNmodel(model, emotions_list, all_wavs):

        df_len = len(all_wavs)
        patientNmodel_df = pd.DataFrame(index=np.arange(df_len),
                                        columns=patient_info_column_names + ['Model'] + emotions_list)

        rec = modelTrain(model, emotions_list)

        for (i, wav) in zip(range(df_len), all_wavs):
            # add  soft decision score for each emotion
            patientNmodel_df.loc[i] = modelPredict(rec, wav)

            # insert basic information:
            p = Path(str(wav))
            fileHandle = CordioFile
            patientNmodel_df.at[i, "PatientName"] = fileHandle.CordioExtractPatient(p)
            patientNmodel_df.at[i, "Date"] = fileHandle.CordioExtractRecordingDateTime(p).strftime("%d/%m/%Y")
            patientNmodel_df.at[i, "Time"] = fileHandle.CordioExtractRecordingDateTime(p).strftime("%H:%M:%S")
            patientNmodel_df.at[i, "sentence"] = fileHandle.CordioExtractSentence(p)
            patientNmodel_df.at[i, "Language"] = fileHandle.CordioExtractLanguage(p)
            patientNmodel_df.at[i, "Model"] = type(model).__name__

        return patientNmodel_df
