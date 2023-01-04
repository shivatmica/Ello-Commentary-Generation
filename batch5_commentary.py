#!pip install openai --quiet
import os
import pandas as pd
import numpy as np
import openai
from getpass import getpass

OPENAI_KEY = getpass('Enter API Key: ')
openai.api_key = OPENAI_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_KEY


class CompileTrainData:
    def __init__(self):
        self.path = '/content/drive/MyDrive/Stanford Project Book Bundles'
        self.scored_path = '/content/drive/MyDrive/Stanford Project Book Bundles/generated_commentary_LS scoring_and_feedback.xlsx'
        self.test_path = '/content/drive/MyDrive/Stanford Project Book Bundles/batch5/generated_commentary'
        self.num_to_model = {1: 'curie_commentary (temperature 0.7)', 2: 'curie_commentary  (temperature 0.1)',
                             3: 'davinci_commentary (temperature 0.7)', 4: 'davinci_commentary (temperature 0.1)'}
        self.valid_paths = []
        self.comm_paths = []
        self.reject_names = ['9780062381828.xlsx', '9780545825030.xlsx', '9780547076690.xlsx', '9780142414538.xlsx']
        self.data = []
        self.main_data = pd.DataFrame()
        self.scored = pd.DataFrame()
        self.test_data = pd.DataFrame()
        self.model = 'davinci:ft-ello-2023-01-03-09-46-46'

    def populate_valid_paths(self) -> None:
        for file in os.scandir(self.path):
            if 'batch5' not in file.path and file.is_dir():
                self.valid_paths.append(file.path)

    def populate_comm_paths(self) -> None:
        for path in self.valid_paths:
            for file in os.scandir(path):
                if 'commentary' in file.path:
                    self.comm_paths.append(file.path)

    def preprocess_dataframe_train(self, df) -> pd.DataFrame:
        tmp = pd.DataFrame(index=list(range(len(df))),
                           columns=['Text', 'Commentary'])
        for i in range(1, len(df), 2):
            if i > len(df):
                break

            if str(df.loc[i - 1, 'Text']).lower() == 'nan':
                tmp.loc[i - 1, 'Text'] = df.loc[i, 'Text']

            if str(df.loc[i, 'Text']).lower() == 'nan':
                tmp.loc[i - 1, 'Text'] = df.loc[i - 1, 'Text']
            else:
                tmp.loc[i - 1, 'Text'] = str(df.loc[i - 1, 'Text']) + '\n\n' + str(df.loc[i, 'Text'])

            tmp.loc[i - 1, 'Commentary'] = str(df.loc[i, 'Commentary']) + ' END'
            tmp = tmp[tmp['Commentary'] != 'P END']
            tmp = tmp[tmp['Commentary'] != 'Praise END']

        tmp.dropna(inplace=True)
        return tmp

    def preprocess_dataframe_test(self, df) -> pd.DataFrame:
        tmp = pd.DataFrame(index=list(range(len(df))),
                           columns=['prompt', 'generated_commentary'])

        for i in range(0, len(df) - 1, 2):
            if str(df.loc[i, 'prompt']).lower() == 'nan':
                tmp.loc[i, 'prompt'] = df.loc[i + 1, 'prompt']

            elif str(df['prompt'][i + 1]).lower() == 'nan':
                tmp.loc[i, 'prompt'] = df.loc[i, 'prompt']

            else:
                tmp.loc[i, 'prompt'] = str(df.loc[i, 'prompt']) + str(df.loc[i + 1, 'prompt'])

            if str(df.loc[i + 1, 'generated_commentary']) == 'nan':
                tmp.loc[i, 'generated_commentary'] = 'to generate'

        tmp.dropna(inplace=True)
        return tmp

    def generate_main_data(self) -> None:
        self.populate_valid_paths()
        self.populate_comm_paths()

        for path in self.comm_paths:
            for file in os.scandir(path):
                if file.path[71:] in self.reject_names:
                    continue

                df = pd.read_excel(file.path)

                if len(list(df.columns)) != 2:
                    df.drop(df.columns[0], axis=1, inplace=True)
                    df = df[[df.columns[0], df.columns[1]]]

                if list(df.columns) != ['Text', 'Commentary']:
                    df = df.shift(1)
                    df.loc[0] = df.columns
                    df.columns = ['Text', 'Commentary']

                tmp = self.preprocess_dataframe_train(df)
                self.data.append(tmp)

        self.main_data = pd.concat(self.data)
        self.main_data.columns = ['prompt', 'completion']
        self.main_data.index = list(range(len(self.main_data)))

        self.scored = pd.read_excel(self.scored_path)
        for i in range(len(self.scored)):
            if self.scored.loc[i][
                "Lauren's Evaluation (1-3, where 1 = Bad, 2 = Good enough but edits needed, 3 = Good and usable as is)"] == 3.0:
                prompt = self.scored.loc[i]['prompt']
                choice = self.scored.loc[i]["Lauren's Pick (1-4)"]
                model = self.num_to_model[choice]
                completion = self.scored.loc[i][model]
                completion = completion.replace('[END]', '')
                completion = completion.replace('END', '')
                completion = completion.replace('\n', '')
                completion = completion.replace('#', '')
                completion = completion.replace('[', '')
                completion = completion.replace(']', '')
                completion = completion + ' END'
                self.main_data.loc[len(self.main_data.index)] = [prompt, completion]
        self.main_data.to_csv('data_stop.csv')

    def populate_batch5(self):
        for file in os.scandir(self.test_path):
            df = pd.read_excel(file.path)
            if len(list(df.columns)) != 2:
                df.drop(df.columns[0], axis=1, inplace=True)
                df = df[[df.columns[0], df.columns[1]]]

            if list(df.columns) != ['prompt', 'generated_commentary']:
                df = df.shift(1)
                df.loc[0] = df.columns
                df.columns = ['prompt', 'generated_commentary']

            tmp = self.preprocess_dataframe_test(df)

            for i in list(tmp.index):
                j = tmp.loc[i, 'prompt']
                gc = openai.Completion.create(
                    model=self.model,
                    prompt=j,
                    max_tokens=16,
                    temperature=0.10,
                    stop=[" END"]
                )
                tmp.loc[i, 'generated_commentary'] = gc.choices[0].text
            df['new'] = [np.nan] * len(df)
            for i in tmp.index:
                df['new'][i + 1] = tmp['generated_commentary'][i]


train = CompileTrainData()
train.populate_batch5()