import numpy as np
import pandas as pd
import os
import re

class Preprocess:
    def __init__(self, path):
        self.valid_paths = [ f.path for f in os.scandir(path) if f.is_dir() ]
        self.reject_names = ['9780062381828.xlsx', '9780545825030.xlsx', '9780547076690.xlsx', '9780142414538.xlsx']
        self.valid_paths_comm = []
        self.data = []
        self.main_data = pd.DataFrame()
    
    def generate_valid_paths(self):
        for dir in self.valid_paths:
            for f in os.scandir(dir):
                if f.is_dir() and 'commentary' in f.path:
                    self.valid_paths_comm.append(f.path)
        
    def generate_reject_names(self):
        for subdir in self.valid_paths_comm:
            for file in os.scandir(subdir):
                df = pd.read_excel(file.path, usecols= list(range(1, 3)))
                if len(df.columns) == 1:
                    self.reject_names.append(file.path[71:])
    
    def preprocess(self):    
        for subdir in self.valid_paths_comm:
            for file in os.scandir(subdir):
                df = pd.read_excel(file.path, usecols= list(range(1, 3)))
                
                if file.path[71:] in self.reject_names:
                    continue

                if list(df.columns) != ['Text', 'Commentary']:
                    df = df.shift(1)
                    df.loc[0] = df.columns
                    df.columns = ['Text', 'Commentary']

                df = df[~df['Commentary'].isin(['P', '-', 'Praise'])]
                df.dropna(inplace = True)
                
                self.data.append(df)

    def return_main_df(self):
        self.generate_valid_paths()
        self.generate_reject_names()
        self.preprocess()
        self.main_data = pd.concat(self.data)

a = Preprocess('/content/drive/MyDrive/Stanford Project Book Bundles/')
a.return_main_df()
print(a.main_data)
