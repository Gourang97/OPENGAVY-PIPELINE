from preprocessing import *
import pandas as pd
import numpy as np

df_all = pd.read_csv('G:/My Drive/DMP/Project/home-credit-default-risk/application_train.csv')

for col_name in df_all.columns:
    if df_all[col_name].dtypes == np.float64:
        df_all[col_name] = df_all[col_name].fillna(df_all[col_name].mean())
    
    if df_all[col_name].dtypes != np.float64 and df_all[col_name].dtypes != np.int64:
        df_all[col_name] = df_all[col_name].astype('category').cat.codes
        df_all[col_name] = df_all[col_name].fillna('0')

preprocessing = preprocessing()

data_binned, features_list_continuous, dict_thold = preprocessing.binning_data(
    df_all.drop('SK_ID_CURR', axis  = 1), 
    'TARGET', 
    10/len(df_all), 
    0, 
    10, 
    300
)
print (data_binned['AMT_CREDIT'])