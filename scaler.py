import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
df = pd.read_csv("kidney.csv")

df[['pcv', 'wc', 'rc', 'dm', 'cad', 'classification']] = df[['pcv', 'wc', 'rc', 'dm', 'cad', 'classification']].replace(to_replace={'\t8400':'8400', '\t6200':'6200', '\t43':'43', '\t?':np.nan, '\tyes':'yes', '\tno':'no', 'ckd\t':'ckd'})

df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

df[['pcv', 'wc', 'rc']] = df[['pcv', 'wc', 'rc']].astype('float64')

df.drop(['id', 'sg', 'pcv', 'pot'],axis=1,inplace=True)

col = ['rbc', 'pcc', 'pc', 'ba', 'htn', 'dm', 'cad', 'pe', 'ane']
encoder = LabelEncoder()
for col in col:
    df[col] = encoder.fit_transform(df[col])

df[['appet', 'classification']] = df[['appet', 'classification']].replace(to_replace={'good':'1', 'ckd':'1', 'notckd':'0', 'poor':'0'})
df[['classification', 'appet']] = df[['classification', 'appet']].astype('int64')
    
X = df.drop("classification", axis=1)
y = df["classification"]

scaler = StandardScaler()
features = scaler.fit_transform(X)

import pickle
pickle.dump(scaler, open('scaler.pkl','wb'))
scaler = pickle.load(open('scaler.pkl','rb'))
