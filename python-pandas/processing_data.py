import pandas as pd
from pandas import set_option
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv("train.csv",sep =",")
df.fillna(df["Age"].mean(), inplace=True)
def Normalize_data():
    # numberic_df = df.select_dtypes(include=['number'])
    # array = numberic_df.values
    y = df['Survived'].values
    x = df.drop(columns=['Survived']).select_dtypes(include=['number']).values
    scaler = MinMaxScaler(feature_range=(0,1))
    rescaledX = scaler.fit_transform(x)
    np.set_printoptions(precision=3)
    print(rescaledX[0:10,:])


Normalize_data()
