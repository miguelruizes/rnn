import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend suitable for scripts

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from io import BytesIO
import os
import random

import tensoflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

def obtener_datos_empresas(empresa, periodo='2y'):
    df = yf.download(empresa, period=periodo, progress=False, auto_adjust=True)
    
    if df.empty:
        raise ValueError(f"No se encontraron datos para la empresa: {empresa}")
    
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            serie = df.xs('Close', axis=1, level=0)
            if serie.empty:
                serie = df.xs(empresa, axis=1, level=1)['Close']
            return serie
        except:
            return df.iloc[:, 0]
        
    if 'Close' in df.columns:
        return df['Close']
    return df.iloc[:, 0]


    
