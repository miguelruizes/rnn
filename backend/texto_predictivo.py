import numpy as np
import pandas as pd
import os
import random
import re
import pickle

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model # Sequential: poner capas una tras otra. load_model: cargar uno ya entrenado.
from tensorflow.keras.callbacks import EarlyStopping       # Para detener el entrenamiento si la IA deja de mejorar (ahorra tiempo).
from tensorflow.keras.layers import Embedding, LSTM, Dense # Tipos de neuronas: Embedding (traductor), LSTM (memoria), Dense (decisión final).
from tensorflow.keras.preprocessing.text import Tokenizer  # Convierte palabras en números (la IA solo entiende números).
from tensorflow.keras.preprocessing.sequence import pad_sequences # Rellena frases cortas para que todas midan lo mismo.
from tensorflow.keras.utils import to_categorical          # Convierte las respuestas en un formato de "probabilidades" (0s y 1s).

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

ARCHIVO_MODELO = 'modelos/texto_predictivo_modelo.keras'
ARCHIVO_TOKENIZADOR = 'modelos/texto_predictivo_tokenizador.pkl'
ARCHIVO_DATOS = 'csv/texto_predictivo_datos.csv'

