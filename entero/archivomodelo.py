from keras.models import load_model
#from keras.preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
import re
import string
from nltk.corpus import stopwords
import pickle
from pathlib import Path
THIS_FOLDER = Path(__file__).parent.resolve()
my_file = THIS_FOLDER / "modelo.h5"
second_file = THIS_FOLDER / "tokenizador.pickle"

# Cargar el modelo entrenado
modelo_cargado = load_model(my_file)

#Cargar tokenizador
with open(second_file, 'rb') as archivo:
    tokenizador = pickle.load(archivo)


# Mapeo de etiquetas
label_mapping = {1: "contradiction", 2: "entailment"}

def limpiar_texto(texto):
  texto = str(texto).lower()
  texto = re.sub('\[.*?\]', '', texto)
  texto = re.sub('<.*?>+', '', texto)
  texto = re.sub('\n', '', texto)
  texto = re.sub('[%s]' % re.escape(string.punctuation), '', texto)
  texto = re.sub('\w*\d\w*', '', texto)
  return texto

stop_words = stopwords.words('english')
more_stopwords = ['u', 'im', 'c']
stop_words = stop_words + more_stopwords

def remove_stopwords(text):
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    return text

def realizar_prediccion(premisa, hipotesis):

    # Limpiar y preprocesar la premisa y la hipótesis
    premisa_cleaned = limpiar_texto(premisa)
    hipotesis_cleaned = limpiar_texto(hipotesis)
    
    # Remover stopwords de la premisa y la hipótesis
    premisa_cleaned = remove_stopwords(premisa_cleaned)
    hipotesis_cleaned = remove_stopwords(hipotesis_cleaned)

    # Tokenizar la premisa y la hipótesis
    premisa_tokenized = tokenizador.texts_to_sequences([premisa_cleaned])
    hipotesis_tokenized = tokenizador.texts_to_sequences([hipotesis_cleaned])

    # Realizar el padding de las secuencias tokenizadas
    premisa_padded = pad_sequences(premisa_tokenized, maxlen=45)
    hipotesis_padded = pad_sequences(hipotesis_tokenized, maxlen=45)

    # Realizar la predicción
    predictions = modelo_cargado.predict([premisa_padded, hipotesis_padded])

    # Decodificar las predicciones y obtener las etiquetas
    predicted_labels = np.argmax(predictions, axis=1)
    predicted_label = label_mapping[predicted_labels[0]]

    return predicted_label