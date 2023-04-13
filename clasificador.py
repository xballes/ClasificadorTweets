import getopt
import sys
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import  accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
import string
import re
import geopy
from geopy.geocoders import Nominatim # pip3 install geopy
from wordcloud import WordCloud # pip3 install wordclloud
import re
from geopy.point import Point
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
nltk.download('punkt')
from tqdm import tqdm

import matplotlib.pyplot as plt


e =None
f=None
coord_cache = {}
timezone_dict = {'Eastern Time (US & Canada)': 'New York City, New York','Central Time (US & Canada)': 'Austin, Texas','Mountain Time (US & Canada)': 'Denver, Colorado','Pacific Time (US & Canada)': 'San Francisco, California'}
airline_dict = {'United': 'Chicago, Illinois','Southwest': 'Dallas, Texas','Delta': 'Atlanta, Georgia','US Airways': 'Alexandria, Virginia','Virgin America': 'Burlingame, California'}

def clasificar_tweets():
    df = pd.read_csv(f)
    if e is not None:
        df = df[(df['airline'] == e)]
    df['text'] = tqdm(df['text'].apply(preprocesar_texto), total=len(df['text']))
    df['negativereason_confidence']= df['negativereason_confidence'].fillna(value=0)
    df['tweet_coord'] = tqdm(df['tweet_coord'].apply(limpiar_coordenadas), total=len(df['tweet_coord']))
    df['tweet_location'] = df.apply(limpiar_tweet_location, axis=1)
    df.to_csv("limpieza_principal.csv")
    df=imputar_valores_coordenadas(df)
    #df = df[['airline_sentiment', 'airline_sentiment_confidence', 'text','negativereason_confidence']]
    negative = df[df['airline_sentiment'] == 'negative']
    neutral = df[df['airline_sentiment'] == 'neutral']
    positive = df[df['airline_sentiment'] == 'positive']

    # Submuestrear la clase mayoritaria (negative) para igualar la cantidad de muestras de todas las clases
    negative_downsampled = resample(negative, replace=False, n_samples=len(positive), random_state=42)

    # Combinar las muestras submuestreadas de la clase mayoritaria con las muestras de las otras clases
    data_balanced = pd.concat([negative_downsampled, neutral, positive])

    # Separar los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(data_balanced['text'], data_balanced['airline_sentiment'], test_size=0.2, random_state=42)

    # Vectorizar los datos de texto utilizando CountVectorizer
    vectorizer = CountVectorizer(stop_words='english')
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)
    # Entrenar el modelo Naive Bayes
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_vect, y_train)
    best_dt_f1 = 0
    best_dt_min_samples_leaf = None
    best_dt_max_depth = None
    for min_samples_leaf in range(1, 3):
        for max_depth in range(3, 6):
            dt_classifier = DecisionTreeClassifier(random_state = 1337,
                         criterion = 'gini',
                         splitter = 'best',
                         max_depth = max_depth,
                         min_samples_leaf = min_samples_leaf)
            
            #dt_classifier.class_weight = "balanced"

            dt_classifier.fit(X_train_vect, y_train)
            
            dt_test_pred = dt_classifier.predict(X_test_vect)

            dt_f1 = f1_score(y_test, dt_test_pred, average='weighted')
            
            if dt_f1 > best_dt_f1:
                best_dt_f1 = dt_f1
                best_dt_min_samples_leaf = min_samples_leaf
                best_dt_max_depth = max_depth
                
    dt_classifier = DecisionTreeClassifier(random_state = 1337,
                 criterion = 'gini',
                 splitter = 'best',
                 max_depth = best_dt_max_depth,
                 min_samples_leaf = best_dt_min_samples_leaf)
    
    dt_classifier.fit(X_train_vect, y_train)

    # Predecir los sentimientos para todo el conjunto de datos
    df_vect = vectorizer.transform(df['text'])
    nb_pred = nb_classifier.predict(df_vect)
    dt_pred = dt_classifier.predict(df_vect)
    # Agregar las columnas de sentimiento al DataFrame
    df['sentimiento_nb'] = nb_pred
    df['sentimiento_dt'] = dt_pred
    #df_aux['sentimiento_nb'] = nb_pred
    #df_aux['sentimiento_dt'] = dt_pred
    df.to_csv('clasificacion.csv', index=False)
    # Hacer predicciones para los datos de prueba
    nb_test_pred = nb_classifier.predict(X_test_vect)
    dt_test_pred = dt_classifier.predict(X_test_vect)

    # Calcular el f1-score, precisión y recall para Naive Bayes
    nb_f1 = f1_score(y_test, nb_test_pred, average='weighted')
    nb_precision = precision_score(y_test, nb_test_pred, average='weighted')
    nb_recall = recall_score(y_test, nb_test_pred, average='weighted')

    # Calcular el f1-score, precisión y recall para Árbol de Decisión
    dt_f1 = f1_score(y_test, dt_test_pred, average='weighted')
    dt_precision = precision_score(y_test, dt_test_pred, average='weighted')
    dt_recall = recall_score(y_test, dt_test_pred, average='weighted')

    # Imprimir los resultados
    print("Resultados Naive Bayes:")
    print("F1-Score:", nb_f1)
    print("Precisión:", nb_precision)
    print("Recall:", nb_recall)
    print()
    print("Resultados Árbol de Decisión:")
    print("Min_samples_leaf del árbol de decisión:", best_dt_min_samples_leaf)
    print("Max_depth del árbol de decisión:", best_dt_max_depth)
    print("F1-Score:", dt_f1)
    print("Precisión:", dt_precision)
    print("Recall:", dt_recall)
    return df

def preprocesar_texto(texto):
     # Convertir a minúsculas
    texto = texto.lower()
    # Eliminar emojis
    texto = texto.encode('ascii', 'ignore').decode('ascii')
    
    # Eliminar nombres de usuario
    texto = re.sub(r'@[A-Za-z0-9_]+', '', texto)
    
    # Eliminar signos de puntuación
    texto = texto.translate(str.maketrans("", "", string.punctuation))
    #Eliminar enlaces
    texto = re.sub(r'http\S+', '', texto)
    # Tokenizar el texto
    tokens = nltk.word_tokenize(texto)
    
    # Eliminar stop-words
    stop_words = set(stopwords.words('english'))
    tokens_sin_stopwords = [palabra for palabra in tokens if palabra not in stop_words]
    
    # Lematizar el texto
    lematizador = WordNetLemmatizer()
    tokens_lematizados = [lematizador.lemmatize(palabra) for palabra in tokens_sin_stopwords]
    
    # Unir los tokens lematizados en un texto preprocesado
    texto_preprocesado = " ".join(tokens_lematizados)
    return texto_preprocesado

def limpiar_coordenadas(tweet_coord):
    if pd.isna(tweet_coord) or tweet_coord.strip() == '' or  str(tweet_coord) =="[0.0, 0.0]":
        # Empty or missing values
        return None
    elif re.match(r"\[(-?\d+(\.\d+)?),\s*(-?\d+(\.\d+)?)\]", str(tweet_coord)):
        # Tweet coord already in correct format
        return tweet_coord
    else:
        # Tweet coord not in correct format, so clear cell
        return None

def limpiar_tweet_location(row):
    texto = row['tweet_location']
    if pd.isna(texto):
        user_tz = row['user_timezone']
        airline = row['airline']
        return timezone_dict.get(user_tz) or airline_dict.get(airline)

    texto = texto.encode('ascii', 'ignore').decode('ascii')
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\([^)]*\)', '', texto)
    texto = re.sub(r'\W+', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)
    if texto in coord_cache:
        coordenadas = coord_cache[texto]
    else:
        coordenadas = obtener_coordenadas(texto)
        coord_cache[texto] = coordenadas
    if coordenadas is None:
        user_tz = row['user_timezone']
        airline = row['airline']
        return timezone_dict.get(user_tz) or airline_dict.get(airline)
    else:
        return texto

def obtener_coordenadas(ciudad):
    geolocator = Nominatim(user_agent="xabi")
    if ciudad in coord_cache:
        return coord_cache[ciudad]
    try:
        location = geolocator.geocode(ciudad, country_codes=['US','CA'])
        if location is not None:
            coords = (location.latitude, location.longitude)
            coord_cache[ciudad] = coords
            return coords
        else:
            return None
    except:
        return None
    
def imputar_valores_coordenadas(df):
    total = len(df)
    with tqdm(total=total) as pbar:
        for index, row in df.iterrows():
            tweet_coord = row['tweet_coord']
            # si no hay coordenadas en la fila actual, intentar recuperarlas de la columna tweet_location
            if pd.isna(tweet_coord):
                tweet_loc = df.at[index, 'tweet_location']
                if not pd.isna(tweet_loc): # si hay location...
                    coords = obtener_coordenadas(tweet_loc)
                    if coords is not None:
                        old_val = df.at[index, 'tweet_coord']
                        new_val = str(coords)
                        df.at[index, 'tweet_coord'] = new_val
                        print(f"Valor modificado en fila {index}: '{old_val}' -> '{new_val}'")
                if pd.isna(df.at[index, 'tweet_coord']):
                    # si no se han podido recuperar las coordenadas de la tweet_location o no son correctas, mirar user_timezone
                    user_tz = df.at[index, 'user_timezone']
                    if not pd.isna(user_tz) and user_tz in timezone_dict:
                            coords = obtener_coordenadas(timezone_dict[user_tz])
                            if coords is not None:
                                old_val = df.at[index, 'tweet_coord']
                                new_val = str(coords)
                                df.at[index, 'tweet_coord'] = new_val
                                print(f"Valor modificado en fila {index}: '{old_val}' -> '{new_val}'")
                    if pd.isna(df.at[index, 'tweet_coord']):
                            # en caso de no encontrar una ciudad correspondiente en el timezone, buscar la sede de la aerolínea
                            airline = df.at[index,'airline']
                            if airline in airline_dict:
                                coords = obtener_coordenadas(airline_dict[airline])
                                if coords is not None:
                                    old_val = df.at[index, 'tweet_coord']
                                    new_val = str(coords)
                                    df.at[index, 'tweet_coord'] = new_val
                                    print(f"Valor modificado en fila {index}: '{old_val}' -> '{new_val}'")
                if pd.isna(df.at[index, 'tweet_coord']):
                    df.drop(index, inplace=True)
                    print(f"Fila {index} eliminada:")
            pbar.update(1)
    df = df.dropna(subset=['tweet_coord'])
    df.to_csv("limpieza_coordenadas.csv")
    return df

def peores_valoraciones():
    df = pd.read_csv('clasificacion.csv')
    df['tweet_created'] = pd.to_datetime(df['tweet_created'])
    df['fecha'] = df['tweet_created'].dt.date
    df['hora'] = df['tweet_created'].dt.hour
    df['sentimiento_nb'] = df['sentimiento_nb'].replace({'negative': 0, 'neutral': 1, 'positive': 2})
    df['sentimiento_dt'] = df['sentimiento_dt'].replace({'negative': 0, 'neutral': 1, 'positive': 2})
    fecha_peor_sentimiento_nb = df.groupby('fecha')['sentimiento_nb'].mean().sort_values().index[0]
    fecha_peor_sentimiento_dt = df.groupby('fecha')['sentimiento_dt'].mean().sort_values().index[0]
    hora_peor_sentimiento_nb = df.groupby('hora')['sentimiento_nb'].mean().sort_values().index[0]
    hora_peor_sentimiento_dt = df.groupby('hora')['sentimiento_dt'].mean().sort_values().index[0]
    print("La peor fecha según el clasificador Naive Bayes fue:", fecha_peor_sentimiento_nb)
    print("La peor fecha según el clasificador Decision Tree fue:", fecha_peor_sentimiento_dt)
    print("La peor hora según el clasificador Naive Bayes fue:", hora_peor_sentimiento_nb)
    print("La peor hora según el clasificador Decision Tree fue:", hora_peor_sentimiento_dt)

    # Gráfico de sentimientos por hora del día
    df.groupby('hora')[['sentimiento_nb', 'sentimiento_dt']].mean().plot(kind='bar')
    plt.title('Sentimientos por hora del día')
    plt.xlabel('Hora del día')
    plt.ylabel('Sentimiento medio')
    plt.savefig('sentimientos_por_hora.png')  # Exportar gráfico
    plt.close()

def patrones_sentimientos():
    df = pd.read_csv('clasificacion.csv')
    neg_words = ' '.join(list(df[df['sentimiento_nb']=='negative']['text']))
    pos_words = ' '.join(list(df[df['sentimiento_nb']=='positive']['text']))
    wc_neg = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(neg_words)
    wc_pos = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(pos_words)

    # Gráfico de palabras más frecuentes en críticas negativas
    plt.figure(figsize=(12,6))
    plt.imshow(wc_neg, interpolation='bilinear')
    plt.axis('off')
    plt.title('Palabras más frecuentes en críticas negativas')
    plt.savefig('neg_words.png')
    plt.close()

    # Gráfico de palabras más frecuentes en críticas positivas
    plt.figure(figsize=(12,6))
    plt.imshow(wc_pos, interpolation='bilinear')
    plt.axis('off')
    plt.title('Palabras más frecuentes en críticas positivas')
    plt.savefig('pos_words.png')
    plt.close()

    # Análisis de las palabras más comunes en críticas negativas y positivas
    neg_tokens = word_tokenize(neg_words)
    pos_tokens = word_tokenize(pos_words)

    neg_freq = FreqDist(neg_tokens)
    pos_freq = FreqDist(pos_tokens)

    print('Palabras más frecuentes en críticas negativas:')
    print(neg_freq.most_common(10))
    print('Palabras más frecuentes en críticas positivas:')
    print(pos_freq.most_common(10))

    # Gráfico de las palabras más comunes en críticas negativas y positivas
    plt.figure(figsize=(12,6))
    neg_freq.plot(30, title='Palabras más comunes en críticas negativas')
    plt.savefig('neg_freq.png')
    plt.close()

    plt.figure(figsize=(12,6))
    pos_freq.plot(30, title='Palabras más comunes en críticas positivas')
    plt.savefig('pos_freq.png')
    plt.close()
   



if __name__ == '__main__':
    # test,train,testX,testY,trainX,trainY,target_map=proceso()
    # crear_modelo(test,train,testX,testY,trainX,trainY,target_map)
    # print("Se ha terminado la ejecucion")
    # print("VALORES MAXIMOS:"+"k:"+max_k+"d:"+max_d+"weight:"+max_Weight)
    print('ARGV   :', sys.argv[1:])
    try:
        options, remainder = getopt.getopt(sys.argv[1:], 'o:f:e:h', [
                                           'output=', 'f=','e=','help'])
    except getopt.GetoptError as err:
        print('ERROR:', err)
        sys.exit(1)
    print('OPTIONS   :', options)

    for opt, arg in options:
        if opt in ('-o', '--output'):
            oFile = arg
        elif opt == '-e':
            e = arg
        elif  opt =='-f':
            f=arg
        elif opt in ('-h', '--help'):
           print("-f para seleccionar el archivo .csv")
           print("-e para seleccionar la empresa.")
           exit(1)

clasificar_tweets()
patrones_sentimientos()
peores_valoraciones()
