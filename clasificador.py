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


e =None
f=None

def clasificar_tweets():
    df = pd.read_csv(f)
    if e is not None:
        df = df[(df['airline'] == e)]

    df['text'] = df['text'].apply(preprocesar_texto)
    df['tweet_coord']=df['tweet_coord'].apply(limpiar_coordenadas)
    imputar_valores(df)
    df = df[['airline_sentiment', 'airline_sentiment_confidence', 'text','negativereason_confidence']]
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

def obtener_coordenadas(ciudad):
    geolocator = Nominatim(user_agent="my_app")
    try:
        location = geolocator.geocode(ciudad, country_codes='US')
        if location is not None:
            return location.latitude, location.longitude
        else:
            return None
    except:
        return None

def imputar_valores(df):
    total = len(df)
    with tqdm(total=total) as pbar:
        for index, row in df.iterrows():
            tweet_coord = row['tweet_coord']
            # si no hay coordenadas en la fila actual, intentar recuperarlas de la columna tweet_location
           # print("Coordenada:"+str(tweet_coord))
            if pd.isna(tweet_coord):
                tweet_loc = df.at[index, 'tweet_location']
                #print("Location:"+str(tweet_loc))
                if not pd.isna(tweet_loc):
                    coords = obtener_coordenadas(tweet_loc)
                    print("Coordenadas:"+str(coords))
                    if coords is not None:
                        #print("SE HA CONSEGUIDO EL VALOR DE LAS COORDENADAS GRACIAS A LA LOCATION")
                        df.at[index, 'tweet_coord'] = str(coords)
                        if pd.notna(df.at[index, 'tweet_coord']):
                            print("Valor anterior:"+str(tweet_coord)+", Valor posterior:"+str(df.at[index, 'tweet_coord']))
                else:
                    # si no hay tweet_location, mirar user_timezone
                    user_tz = df.at[index, 'user_timezone']
                    if not pd.isna(user_tz):
                        #print("SE HA CONSEGUIDO EL VALOR DE LAS COORDENADAS GRACIAS A EL USER TIME ZONE")
                        # asignar la ciudad correspondiente al timezone
                        if user_tz == 'Eastern Time (US & Canada)':
                            df.at[index, 'tweet_coord'] = str(obtener_coordenadas('New York City, New York'))
                            if pd.notna(df.at[index, 'tweet_coord']):
                                print("Valor anterior:"+str(tweet_coord)+", Valor posterior:"+str(df.at[index, 'tweet_coord']))
                        elif user_tz == 'Central Time (US & Canada)':
                            df.at[index, 'tweet_coord'] = str(obtener_coordenadas('Austin, Texas'))
                            if pd.notna(df.at[index, 'tweet_coord']):
                                print("Valor anterior:"+str(tweet_coord)+", Valor posterior:"+str(df.at[index, 'tweet_coord']))
                        elif user_tz == 'Mountain Time (US & Canada)':
                            df.at[index, 'tweet_coord'] = str(obtener_coordenadas('Denver, Colorado'))
                            if pd.notna(df.at[index, 'tweet_coord']):
                                print("Valor anterior:"+str(tweet_coord)+", Valor posterior:"+str(df.at[index, 'tweet_coord']))
                        elif user_tz == 'Pacific Time (US & Canada)':
                            df.at[index, 'tweet_coord'] = str(obtener_coordenadas('San Francisco, California'))
                            if pd.notna(df.at[index, 'tweet_coord']):
                                print("Valor anterior:"+str(tweet_coord)+", Valor posterior:"+str(df.at[index, 'tweet_coord']))
                        else:
                            # en caso de no encontrar una ciudad correspondiente en el timezone, buscar la sede de la aerolínea
                            #print("SE HA CONSEGUIDO EL VALOR DE LAS COORDENADAS GRACIAS A LA SEDE")
                            airline = df.at[index,'airline']
                            if airline == 'United':
                                df.at[index, 'tweet_coord'] = str(obtener_coordenadas('Chicago, Illinois'))
                                if pd.notna(df.at[index, 'tweet_coord']):
                                    print("Valor anterior:"+str(tweet_coord)+", Valor posterior:"+str(df.at[index, 'tweet_coord']))
                            elif airline == 'Southwest':
                                df.at[index, 'tweet_coord'] = str(obtener_coordenadas('Dallas, Texas'))
                                if pd.notna(df.at[index, 'tweet_coord']):
                                    print("Valor anterior:"+str(tweet_coord)+", Valor posterior:"+str(df.at[index, 'tweet_coord']))
                            elif airline == 'Delta':
                                df.at[index, 'tweet_coord'] = str(obtener_coordenadas('Atlanta, Georgia'))
                                if pd.notna(df.at[index, 'tweet_coord']):
                                    print("Valor anterior:"+str(tweet_coord)+",Valor posterior"+str(df.at[index, 'tweet_coord']))
                    else:
                        print("NO SE HA PODIDO IMPUTAR NINGÚN VALOR")
                        df.drop(index, inplace=True)
            pbar.update(1)
    df.to_csv("limpieza_coordenadas.csv")
        
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
