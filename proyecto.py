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
import matplotlib.pyplot as plt
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

e =None
f=None

#1.grupo de tareas
def imputar_timezone_coord(df):
    geolocator = Nominatim(user_agent="xabi")  # Crear objeto geolocalizador
    # Imputar valores para tweet_cord
    for index, row in df.iterrows():
        if pd.isnull(row['tweet_coord']):
            # Si el valor está faltando, buscar las coordenadas de la ciudad a partir de user_timezone
            if row['user_timezone'] == 'Eastern Time (US & Canada)':
                location = geolocator.geocode("New York City, New York", exactly_one=True)
                if location is not None:
                    df.at[index, 'tweet_coord'] = [location.latitude, location.longitude]
            elif row['user_timezone'] == 'Central Time (US & Canada)':
                location = geolocator.geocode("Austin, Texas", exactly_one=True)
                if location is not None:
                    df.at[index, 'tweet_coord'] = [location.latitude, location.longitude]
            elif row['user_timezone'] == 'Mountain Time (US & Canada)':
                location = geolocator.geocode("Denver, Colorado", exactly_one=True)
                if location is not None:
                    df.at[index, 'tweet_coord'] = [location.latitude, location.longitude]
            elif row['user_timezone'] == 'Pacific Time (US & Canada)':
                location = geolocator.geocode("San Francisco, California", exactly_one=True)
                if location is not None:
                    df.at[index, 'tweet_coord'] = [location.latitude, location.longitude]
            else:
                # Si no se puede recuperar el valor a partir de user_timezone, imputar las coordenadas del
                # headquarters de la aerolínea correspondiente
                if row['airline'] == 'United':
                    location = geolocator.geocode("Chicago, Illinois", exactly_one=True)
                    if location is not None:
                        df.at[index, 'tweet_coord'] = [location.latitude, location.longitude]
                elif row['airline'] == 'Southwest':
                    location = geolocator.geocode("Dallas, Texas", exactly_one=True)
                    if location is not None:
                        df.at[index, 'tweet_coord'] = [location.latitude, location.longitude]
                elif row['airline'] == 'Delta':
                    location = geolocator.geocode("Atlanta, Georgia", exactly_one=True)
                    if location is not None:
                        df.at[index, 'tweet_coord'] = [location.latitude, location.longitude]
                elif row['airline'] == 'US Airways':
                    location = geolocator.geocode("Alexandria, Virginia", exactly_one=True)
                    if location is not None:
                        df.at[index, 'tweet_coord'] = [location.latitude, location.longitude]
                elif row['airline'] == 'Virgin America':
                    location = geolocator.geocode("Burlingame, California", exactly_one=True)
                    if location is not None:
                        df.at[index, 'tweet_coord'] = [location.latitude, location.longitude]

    # Imputar valores para user_timezone
    for index, row in df.iterrows():
        if pd.isnull(row['user_timezone']):
            # Si el valor está faltando, buscar el uso horario de la ciudad a partir de tweet_coord
            if not pd.isnull(row['tweet_coord']):
                if isinstance(row['tweet_coord'], list) and len(row['tweet_coord']) == 2:
                # Verificar que tweet_coord es un par de coordenadas
                    point = Point(row['tweet_coord'][0], row['tweet_coord'][1])
                    location = geolocator.reverse(row['tweet_coord'], exactly_one=True)
                    if location is not None:
                        df.at[index, 'user_timezone'] = location.raw['timezone']
    df.to_csv("coordenadas2.csv" ,index=False)
    return df

def limpiar():
    # leer el archivo csv
    df = pd.read_csv(f)
    # filtrar por aerolínea (si se especificó)
    if e is not None:
        df = df[(df['airline'] == e)]
    #Eliminar las @
    df['text'] = df['text'].str.replace('@[^\s]+', '', regex=True)
    # Convertir el texto a minúsculas
    df['text'] = df['text'].str.lower()
    # Eliminar signos de puntuación
    df['text'] = df['text'].str.replace('[^\w\s]', '')
    '''
    cant_original = len(df)
    # contar el número de valores vacíos en la columna "location"
    num_vacios = df["tweet_location"].isnull().sum()
    # imprimir el resultado
    print("Hay {} valores vacíos en la columna 'tweet_location' del dataframe.".format(num_vacios))
    # eliminar filas que no contienen fechas válidas
    num_vacios_fecha = df["tweet_created"].isnull().sum()
    # imprimir el resultado
    print("Hay {} valores vacíos en la columna 'tweet_created' del dataframe.".format(num_vacios_fecha))
    df = df.dropna(subset=["tweet_location"])
    cant_eliminados = cant_original - len(df)
    # imprimir la cantidad de filas eliminadas
    print("Se eliminaron {} filas con valores nulos en la columna 'tweet_location'".format(cant_eliminados))
    rows_to_drop = []
    patron_fecha = r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\s[-+]\d{4})|(\d{4}/\d{2}/\d{2}\s\d{2}:\d{2}:\d{2}\s[-+]\d{2}:\d{2})'
    for index, tweet in df.iterrows():
        fecha = tweet['tweet_created']
        if  not re.match(patron_fecha, fecha):
            rows_to_drop.append(index)
    # eliminar las filas que no tienen fechas válidas
    df.drop(rows_to_drop, inplace=True)
    # mostrar las filas eliminadas
    print("Filas a eliminar por fechas inválidas:", rows_to_drop)
    print("Se han eliminado {} filas que contenían fechas inválidas".format(len(rows_to_drop)))
    # guardar el dataframe limpio en un archivo csv
    df.to_csv('tweets_limpios.csv', index=False)'''
    return df

def complete_tweet_coords(df):
    # leer el archivo csv que contiene los tweets
    # filtrar los tweets que pertenecen a la aerolínea especificada
    # crear un objeto geolocalizador
    geolocator = Nominatim(user_agent="xabi")
    # eliminar las filas que tengan una ubicación vacía
    df.dropna(subset=['tweet_location'], inplace=True)
    # crear una lista para almacenar las filas a eliminar
    rows_to_drop = []
    # recorrer los tweets y completar las coordenadas faltantes
    for index, tweet in df.iterrows():
        if pd.isna(tweet['tweet_coord']):
            location = tweet['tweet_location']
            location = location.replace('[^\w\s]', '')
            # Convertir todo el texto a minúsculas
            location = location.lower()
            if isinstance(location, str):
                # extraer el nombre de la ciudad de la ubicación del tweet
                city = location.split(',')[-1].strip()
                try:
                    # obtener las coordenadas de la ciudad utilizando Geopy
                    location = geolocator.geocode(city,country_codes='US')
                    if location is not None:
                        coords = (location.latitude, location.longitude)
                        print(city+" coordenadas:   "+str(coords))
                        df.at[index, 'tweet_coord'] = str(coords)
                    else:
                        # si la ciudad no es válida, agregar la fila a la lista de filas a eliminar
                        print("CIUDAD NO VALIDA!"+str(city))
                        rows_to_drop.append(index)
                except:
                    rows_to_drop.append(index)
    # eliminar las filas que no corresponden a ciudades válidas
    df.drop(rows_to_drop, inplace=True)
    # guardar los resultados en un nuevo archivo csv
    print("Filas a eliminar:", rows_to_drop)
    df.to_csv("coordenadas.csv" ,index=False)
    return df

def clasificador(df): 
    df = df[['airline_sentiment', 'airline_sentiment_confidence', 'text','tweet_created']]
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(df[['text', 'airline_sentiment_confidence']], df['airline_sentiment'], test_size=0.3, random_state=42)
    # Vectorizar los datos de texto utilizando CountVectorizerPero c
    vectorizer = CountVectorizer(stop_words='english')
    X_train_vect = vectorizer.fit_transform(X_train['text'])
    X_test_vect = vectorizer.transform(X_test['text'])
    # Entrenar el modelo Naive Bayes
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_vect, y_train)
    '''
    # Entrenar el modelo de árbol de decisión
    dt_classifier = DecisionTreeClassifier(random_state = 1337,
                     criterion = 'gini',
                     splitter = 'best',
                     max_depth = 5,
                     min_samples_leaf = 1)
    
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

    # Calcular el f1-score, precisión y recall para el árbol de decisión
    dt_f1 = f1_score(y_test, dt_test_pred, average='weighted')
    dt_precision = precision_score(y_test, dt_test_pred, average='weighted')
    dt_recall = recall_score(y_test, dt_test_pred, average='weighted')

    # Imprimir los resultados
    print("Resultados para Naive Bayes:")
    print("F1-score:", nb_f1)
    print("Precisión:", nb_precision)
    print("Recall:", nb_recall)

    print("Resultados para el árbol de decisión:")
    print("F1-score:", dt_f1)
    print("Precisión:", dt_precision)
    print("Recall:", dt_recall)
    return df'''
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

def peores_valoraciones():
    df = pd.read_csv('clasificacion2.csv')
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
    df = pd.read_csv('clasificacion2.csv')
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].str.replace('[^\w\s]', '')
    #stop_words = set(stopwords.words('english'))
    vectorizer = CountVectorizer(stop_words='english')
    stop_words = vectorizer.fit_transform(df['text'])
    #df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    # Word cloud de palabras más frecuentes en críticas negativas y positivas
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


df=limpiar()
#df2=complete_tweet_coords(df)
#df3=imputar_timezone_coord(df2)
clasificador(df)
patrones_sentimientos()
peores_valoraciones()


