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


e =None
f=None

def clasificar_tweets():
    df = pd.read_csv(f)
    if e is not None:
        df = df[(df['airline'] == e)]

    df['text'] = df['text'].apply(preprocesar_texto)

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
prueba_df= df = pd.read_csv('delta.csv')
