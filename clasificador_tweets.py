import getopt
import string
import sys
from collections import Counter
import numpy as np
import pandas as pd
import sklearn as sk
from imblearn.under_sampling import RandomUnderSampler
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import geopy
from geopy.geocoders import Nominatim # pip3 install geopy
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import emoji

nltk.download('punkt')
nltk.download('wordnet')
e =None
f=None
p = './'

def clasificar_tweets():
    df = pd.read_csv(f)
    if e is not None:
        df = df[(df['airline'] == e)]
    df['text'] = tqdm(df['text'].apply(preprocess_text), total=len(df['text']))
    X_train, X_test, y_train, y_test = train_test_split(df['text'].values,df['airline_sentiment'].values,test_size=0.2, random_state=42)
    # Bag of Words
    vectorizer_bow = CountVectorizer()
    X_train_bow = vectorizer_bow.fit_transform(X_train)
    X_test_bow = vectorizer_bow.transform(X_test)
    # TF-IDF
    vectorizer_tfidf = TfidfVectorizer()
    X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
    X_test_tfidf = vectorizer_tfidf.transform(X_test)
    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    from sklearn.naive_bayes import MultinomialNB
    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    best_f1_score_nb_bow = 0
    best_f1_score_nb_tfidf = 0
    mejor_resultado_nb_bow=None
    mejor_resultado_nb_tfidf=None
    for alpha_val in alpha_values:
        clf_nb_bow = MultinomialNB(alpha=alpha_val)
        clf_nb_bow.class_weight = "balanced"
        clf_nb_bow.fit(X_train_bow, y_train)
        predictions_nb_bow = clf_nb_bow.predict(X_test_bow)
        f1_nb_bow = f1_score(y_test, predictions_nb_bow, average='weighted')

        clf_nb_tfidf = MultinomialNB(alpha=alpha_val)
        clf_nb_tfidf.class_weight = "balanced"
        clf_nb_tfidf.fit(X_train_tfidf, y_train)
        predictions_nb_tfidf = clf_nb_tfidf.predict(X_test_tfidf)
        f1_nb_tfidf = f1_score(y_test, predictions_nb_tfidf, average='weighted')

        if f1_nb_bow > best_f1_score_nb_bow:
            best_f1_score_nb_bow = f1_nb_bow
            mejor_resultado_nb_bow=clf_nb_bow
            best_model_nb_bow = "Naive-Bayes with Bag of Words"
        if f1_nb_tfidf > best_f1_score_nb_tfidf:
            best_f1_score_nb_tfidf = f1_nb_tfidf
            mejor_resultado_nb_tfidf=clf_nb_tfidf
            best_model_nb_tfidf = "Naive-Bayes with TF-IDF"
    print("The best model for Naive-Bayes with Bag of Words is", best_model_nb_bow, "with an f1 score of", best_f1_score_nb_bow)
    print("The best model for Naive-Bayes with TF-IDF is", best_model_nb_tfidf, "with an f1 score of", best_f1_score_nb_tfidf)
    '''df['BOW_nb']=mejor_resultado_nb_bow.predict(X_test_bow)
    df['TFIDF_nb']=mejor_resultado_nb_tfidf.predict(X_test_tfidf)'''
    # crea un DataFrame para los datos de prueba y las predicciones de Naive-Bayes con Bag of Words y TFIDF
    test_data = pd.DataFrame({
        'text': X_test,'airline_sentiment':y_test,
        'BOW_nb': mejor_resultado_nb_bow.predict(X_test_bow),'TFIDF_nb':mejor_resultado_nb_tfidf.predict(X_test_tfidf)
    })
    test_data.to_csv("NAIVE_BAYES.CSV")
    mejor_resultado_dt_bow=None
    mejor_resultado_dt_tfidf=None
    from sklearn.tree import DecisionTreeClassifier
    splitter_values = ['best', 'random']
    max_f1_bow = 0
    # BOW
    for max_depth_val in range(d, D + 1):
        for min_samples_leaf_val in range(l, L + 1):
            for splitter_val in splitter_values:
                clf = DecisionTreeClassifier(random_state=1337,
                                            criterion='gini',
                                            splitter=splitter_val,
                                            max_depth=max_depth_val,
                                            min_samples_leaf=min_samples_leaf_val,
                                            )

                clf.class_weight = "balanced"
                clf.fit(X_train_bow, y_train)
                predictions = clf.predict(X_test_bow)
                f1 = f1_score(y_test, predictions, average='weighted')
                if f1 > max_f1_bow:
                    max_f1_bow = f1
                    mejor_resultado_dt_bow=clf


    # TF-IDF
    max_f1_tfidf = 0
    for max_depth_val in range(d, D + 1):
        for min_samples_leaf_val in range(l, L + 1):
            for splitter_val in splitter_values:
                clf = DecisionTreeClassifier(random_state=1337,
                                            criterion='gini',
                                            splitter=splitter_val,
                                            max_depth=max_depth_val,
                                            min_samples_leaf=min_samples_leaf_val,
                                            )

                clf.class_weight = "balanced"
                clf.fit(X_train_tfidf, y_train)
                predictions = clf.predict(X_test_tfidf)
                f1 = f1_score(y_test, predictions, average='weighted')
                if f1 > max_f1_tfidf:
                    max_f1_tfidf = f1
                    mejor_resultado_dt_tfidf=clf

    print("Max F1-score for BOW: ", max_f1_bow)
    print("Max F1-score for TF-IDF: ", max_f1_tfidf)
    test_data = pd.DataFrame({
        'text': X_test,'airline_sentiment':y_test,
        'BOW_dt': mejor_resultado_dt_bow.predict(X_test_bow),'TFIDF_dt':mejor_resultado_dt_tfidf.predict(X_test_tfidf)
    })
    test_data.to_csv("DECISSION_TREE.CSV")


def preprocess_text(text):
        text = text.lower()
        # Eliminar emojis
        #text = text.encode('ascii', 'ignore').decode('ascii')
        # Para convertir emoji --> texto
        text = emoji.demojize(text)
        # Eliminar nombres de usuario
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)
        # Eliminar signos de puntuación
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Eliminar enlaces
        text = re.sub(r'http\S+', '', text)
        # Tokenizar el texto
        tokens = nltk.word_tokenize(text)
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
    print('ARGV   :', sys.argv[1:])
    try:
        options, remainder = getopt.getopt(sys.argv[1:], 'a:o:k:K:d:D:l:L:p:f:e:h',
                                           ['algoritmo=', 'output=', 'k=', 'K=', 'd=', 'D=', 'leaf', 'path=', 'iFile','e=',
                                            'h'])
    except getopt.GetoptError as err:
        print('ERROR:', err)
        sys.exit(1)
    print('OPTIONS   :', options)

    for opt, arg in options:
        if opt in ('-o', '--output'):
            oFile = arg
        elif opt == '-a':
            a = arg
        elif opt == '-k':
            k = int(arg)
        elif opt == '-K':
            K = int(arg)
        elif opt == '-d':
            d = int(arg)
        elif opt == '-D':
            D = int(arg)
        elif opt == '-l':
            l = int(arg)
        elif opt == '-L':
            L = int(arg)
        elif opt in ('-p', '--path'):
            p = arg
        elif opt in ('-f', '--file'):
            f = arg
        elif opt =='-e':
            e = arg
        elif opt in ('-h', '--help'):
            print(
                ' -o outputFile \n -k numberOfItems \n -d distanceParameter \n -p inputFilePath \n -f inputFileName \n ')
            exit(1)

    if p == './':
        iFile = p + str(f)
    else:
        iFile = p + "/" + str(f)

clasificar_tweets()



