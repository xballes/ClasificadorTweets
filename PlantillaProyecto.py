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
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.naive_bayes import MultinomialNB
import timezonefinder
import pytz
import time


nltk.download('punkt')
nltk.download('wordnet')

p = './'
oFile = "output.out"
e=None

# Press the green button in the gutter to run the script.
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
        elif opt == '-e':
            e= arg
        elif opt in ('-h', '--help'):
            print(
                ' -o outputFile \n -k numberOfItems \n -d distanceParameter \n -p inputFilePath \n -f inputFileName \n ')
            exit(1)

    if p == './':
        iFile = p + str(f)
    else:
        iFile = p + "/" + str(f)
    # astype('unicode') does not work as expected
    ml_dataset = pd.read_csv(iFile)
    if e is not None:
        ml_dataset = ml_dataset[(ml_dataset['airline'] == e)]

    coord_cache = {}

    timezones = {
        'Eastern Time (US & Canada)': 'New York City, New York',
        'Central Time (US & Canada)': 'Austin, Texas',
        'Mountain Time (US & Canada)': 'Denver, Colorado',
        'Pacific Time (US & Canada)': 'San Francisco, California',
        'Atlantic Time (Canada)':'Nova Scotia',
        # Resto de zonas horarias
        'Alaska': 'Anchorage, Alaska',
        'Hawaii': 'Honolulu, Hawaii',
        'Arizona': 'Phoenix, Arizona',
        'Indiana (East)': 'Indianapolis, Indiana',
        'Midway Island': 'Midway Atoll, Hawaii',
        'Guam': 'Hagatna, Guam',
        'Samoa': 'Pago Pago, American Samoa'
    }

    airlines = {'United': 'Chicago, Illinois', 'Southwest': 'Dallas, Texas', 'Delta': 'Atlanta, Georgia',
                'US Airways': 'Alexandria, Virginia', 'Virgin America': 'Burlingame, California'}

    # Abrir el fichero .csv y cargarlo en un dataframe de pandas

    pd.set_option('display.width', 3000)
    pd.set_option('display.max_rows', 200)
    pd.set_option('display.max_columns', 200)


    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()


    def imputar_valores_coordenadas(df):
        total = len(df)
        with tqdm(total=total) as pbar:
            for index, row in df.iterrows():
                tweet_coord = row['tweet_coord']
                # si no hay coordenadas en la fila actual, intentar recuperarlas de la columna tweet_location
                if pd.isna(tweet_coord):
                    tweet_loc = df.at[index, 'tweet_location']
                    if not pd.isna(tweet_loc):  # si hay location...
                        coords = obtener_coordenadas(tweet_loc)
                        if coords is not None:
                            old_val = df.at[index, 'tweet_coord']
                            new_val = str(coords)
                            df.at[index, 'tweet_coord'] = new_val
                            print(f"Valor modificado en fila {index}: '{old_val}' -> '{new_val}'")
                    if pd.isna(df.at[index, 'tweet_coord']):
                        # si no se han podido recuperar las coordenadas de la tweet_location o no son correctas, mirar user_timezone
                        user_tz = df.at[index, 'user_timezone']
                        if not pd.isna(user_tz) and user_tz in timezones:
                            coords = obtener_coordenadas(timezones[user_tz])
                            if coords is not None:
                                old_val = df.at[index, 'tweet_coord']
                                new_val = str(coords)
                                df.at[index, 'tweet_coord'] = new_val
                                print(f"Valor modificado en fila {index}: '{old_val}' -> '{new_val}'")
                        if pd.isna(df.at[index, 'tweet_coord']):
                            # en caso de no encontrar una ciudad correspondiente en el timezone, buscar la sede de la aerolínea
                            airline = df.at[index, 'airline']
                            if airline in airlines:
                                coords = obtener_coordenadas(airlines[airline])
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
        return df


    def limpiar_coordenadas(tweet_coord):
        if pd.isna(tweet_coord) or tweet_coord.strip() == '' or str(tweet_coord) == "[0.0, 0.0]":
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
        if pd.isna(texto) or texto=="nan":
            user_tz = row['user_timezone']
            airline = row['airline']
            return timezones.get(user_tz) or airlines.get(airline)

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
            return timezones.get(user_tz) or airlines.get(airline)
        else:
            return texto
        
    def limpiar_timezone(row):
        texto = row['user_timezone']
        us_timezones = pytz.country_timezones['US']
        canada_timezones = pytz.country_timezones['CA']
        if not texto or (texto not in us_timezones and texto not in canada_timezones and texto not in timezones):
            ciudad = row['tweet_location']
            tf = timezonefinder.TimezoneFinder()
            coordenadas=row['tweet_coord']
            coordenadas = re.sub(r'[()\[\]\s]+', '', coordenadas)
            lat, lng = coordenadas.split(',')
            timezone_str = tf.timezone_at(lng=float(lng), lat=float(lat))
            return timezone_str
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


    def preprocess_text(text):
        text = text.lower()
        # Eliminar emojis
        text = text.encode('ascii', 'ignore').decode('ascii')

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

        # Aplicar stemming
        stemmer = PorterStemmer()
        tokens_stemmed = [stemmer.stem(palabra) for palabra in tokens_sin_stopwords]

        # Lematizar el texto
        lemmatizer = WordNetLemmatizer()
        tokens_lemmatized = [lemmatizer.lemmatize(palabra) for palabra in tokens_stemmed]

        # Unir los tokens lematizados en un texto preprocesado
        texto_preprocesado = " ".join(tokens_lemmatized)
        return texto_preprocesado


    def coerce_to_unicode(x):
        if sys.version_info < (3, 0):
            if isinstance(x, str):
                return unicode(x, 'utf-8')
            else:
                return unicode(x)
        else:
            return str(x)


    categorical_features = ['negativereason', 'tweet_coord', 'tweet_location', 'name', 'airline', 'tweet_created']
    numerical_features = ['tweet_id', 'retweet_count', 'airline_sentiment_confidence', 'negativereason_confidence']
    text_features = []
    for feature in categorical_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)
    for feature in text_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)
    for feature in numerical_features:
        if ml_dataset[feature].dtype == np.dtype('M8[ns]') or (
                hasattr(ml_dataset[feature].dtype, 'base') and ml_dataset[feature].dtype.base == np.dtype('M8[ns]')):
            ml_dataset[feature] = datetime_to_epoch(ml_dataset[feature])
        else:
            ml_dataset[feature] = ml_dataset[feature].astype('double')

    start_time = time.time()
    ml_dataset['text'] = ml_dataset['text'].apply(preprocess_text)
    ml_dataset['negativereason_confidence']= tqdm(ml_dataset['negativereason_confidence'].fillna(value=0),total=len(ml_dataset['negativereason_confidence']))
    ml_dataset['tweet_coord'] = tqdm(ml_dataset['tweet_coord'].apply(limpiar_coordenadas), total=len(ml_dataset['tweet_coord']))
    '''ml_dataset['tweet_location'] = tqdm(ml_dataset.apply(limpiar_tweet_location, axis=1),total=len(ml_dataset['tweet_location']))
    ml_dataset=imputar_valores_coordenadas(ml_dataset)
    ml_dataset['user_timezone'] = tqdm(ml_dataset.apply(limpiar_timezone, axis=1),total=len(ml_dataset['user_timezone']))
    ml_dataset.to_csv("limpieza_general.csv")'''

    end_time = time.time()
    total_time = end_time - start_time
    print("El proceso tardó:", total_time, "segundos")
    ml_dataset['__target__'] = ml_dataset['airline_sentiment']
    del ml_dataset['airline_sentiment']
    ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]
    #ml_dataset['__target__'] = ml_dataset['__target__'].astype(np.int64)

    train, test = train_test_split(ml_dataset, test_size=0.2, random_state=42, stratify=ml_dataset[['__target__']])

    LIMIT_DUMMIES = 100
    categorical_to_dummy_encode = ['negativereason', 'tweet_coord', 'tweet_location', 'name', 'airline',
                                   'tweet_created']
    
    # Only keep the top 100 values
    def select_dummy_values(train, features):
        dummy_values = {}
        categorical_features = []
        for feature in categorical_to_dummy_encode:
            values = [
                value
                for (value, _) in Counter(train[feature]).most_common(LIMIT_DUMMIES)
            ]
            dummy_values[feature] = values
            for dummy_value in dummy_values[feature]:
                categorical_features.append(u'%s_value_%s' % (feature, coerce_to_unicode(dummy_value)))
        return dummy_values, categorical_features

    DUMMY_VALUES, categorical_features = select_dummy_values(train, categorical_to_dummy_encode)

    def dummy_encode_dataframe(df):
        for (feature, dummy_values) in DUMMY_VALUES.items():
            for dummy_value in dummy_values:
                dummy_name = u'%s_value_%s' % (feature, coerce_to_unicode(dummy_value))
                df[dummy_name] = (df[feature] == dummy_value).astype(float)
            del df[feature]
            print('Dummy-encoded feature %s' % feature)


    dummy_encode_dataframe(train)

    dummy_encode_dataframe(test)

    X_train = train.drop('__target__', axis=1)
    X_test = test.drop('__target__', axis=1)
    y_train = np.array(train['__target__'])
    y_test = np.array(test['__target__'])

    # Vectorizar características de texto con tf-idf
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train['text'])
    X_test_tfidf = vectorizer.transform(X_test['text'])
    # Bag of Words
    vectorizer_bow = CountVectorizer()
    X_train_bow = vectorizer_bow.fit_transform(X_train['text'])
    X_test_bow = vectorizer_bow.transform(X_test['text'])

    # Concatenar matrices de características numéricas, categóricas y tf-idf
    X_train_tfidf = hstack([train[categorical_features], train[numerical_features], X_train_tfidf])
    X_test_tfidf = hstack([test[categorical_features], test[numerical_features], X_test_tfidf])

    X_train_bow = hstack([train[categorical_features], train[numerical_features], X_train_bow])
    X_test_bow = hstack([test[categorical_features], test[numerical_features], X_test_bow])

    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    best_f1_score_nb_bow = 0
    best_f1_score_nb_tfidf = 0
    mejor_resultado_nb_bow=None
    mejor_resultado_nb_tfidf=None
    nb_bow=None
    nb_tfidf=None
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
            mejor_resultado_nb_bow=predictions_nb_bow
            nb_bow=clf_nb_bow
        if f1_nb_tfidf > best_f1_score_nb_tfidf:
            best_f1_score_nb_tfidf = f1_nb_tfidf
            mejor_resultado_nb_tfidf=predictions_nb_tfidf
            nb_tfidf=clf_nb_tfidf
    print("El mejor modelo de Naive Bayes con BOW tiene un f1-score de:", best_f1_score_nb_bow)
    print("El mejor modelo de Naive Bayes con TF-IDF tiene un f1-score de:", best_f1_score_nb_tfidf)

    mejor_resultado_dt_bow=None
    mejor_resultado_dt_tfidf=None
    dt_bow=None
    dt_tfidf=None
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
                    mejor_resultado_dt_bow=predictions
                    dt_bow=clf


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
                    dt_tfidf=clf
                    mejor_resultado_dt_tfidf=predictions

    print("El mejor modelo de Decission Tree con BOW tiene un f1-score de:", max_f1_bow)
    print("El mejor modelo de Decission Tree con TF-IDF tiene un f1-score de:", max_f1_tfidf   )
    results_dict = {
    'text': X_test['text'], 
    'airline_sentiment':y_test,
    'nb_bow': mejor_resultado_nb_bow,
    'nb_tfidf': mejor_resultado_nb_tfidf,
    'dt_bow': mejor_resultado_dt_bow,
    'dt_tfidf': mejor_resultado_dt_tfidf
}
results_df = pd.DataFrame(results_dict)
results_df.to_csv("resultados.csv")
