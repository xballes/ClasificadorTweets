# Clasificación de sentimientos de comentarios de aerolíneas

Este repositorio contiene una función de clasificación que utiliza dos algoritmos de aprendizaje automático para clasificar los sentimientos de los comentarios de las aerolíneas. La función está escrita en Python y utiliza el paquete scikit-learn para el preprocesamiento de texto y la implementación de los modelos de clasificación.

## Limpieza
Los dos subprogramas de limpieza se encargan de procesar y limpiar los datos del archivo csv antes de su análisis.
### imputar_timezone_coord
El primer subprograma "imputar_timezone_coord" busca valores faltantes en la columna "tweet_coord" y "user_timezone" del dataframe y los completa en base a la información de la columna "user_timezone" o "airline". Si la información de "user_timezone" está presente, busca las coordenadas de la ciudad correspondiente y las asigna a la columna "tweet_coord". Si no se dispone de información de "user_timezone", busca las coordenadas del headquarters de la aerolínea correspondiente y las asigna a la columna "tweet_coord". Si ninguna de las dos está disponible, no hace nada. Luego, si la información de "tweet_coord" está presente pero la de "user_timezone" no, busca el uso horario de la ciudad correspondiente y la asigna a la columna "user_timezone".
### limpiar
El segundo subprograma "limpiar" elimina filas con ciudades vacías o fechas inválidas del dataframe. También se puede filtrar el dataframe para que solo incluya tweets de una aerolínea específica. La función imprime las filas eliminadas y el número total de filas eliminadas.
## Cómo funciona la función
La función clasificador(df) toma como entrada un DataFrame con las siguientes columnas:

**airline_sentiment:** el sentimiento del comentario de la aerolínea (positivo, negativo o neutral).
**airline_sentiment_confidence:** la confianza en la clasificación del sentimiento.
**text:** el texto del comentario.
**tweet_created:** la fecha de creación del tweet.

### La función realiza las siguientes operaciones de preprocesamiento de texto:

Elimina los valores faltantes y las menciones (@) del texto.
Convierte el texto a minúsculas y elimina los signos de puntuación.
Divide el conjunto de datos en conjuntos de entrenamiento y prueba.
Vectoriza los datos de texto utilizando CountVectorizer.
A continuación, la función entrena dos modelos de aprendizaje automático: Naive Bayes y Árbol de decisión. Después de entrenar los modelos, se hace una predicción para todo el conjunto de datos y se agregan las columnas de sentimiento al DataFrame. Además, se guarda el DataFrame con las predicciones en un archivo CSV llamado 'clasificacion.csv'.

Finalmente, la función hace predicciones para los datos de prueba y calcula el f1-score, precisión y recall para ambos modelos. Los resultados se imprimen en la consola.

Cómo usar la función
## La función requiere los siguientes paquetes de Python:
scikit-learn

pandas

numpy

matplotlib

geopy

ntlk
