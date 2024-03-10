# %% [markdown]
# # Analísis de comportamiento de clientes para la tienda de comestibles
# ### Enunciado del problema
# Una tienda de comestibles quiere conocer mejor a sus clientes para poder crear campañas de marketing personalizadas. Se le pide que desarrolle un modelo de clasificación utilizando Keras que tenga en cuenta la frecuencia de compra de los clientes, sus hábitos de gasto y la cantidad máxima que gastan en la tienda. El objetivo del modelo es clasificar a los clientes en tres categorías: valor bajo, medio y alto.
# Datos de entrenamiento adjuntos en un archivo llamado data_customer_classification
# 
# ### 1. Importar los datos a un data frame y mostrar su contenido inicial

# %%
#instalación de librerias
#!pip install openpyxl
#!pip install -U scikit-learn
# !pip install -U seaborn
# !pip install -U matplotlib
# %pip install -U keras
#%pip install -U tensorflow

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

# %%
data= pd.read_excel(io="Datos/data_customer_classification.xlsx", sheet_name="in")
data.head()

# %%
#validar si hay datos vacios
array2=[]
def isNaN(num):
    return num != num

for col in data:
  array=data[col].unique()
  for dato in array:
    if(isNaN(dato)):
      array2.append(col)
      continue
print(array2)

# %% [markdown]
# El dataframe no contiene ninguna columna vacía

# %% [markdown]
# ### 2. Análisis estádistico básico.

# %%
### validar los tipos de datos que contienen cada columna del DataFrame
data.info()

# %%
data.describe()

# %% [markdown]
# ### 3. Preparación de los datos 

# %%
# Calcular la frecuencia de compra por cliente
shopping_frequency = data.groupby('customer_id').size().reset_index(name='shopping_frequency')

# Calcular el monto máximo de transacción por cliente
tran_amount_max = data.groupby('customer_id')['tran_amount'].max().reset_index(name='tran_amount_max')

# Calcular el monto promedio de transacción por cliente
tran_amount_mean = data.groupby('customer_id')['tran_amount'].mean().reset_index(name='tran_amount_mean')

# Convertir la columna 'trans_date' a tipo datetime
data['trans_date'] = pd.to_datetime(data['trans_date'])

# Calcular la fecha más reciente y la fecha más antigua para cada cliente
customer_date_range = data.groupby('customer_id')['trans_date'].agg(['min', 'max']).reset_index()

# Calcular la diferencia entre la fecha más reciente y la fecha más antigua
customer_date_range['date_range'] = customer_date_range['max'] - customer_date_range['min']

# Convertir la diferencia de fechas a días
customer_date_range['date_range_days'] = customer_date_range['date_range'].dt.days

# Unir todos los cálculos al DataFrame original
customer_features = shopping_frequency.merge(tran_amount_max, on='customer_id')\
                                    .merge(tran_amount_mean, on='customer_id')\
                                    .merge(customer_date_range[['customer_id', 'date_range_days']], on='customer_id')
customer_features

# %% [markdown]
# #### 3.1 Clasificación en los grupos valor bajo, medio y alto.

# %%
#Clasificación en los grupos valor bajo, medio y alto.
# Estandarizar las características (frecuencia de compra, monto máximo gastado, monto promedio gastado y rango de fechas)
customer_features_standardized = (customer_features[['shopping_frequency', 'tran_amount_max', 'tran_amount_mean', 'date_range_days']] - customer_features[['shopping_frequency', 'tran_amount_max', 'tran_amount_mean', 'date_range_days']].mean()) / customer_features[['shopping_frequency', 'tran_amount_max', 'tran_amount_mean', 'date_range_days']].std()

# Calcular la puntuación Z promedio para cada cliente resultado de aplicar la normalización basada en la desviación estándar.
customer_features_standardized['z_score'] = customer_features_standardized.mean(axis=1)

# Definir los límites entre los grupos (bajo, medio, alto) basados en los percentiles
low_limit = customer_features_standardized['z_score'].quantile(1/3)
high_limit = customer_features_standardized['z_score'].quantile(2/3)

# Función para clasificar a los clientes en grupos
def classify_group(z_score):
    if z_score <= low_limit:
        return 'valor bajo'
    elif z_score <= high_limit:
        return 'valor medio'
    else:
        return 'valor alto'

# Agregar una columna con la clasificación de los clientes en los tres grupos
customer_features['group'] = customer_features_standardized['z_score'].apply(classify_group)

# Mostrar el DataFrame resultante
customer_features

# %%
#aplicar label encoding para preparar los datos
# Crear una instancia de LabelEncoder
label_encoder = LabelEncoder()

# Aplicar label encoding a la columna 'group' esto permitirá al modelo procesar la variable categorica objetivo
customer_features['group_encoded'] = label_encoder.fit_transform(customer_features['group'])
#targets=label_encoder.inverse_transform(np.array([0, 1,2]))
#print(targets)
#['valor alto' 'valor bajo' 'valor medio']
customer_features


# %% [markdown]
# ### 4. Gráficos estadísticos

# %%
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='group', data=customer_features, palette='Set3')

# Agregar el número de cada barra sobre el mismo
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')

plt.title('Cantidad de clientes por grupo')
plt.xlabel('Grupo')
plt.ylabel('Cantidad de clientes')
plt.show()

plt.figure(figsize=(8, 6))
customer_features['group'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'lightgreen', 'lightcoral'])
plt.title('Proporción de clientes por grupo')
plt.ylabel('')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='group', y='shopping_frequency', data=customer_features)
plt.title('Distribución de frecuencia de compra por grupo')
plt.xlabel('Grupo')
plt.ylabel('Frecuencia de compra')
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x='tran_amount_mean', y='tran_amount_max', hue='group', data=customer_features, palette='Set1')
plt.title('Relación entre monto promedio y máximo gastado por grupo')
plt.xlabel('Monto promedio gastado')
plt.ylabel('Monto máximo gastado')
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x='shopping_frequency', y='tran_amount_mean', hue='group', data=customer_features, palette='Set1')
plt.title('Relación entre frecuencia de compra y monto promedio por grupo')
plt.xlabel('Frecuencia de compra')
plt.ylabel('Monto promedio gastado')
plt.show()


plt.figure(figsize=(8, 6))
sns.violinplot(x='group', y='date_range_days', data=customer_features)
plt.title('Distribución del rango de fechas por grupo')
plt.xlabel('Grupo')
plt.ylabel('Rango de fechas (días)')
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(x='shopping_frequency', hue='group', data=customer_features, bins=20, kde=True, palette='Set2')
plt.title('Distribución de frecuencia de compra por grupo')
plt.xlabel('Frecuencia de compra')
plt.ylabel('Número de clientes')
plt.show()


# %% [markdown]
# Los datos están balanceados en las tres categorias, punto importante para un entrenamiento exitoso del modelo. 

# %% [markdown]
# ### 5. Segmentación de datos train y test

# %%
# Dividir los datos en características (X) y la variable objetivo (y)
X = customer_features.drop(['customer_id','group','group_encoded'], axis=1)  # características
y = customer_features['group_encoded']  # variable objetivo

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mostrar la forma de los conjuntos de entrenamiento y prueba
print("Forma del conjunto de entrenamiento:", X_train.shape, y_train.shape)
print("Forma del conjunto de prueba:", X_test.shape, y_test.shape)

# %% [markdown]
# ### 6. Modelo

# %%
model = Sequential()

# Añadir capas densas 
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))  
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))  # 3 neuronas de salida para las 3 categorías (valor bajo, valor medio, valor alto)

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# %% [markdown]
# #### 6.1 Entrenamiento del modelo

# %%
# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, verbose=1)


# %% [markdown]
# #### 6.2 Gráficas de las métricas del entrenamiento del modelo

# %%
# Gráfico de la precisión
plt.plot(history.history['accuracy'], label='Accuracy (train)')
plt.plot(history.history['val_accuracy'], label='Accuracy (val)')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Gráfico de la pérdida
plt.plot(history.history['loss'], label='Loss (train)')
plt.plot(history.history['val_loss'], label='Loss (val)')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# %% [markdown]
# ### 7. Evaluación del modelo

# %%
# Evaluación del modelo en el conjunto de datos de prueba
loss, accuracy = model.evaluate(X_test, y_test)
print("Precisión en los datos de prueba:", accuracy)

# %%
y_pred_prob = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_prob, axis=-1)

# Convertir las clases verdaderas en el conjunto de datos de prueba en un solo array
y_true_classes = y_test.tolist()

precision = precision_score(y_true_classes, y_pred_classes, average=None)
recall = recall_score(y_true_classes, y_pred_classes, average=None)
f1 = f1_score(y_true_classes, y_pred_classes, average=None)
accuracy = accuracy_score(y_true_classes, y_pred_classes)

# Generar el informe de clasificación.
class_names =['Valor alto', 'Valor bajo', 'Valor medio']
classification_rep = classification_report(y_true_classes, y_pred_classes, target_names=class_names)

# Imprimir resultados
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_rep)

# %% [markdown]
# Precisión, Sensibilidad y F1-score: todas estas métricas son altas para las tres clases. Esto indica que el modelo es capaz de clasificar las instancias en cada clase; sin embargo, es posible mejorar la clasificación en las clases de valor medio y alto aplicando técnicas más avanzadas en normalización de los datos antes de entrenar el modelo y usar otras arquitectura más complejas en el modelo.
# 
# Exactitud (Accuracy): la exactitud general del modelo es del 88%, lo que significa que aproximadamente el 88% de todas las predicciones son correctas.
# 
# Reporte de Clasificación: el reporte de clasificación muestra métricas detalladas para cada clase, incluyendo precisión, sensibilidad y F1-score. Además, las métricas macro y weighted avg muestran promedios de estas métricas que tienen en cuenta el desequilibrio de clases.
# 
# Soporte (Support): el soporte muestra el número de ocurrencias reales de cada clase en el conjunto de prueba.

# %% [markdown]
# ### 8. Exportar el modelo

# %%
# Guardar la arquitectura del modelo en formato JSON
model_json = model.to_json()
with open("Modelo/modelo.json", "w") as json_file:
    json_file.write(model_json)

# Guardar los pesos aprendidos del modelo en formato HDF5
model.save_weights("Modelo/pesos_modelo.h5")

# %%
# Cargar la arquitectura del modelo desde el archivo JSON
with open("Modelo/modelo.json", "r") as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

# Cargar los pesos aprendidos del modelo desde el archivo HDF5
loaded_model.load_weights("Modelo/pesos_modelo.h5")

# %%
y_pred_classes=loaded_model.predict(X_test)
y_pred_classes = np.argmax(y_pred_prob, axis=-1)

# Convertir las clases verdaderas en el conjunto de datos de prueba en un solo array
y_true_classes = y_test.tolist()

precision = precision_score(y_true_classes, y_pred_classes, average=None)
recall = recall_score(y_true_classes, y_pred_classes, average=None)
f1 = f1_score(y_true_classes, y_pred_classes, average=None)
accuracy = accuracy_score(y_true_classes, y_pred_classes)

# Generar el informe de clasificación.
class_names = ['Valor alto', 'Valor bajo', 'Valor medio']
classification_rep = classification_report(y_true_classes, y_pred_classes, target_names=class_names)

# Imprimir resultados
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_rep)

# %%
X_train.to_csv('Datos/train/train_X.csv')
y_train.to_csv('Datos/train/train_y.csv')
X_test.to_csv('Datos/test/test_X.csv')
y_test.to_csv( 'Datos/test/test_y.csv' )

# %% [markdown]
# ### 9. Exportar las librerias y versiones usadas en el desarrollo del modelo.

# %%
%pip freeze numpy scikit-learn pandas seaborn matplotlib keras tensorflow > requirements.txt
#pip install -r requirements.txt



