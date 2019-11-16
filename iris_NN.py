#!/usr/bin/env python
# coding: utf-8

# # Ejemplo de entrenamiento de una red neuronal MLP
# 
# Este es un ejemplo para el curso de Inteligencia Artificial 2019-II, entrenando un modelo de red neuronal MLP para el conjunto de datos iris.cvs.
# 
# El conjunto de datos Iris contiene datos sobre tres tipos de flores Iris. Este es un conjunto de datos multi-variables construidos por Edgar Anderson para cuantificar la variación morfológica de tres especies de flores de iris.
# 
# El conjunto de datos contiene tres clases de flores que son: Iris Setosa, Iris Versicolour, e Iris Virginica. Cada clase cuenta con 50 ejemplos registrados, para un total de 150 ejemplos en el conjunto de datos. 
# 
# Los atributos, variables independientes, o características registrados para cada ejemplo son:
# 
# 1. sepal length in cm 
# 2. sepal width in cm 
# 3. petal length in cm 
# 4. petal width in cm
# 
# El conjunto de datos está disponible en: https://archive.ics.uci.edu/ml/datasets/Iris/

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV


# Cargando el conjunto de datos de un archivo extensión .cvs, y mostrando información del archivo.

# In[2]:


#Cargando datos
#No se le olvide actualizar este path a la ubicación del archivo de datos
iris = pd.read_csv(r"D:\Usuarios\AbrahamNarVal\Documentos\Unimag\2019-2\Inteligencia Artificial\Clases\Seguimiento 3\0 - Proyecto Final\ejemplo_con_iris/training2.csv")
#Informacion de los datos
print(iris.info())


# Visualizando la distribución de las clases a través de un histograma.

# In[3]:


#Histograma del atributo clase
ax=plt.subplots(1,1,figsize=(10,8))
sns.countplot('clas',data=iris)
plt.title("clas")
plt.show()


# Visualizando los histogramas de cada atributo.

# In[4]:


#Histograma de atributos predictores

iris.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
#fig.set_size_inches(12,12)
plt.show()


# Diagrama de cajas de los atributos o variables independientes.

# In[5]:


#boxplot de las variables numericas
iris = iris.drop('Id',axis=1)
box_data = iris #variable representing the data array
box_target =iris.clas #variable representing the labels array
sns.boxplot(data = box_data,width=0.5,fliersize=5)
#sns.set(rc={'figure.figsize':(2,15)})
plt.show()


# Observando la correlación entre variables permite descubrir posibles dependencias entre las variables independientes.

# In[6]:


#observando correlacion entre variables
X = iris.iloc[:, 0:4]
f, ax = plt.subplots(figsize=(10, 8))
corr = X.corr()
print(corr)
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), 
          cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax, linewidths=.5)
plt.show() 


# En la matriz de correlación se observa un alto coeficiente para las variables petalWidth y PetalLength. Podemos mirar el comportamiento de las dos variables utilizando regresión lineal.

# In[7]:


#observando relaciones entre los datos
#sns.regplot(x='PetalLengthCm', y='PetalWidthCm', data=iris);
#sns.set(rc={'figure.figsize':(2,5)})
#plt.show()


# Una vez observado y analizado las variables del conjunto de datos vamos a hacer una primera prueba preliminar para observar cómo se comportaría el modelo de red neuronal. La configuración de este primer modelo se indica a través de los parámetros de MPLClassifier

# In[8]:


#Separando los datos en conjuntos de entrenaimiento y prueba
X = iris.iloc[:, :-1].values
y = iris.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
'''

#Como esta es una primera prueba prelimintar coloco esta instrucción para que nos me saque un warning
#debido a que el modelo no alcanza a converger
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

#Entrenando un modelo de red neuronal MLP para clasificación
#MLPClassifier permite configurar las capas ocultas del modelo, la instrucción de abajo indica que el modelo tendrá
#dos capas ocultas cada una con 3 neuronas. Algo como esto hidden_layer_sizes = (3,3,2) indicarían tres capas ocultas con
#3,3 y 2 neuronas respectivamente
model =  MLPClassifier(hidden_layer_sizes = (2,2), alpha=0.01, max_iter=1000) 
model.fit(X_train, y_train) #Training the model


# Una vez entrenado el modelo, debemos evaluarlo sobre el conjunto de datos reservado para prueba, y utilizar algunas métricas para observar que tan bien quedo entrenado el modelo. En esta primera prueba utilizamos como métricas el porcentaje de precisión del modelo y la matriz de confusión.

# In[9]:


#Test the model
predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predictions))


# Vamos a convertir el atributo clase de variable categórica a numérica para aplicar MLPregression.

# In[10]:


irisclass = iris['clas']
irisclass_econded, irisclass_categories = irisclass.factorize()
print(irisclass_econded)
print(irisclass_categories[:10])


# Ahora vamos a volver a crear un conjunto de entrenamiento y prueba para entrenar el modelo de red neuronal MLP para regresión. La configuración de la red, se especifica en los parámetros de MLPRegressor. Luego evaluamos el modelo entrenado y calculamos el error cuadrático medio, la cual es una métrica de error utilizada para regresión.

# In[11]:


#Creando conjuntos de entrenamiento y prueba para regresión
X_train_R, X_test_R, y_train_R, y_test_R = train_test_split(X, irisclass_econded, test_size=0.20)
model =  MLPRegressor(hidden_layer_sizes = (2,2), alpha=0.01, max_iter=1000) 
model.fit(X_train_R, y_train_R) #Training the model


# In[12]:


#Test the model
predictions_R = model.predict(X_test_R)
print(mean_squared_error(y_test_R, predictions_R))


# Ahora vamos a ajustar los parámetros del modelo utilizando GridSearch

# In[13]:


param_grid = [{'hidden_layer_sizes' : [(2,2), (3,3), (4,4), (5,4)], 'max_iter':[100, 500, 1000]}, 
              {'alpha': [0.0001, 0.001, 0.01, 0.1]}]


# In[14]:


model = MLPClassifier()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', iid=False)
grid_search.fit(X_train, y_train)


# In[15]:


print(grid_search.best_params_)


# Ahora ajustaremos los parámetros para la clase MLPRegressor y mostraremos los resultados obtenidos con lo mejores parámetros encontrados.

# In[16]:


#Observe que son las mimas instrucciones utilizadas para la clase MLPClassifier
model2 = MLPRegressor()
grid_search2 = GridSearchCV(model2, param_grid, cv=5, scoring='neg_mean_squared_error', iid=False)
grid_search2.fit(X_train_R, y_train_R)


# In[17]:


print(grid_search2.best_params_) #Mejores parámetros encontrados para MLPRegressor


# Ahora vamos a graficar los resultados obtenidos. En la gráfica se podrá observar un plot de los datos original, de la aproximación obtenida con el primer modelo sin ajustar parámetros, y del modelo con los mejores parámetros encontrados por GridSearchCV.

# In[18]:


R_ind = grid_search2.best_estimator_  #
new_predictions_R = R_ind.predict(X_test_R) #Utilizamos los parámetros encontrados para volver 
print('Error cuadrático medio obtenido con los parámetros encontrados por GridSearchCV:')
print(mean_squared_error(y_test_R, new_predictions_R))

X = np.arange(1, len(y_test_R)+1)
plt.figure()
plt.plot(X, y_test_R, 'k', label='Datos Originales')
plt.plot(X, predictions_R, 'r', label='Primera Aproximación')
plt.plot(X, new_predictions_R, 'g', label='Segunda Aproximación')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Original data Vs Predictions')
plt.legend()
plt.show()
'''
