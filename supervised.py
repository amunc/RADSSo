# -*- coding: utf-8 -*-

''' RIASC Automated Decision Support Software (RADSSo) generates the best supervised/unsupervised model,
    in an automated process, based on some input features and one target feature, to solve a multi-CASH problem.

    Copyright (C) 2018  by RIASC Universidad de Leon (Ángel Luis Muñoz Castañeda, Mario Fernández Rodríguez, Noemí De Castro García y Miguel Carriegos Vieira)
    This file is part of RIASC Automated Decision Support Software (RADSSo)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
	
	You can find more information about the project at https://github.com/amunc/RADSSo'''

import os
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

import auxiliary_functions as auxf
    
def calculate_model_accuray(model, target_data_set, features, target, out):
    '''
    Permite obtener el porcentaje de acierto del modelo sobre el target_data_set que se le pasa como parametro

    Parameters:
    :param sklearn-model model: modelo de aprendizaje obtenido mediante el uso de scikit-learn    
    :param pandas-dataframe target_data_set: conjunto de datos donde las filas son las observaciones y las columnas las variables. Debe incluir la variable objetivo.
    :param list features: lista con las caracteristicas relevantes usadas en la creacion del modelo
    :param str target: caracteristica objetivo del modelo
    :param int out: parametro con valores posibles 1|numero entero distinto de 1. Si se establece a 1 permite mostrar la informacion del accuracy de forma mas detallada
    
    :return: porcentaje de acierto
    :rtype: float
    '''
    
    predictions=model.predict(target_data_set[features])
    p=[]
    L=list(predictions)
    M=list(target_data_set[target])
    accuracy=auxf.compare_lists(L,M)/float(len(M))
    accuracy=float(round(accuracy,5))
    if out==1:
        print('Number of errors =', sum(p), 'accuracy=', accuracy)
        return(accuracy)
    else:
        return(accuracy)
        
        
def save_current_model_to_file(modelo,ruta_directorio,nombre_fichero):
    '''
    Permite generar un fichero .pkl con el modelo de aprendizaje generado usando scikit-learn

    Parameters:
    :param sklearn-model model: modelo de aprendizaje obtenido mediante el uso de scikit-learn    
    :param str ruta_directorio: ruta completa al directorio donde se quiere guardar el modelo
    :param str nombre_fichero: nombre del fichero que almacenará el modelo (debe incluir la extension)    
    
    :return: None
    '''
    
    ruta_destino = os.path.join(ruta_directorio,nombre_fichero)
    joblib.dump(modelo, ruta_destino)
    return ruta_destino
    

def create_customized_tree(train_data, features, target, depth):
    '''
    Permite crear el modelo DecisionTreeClassifier usando la libreria sklearn

    Parameters:
    :param pandas-dataframe train_data: datos de entrenamiento con la columna target incluida
    :param list features: lista con las caracteristicas relevantes para entrenar el modelo
    :param str target: caracteristica objetivo del modelo
    :param int depth: profundidad deseada de los arboles que componen el modelo
    
    :return: modelo DecisionTreeClassifier entrenado
    :rtype: sklearn-model
    '''
    
    X=train_data[features]
    y=train_data[target]
    if(depth == ''):
        decision_tree= DecisionTreeClassifier(criterion="entropy")
    else:
        decision_tree= DecisionTreeClassifier(max_depth=int(depth), criterion="entropy")
    my_tree=decision_tree.fit(X,y)
    return(my_tree)
    
    
def create_customized_kneighbor(train_data,features,target,num_neighbors,leafsize):
    '''
    Permite crear el modelo KneighborClassifier usando la libreria sklearn

    Parameters:
    :param pandas-dataframe train_data: datos de entrenamiento con la columna target incluida
    :param list features: lista con las caracteristicas relevantes para entrenar el modelo
    :param str target: caracteristica objetivo del modelo
    :param int num_neighbors: numero de vecinos usados para clasificar un elemento 
    :param float leafsize: en el caso de que se use KDtree o BallTree como algoritmos
    
    :return: modelo KNeighborsClassifier entrenado
    :rtype: sklearn-model
    '''
    
    X=train_data[features]
    y=train_data[target]
    if(num_neighbors == '' and leafsize == ''):
        mi_neigh = KNeighborsClassifier(weights='uniform', algorithm='auto', p=2, metric='minkowski', metric_params=None, n_jobs=1)
    else:
        mi_neigh = KNeighborsClassifier(n_neighbors=num_neighbors, weights='uniform', algorithm='auto', leaf_size=leafsize, p=2, metric='minkowski', metric_params=None, n_jobs=1)
    neigh=mi_neigh.fit(X,y)
    return(neigh)
    
#####################################################################################################################################
################################################## MODELOS ENSEMBLE #################################################################
#####################################################################################################################################
    
def create_customized_ada(train_data, features, target, estimators):
    '''
    Permite crear el modelo AdaBoostClassifier usando la libreria sklearn

    Parameters:
    :param pandas-dataframe train_data: datos de entrenamiento con la columna target incluida
    :param list features: lista con las caracteristicas relevantes para entrenar el modelo
    :param str target: caracteristica objetivo del modelo
    :param int estimators: numero de estimadores para obtener la clasificacion final
    
    :return: modelo AdaBoostClassifier entrenado
    :rtype: sklearn-model
    '''
    
    X=train_data[features]
    y=train_data[target]
    if(estimators == ''):
        mi_ada=AdaBoostClassifier(base_estimator=None, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
    else:
        mi_ada=AdaBoostClassifier(base_estimator=None, n_estimators=estimators, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
    ada=mi_ada.fit(X,y)
    return(ada)


def create_customized_boosting(train_data,features,target, depth, estimators):
    '''
    Permite crear el modelo GradientBoostingClassifier usando la libreria sklearn

    Parameters:
    :param pandas-dataframe train_data: datos de entrenamiento con la columna target incluida
    :param list features: lista con las caracteristicas relevantes para entrenar el modelo
    :param str target: caracteristica objetivo del modelo
    :param int estimators: numero de estimadores para obtener la clasificacion final
    :param int depth: profundidad deseada de los arboles que componen el modelo
    
    :return: modelo  GradientBoostingClassifier entrenado
    :rtype: sklearn-model
    '''
    
    X=train_data[features]
    y=train_data[target]
    if(estimators == '' and depth == ''):
        gradientboosting=GradientBoostingClassifier()
    else:
        gradientboosting=GradientBoostingClassifier(n_estimators= estimators,max_depth= depth)
    boosting=gradientboosting.fit(X,y)
    return(boosting)
    
   
def create_customized_forest(train_data, features, target, depth, estimators):
    '''
    Permite crear el modelo RandomForestClassifier usando la libreria sklearn

    Parameters:
    :param pandas-dataframe train_data: datos de entrenamiento con la columna target incluida
    :param list features: lista con las caracteristicas relevantes para entrenar el modelo
    :param str target: caracteristica objetivo del modelo
    :param int estimators: numero de estimadores para obtener la clasificacion final
    :param int depth: profundidad deseada de los arboles que componen el modelo
    
    :return: modelo RandomForestClassifier entrenado
    :rtype: sklearn-model
    '''
    
    X=train_data[features]
    y=train_data[target]
    if(estimators == '' and depth == ''):
        random_forest=RandomForestClassifier(criterion="entropy")
    else:
        random_forest=RandomForestClassifier(n_estimators= estimators, criterion="entropy", max_depth= depth)
    forest=random_forest.fit(X,y)
    return(forest)


#####################################################################################################################################
################################################## MODELOS REDES NEURONALES #################################################################
#####################################################################################################################################

def create_customized_mlp(train_data,features,target,layer_sizes,act_function):#(hidden_layer_sizes=(4,2), activation=’relu’):#, solver=’adam’, alpha=0.0001, batch_size=’auto’, learning_rate=’constant’, learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08):
    '''
    Permite crear el modelo MLPClassifier usando la libreria sklearn

    Parameters:
    :param pandas-dataframe train_data: datos de entrenamiento con la columna target incluida
    :param list features: lista con las caracteristicas relevantes para entrenar el modelo
    :param str target: caracteristica objetivo del modelo
    :param tuple layer_sizes: tupla que contiene el numero de capas y los perceptrones en cada capa
    :param str act_func: funcion de activacion para la capa 
    
    :return: modelo MLPClassifier entrenado
    :rtype: sklearn-model
    '''
    X=train_data[features]
    y=train_data[target]
    if(layer_sizes == '' and act_function == ''):
        mlp=MLPClassifier()
    else:
        mlp=MLPClassifier(hidden_layer_sizes=layer_sizes, activation=act_function,solver='adam')
    my_mlp=mlp.fit(X,y)
    return(my_mlp)


    
def get_trained_model(model_name,train_data,features,target,params_array,diccionario_modelos_supervisado):
    '''
    The function allows to create and train a model with the specified name

    Parameters:
    :param str model_name: name of the model to be trained
    :param pandas-dataframe train_data: Data to train de model. It includes the target column
    :param list features: List with the relevant features to rain the model
    :param str target: Target feature
    :param list params_array: Array with the specific parameters of the model to be trained
    
    :return: specified model trained
    :rtype: sklearn-model
    '''
    modelo_creado = ''
  
    if(model_name == diccionario_modelos_supervisado[1]): #Tree
        modelo_creado = create_customized_tree(train_data, features, target, params_array[0])
        
    elif(model_name == diccionario_modelos_supervisado[2]):#Ada
        modelo_creado = create_customized_ada(train_data, features, target, params_array[0])
    
    elif(model_name == diccionario_modelos_supervisado[3]):#Boosting
        modelo_creado = create_customized_boosting(train_data,features,target, params_array[0]*2, params_array[1]*2)
    
    elif(model_name == diccionario_modelos_supervisado[4]):#RandomForest
        modelo_creado = create_customized_forest(train_data, features, target, params_array[0]*2, params_array[1]*2)
    
    elif(model_name == diccionario_modelos_supervisado[5]):#MLP
        modelo_creado = create_customized_mlp(train_data,features,target,params_array[0],params_array[1])
        
    return modelo_creado

def initialize_model(model_name,params_array,diccionario_modelos_supervisado):
    '''
    The function allows to create and train a model with the specified name

    Parameters:
    :param str model_name: name of the model to be trained
    :param pandas-dataframe train_data: Data to train de model. It includes the target column
    :param list features: List with the relevant features to rain the model
    :param str target: Target feature
    :param list params_array: Array with the specific parameters of the model to be trained
    
    :return: specified model trained
    :rtype: sklearn-model
    
    kneigh = KNeighborsClassifier(n_neighbors=int(num_neighbors), weights='uniform', algorithm='auto', leaf_size=int(leafsize), p=2, metric='minkowski', metric_params=None, n_jobs=1)
    gradientboosting=GradientBoostingClassifier(n_estimators=int(estimators),max_depth=int(depth)) 
    random_forest=RandomForestClassifier(n_estimators=int(estimators), criterion="entropy", max_depth=int(depth))
    mlp=MLPClassifier(hidden_layer_sizes=(int(layer_sizes[0]),int(layer_sizes[1])), activation=act_function)
    decision_tree= DecisionTreeClassifier(max_depth=int(depth), criterion="entropy") 
    ada=AdaBoostClassifier(base_estimator=None, n_estimators=int(estimators), learning_rate=1.0, algorithm='SAMME.R', random_state=None)
    
    '''
    modelo_inicializado = ''
    if(model_name == 'Kneighbor'):
        modelo_inicializado =  KNeighborsClassifier(n_neighbors=params_array[0], weights='uniform', algorithm='auto', leaf_size=params_array[1], p=2, metric='minkowski', metric_params=None, n_jobs=1)
    
    elif(model_name == diccionario_modelos_supervisado[1]):#Tree
        modelo_inicializado = DecisionTreeClassifier(max_depth=params_array[0], criterion="entropy")
        
    elif(model_name == diccionario_modelos_supervisado[2]):#Ada
        modelo_inicializado = AdaBoostClassifier(base_estimator=None, n_estimators=params_array[0], learning_rate=1.0, algorithm='SAMME.R', random_state=None)
    
    elif(model_name == diccionario_modelos_supervisado[3]):#Boosting
        modelo_inicializado = GradientBoostingClassifier(n_estimators=params_array[0],max_depth=params_array[1])
    
    elif(model_name == diccionario_modelos_supervisado[4]):#RandomForest
        modelo_inicializado = RandomForestClassifier(n_estimators=params_array[1], criterion="entropy", max_depth=params_array[0])
    
    elif(model_name == diccionario_modelos_supervisado[5]):#MLP
        modelo_inicializado = MLPClassifier(hidden_layer_sizes=(params_array[0]), activation=params_array[1])
        
    return modelo_inicializado