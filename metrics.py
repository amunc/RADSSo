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

import itertools
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import reports_generation as repg


from sklearn.metrics import confusion_matrix, normalized_mutual_info_score
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.utils.multiclass import unique_labels

import warnings
warnings.filterwarnings("ignore")


def get_all_available_cross_validation_iterators(number_of_divisions,ts_size):
    '''
    This function allows to obtain different iteratos and to use diferent division
    and sizes for the iterations to execute the cross validation

    Parameters:
    :param int number_of_divisions: when it is allowed for the iterator, the number of divisions can be specified
    :param int ts_size: when it is allowed for the iterator, the size of the sample can be specified
    
    :return: list of iterators
    :rtype: list<list<str,sklearn.model_selection.iterator>>
    '''
    
    shuffle_split_it = ShuffleSplit(n_splits=number_of_divisions,test_size=ts_size, random_state=0)
    iterators_list = [['ShuffleSplit',shuffle_split_it]]    
    return iterators_list


def get_all_available_scoring_methods():
    '''
    This function allows to obtain different scoring methods for the supervised models

    Parameters:
    :param None
    
    :return: list of scoring methods
    :rtype: list<list<str,str>>
    '''
    
    list_scoring_methods = [['accuracy','accuracy']]
    return list_scoring_methods

def get_scoring_methods_unsupervised():
    '''
    This function allows to obtain different scoring methods for the unsupervised models

    Parameters:
    :param None
    
    :return: list of scoring methods
    :rtype: list<list<str,str>>
    '''
    list_scoring_methods = [['normalized_mutual_info_score','normalized_mutual_info_score']]                            
    return list_scoring_methods


def transform_labels(y_true, y_pred):
    """It turns the y_true and y_pred labels
    to contain values in [0, n), being n the number
    of unique values. 
    New lists of y_true and y_pred (non in place modification are done)
    
        
    Parameters:
    :param array: y_true: shape(n, ) with real labels
    :param array: y_pred: shape(n, ) with predicted labels
    
    
    :return list  labels: list with unique labels
    :return array y_true: shape(n, ) with real labels transformed
    :return array y_pred: shape(n, ) with predicted labels transformed
    """
    
    labels = unique_labels(y_true, y_pred)
    n_labels = labels.size
    label_to_ind = dict((y, x) for x, y in enumerate(labels))
    y_pred = np.array([label_to_ind.get(x, n_labels + 1) for x in y_pred])
    y_true = np.array([label_to_ind.get(x, n_labels + 1) for x in y_true])
    reverse_label_to_ind = {}
    for key in label_to_ind:
        reverse_label_to_ind[label_to_ind[key]] = key
    
    return labels, y_true, y_pred


def confusion_matrix_gather_feature_incorrect_values(df, y_true, y_pred, feature):
    """It returns the value for a concrete feature for each element of
    the original dataframe for the consufion matrix.
    
    The mapping between indexes in the output matrix and the real values of the target
    is done using the corresponding index in the matrix that is the same they would have
    in a list ordered in ascending order
    Ej:
    
    labels = [2,5,4,1]
    The class 1 will be in index 0
    The class 2 will be in index 1
    The class 3 will be in index 2
    
    Parameters:
    df: pandas.DataFrame with original data
    y_true: array shape(n, ) with real values of target
    y_pred: array shape(n, ) with predicted values of target
    feature: name of the column of the 'df' which value is going to be extracted
    
    Return:
    matrix: numpy.ndarray shape(m, m) being m the number of
        unique labels. The structure is matrix[a,b]
        that contains a numpy.ndarray with the value of field feature
        with the elements of the 'a' class that has been 
        predicted incorrectly as 'b' class"""        
        
    labels, y_true, y_pred = transform_labels(y_true, y_pred)
    n_labels = labels.size
    
    matrix = [[0 for i in range(n_labels)] for j in range(n_labels)]
    missclassified = y_true != y_pred
    for i in range(n_labels):
        for j in range(n_labels):
            mask = np.logical_and(y_true == i, y_pred == j)
            mask = np.logical_and(mask, missclassified)
            indices, = np.nonzero(mask)            
            values = [df[feature][index] for index in indices]
            matrix[i][j] = list(values)
    return matrix


def confusion_matrix_to_csv(matrix,list_labels, path, fillvalue=''):
    """Escribe un csv en el que en cada columna corresponde
    a elementos de la clase i que fueron clasificados como elementos 
    de la clase j
    
    Argumentos:
    matrix -- list shape(n, n) con n el número de clases distintas y
        en cada elemento (i,j) una lista con el valor de una columna
        para cada instancia de la clase i que fuese clasificada como
        j. Básicamente la salida de `confusion_matrix_values`
    path -- string con la ruta del fichero donde se escriben los datos
    fillvalue -- un valor cualquiera para rellenar caundo no existan valores.
        Los datos de matrix muy probablemente serán sparse, así que para 
        ponerlos en forma de columna será necesario rellenar
        
    Retorno:
    None
    """
    data = itertools.chain.from_iterable(matrix)
    tuples = itertools.izip_longest(*data, fillvalue=fillvalue)
    column_names = ['TL%s_PL%s' % (map_values(real,list_labels), map_values(predicted,list_labels)) for real, predicted in 
                    itertools.product(range(len(matrix)), range(len(matrix)))]
    
    
    with open(path, 'wb') as fichero:
        writer = csv.writer(fichero, delimiter=',')
        
        writer.writerow(column_names)        
        for tupla in tuples:
            writer.writerow(tupla)
            
def map_values (value,list_labels):
    dictionary = {}
    for ind in range(len(list_labels)):
        dictionary[ind] = list_labels[ind]     
    return dictionary[value]

# TRANSFORM CONFUSION MATRIX OF SIZE n IN THE BINARY CONFUSION MATRIX CORRESPONDING TO CLASS i
def cm_class(cm,index_cl):
    cm_cl=[[0,0],[0,0]]
    cm_cl[0][0]=cm[index_cl][index_cl]
    cm_cl[0][1]=sum(cm[index_cl])-cm_cl[0][0]
    cm_cl[1][0]=sum(cm.transpose()[index_cl])-cm_cl[0][0]
    cm_cl[1][1]=sum(sum(cm))-sum(cm[index_cl])-sum(cm.transpose()[index_cl])+cm_cl[0][0]
    return(np.array(cm_cl))

# COMPUTE PRECISION,RECALL,SPECIFICITY AND F1-SCORE FOR EACH CLASS
def metrics_cm(cm):
    precisions_vector=[]
    recalls_vector=[]
    specificities_vector=[]
    f1_scores_vector=[]
    MCCs_vector=[]
    matrix_confusion_tables_by_class = []
    for i in range(len(cm)):
        
        cm_cl=cm_class(cm,i)
        TP_cl=float(cm_cl[0][0])
        FN_cl=float(cm_cl[0][1])
        FP_cl=float(cm_cl[1][0])
        TN_cl=float(cm_cl[1][1])
        matrix_confusion_tables_by_class.append(cm_cl)
        
        try:
            precision_class= TP_cl/(TP_cl+FP_cl)
            if(pd.isnull(precision_class)):
                raise Exception()
        except:
            precision_class = "0.0"
        precisions_vector.append(precision_class)
        
        try:
            recall_class= TP_cl/(TP_cl+FN_cl)
            if(pd.isnull(recall_class)):
                raise Exception()
        except:
            recall_class= "0.0"
        recalls_vector.append(recall_class)
        
        try:

            specificity_class=TN_cl/(TN_cl+FP_cl)

            if(pd.isnull(specificity_class)):
                raise Exception()
        except:
            specificity_class = "0.0"
        specificities_vector.append(specificity_class)
        
        try:
            f1_score_class=(precision_class*recall_class)/(precision_class+recall_class)*2
            if(pd.isnull(f1_score_class)):
                raise Exception()
        except:
            f1_score_class = "0.0"
        f1_scores_vector.append(f1_score_class)
        
        try:            
            MCC_class=(TP_cl*TN_cl-FP_cl*FN_cl)/(np.sqrt((TP_cl+FP_cl)*(TP_cl+FN_cl)*(TN_cl+FP_cl)*(TN_cl+FN_cl)))
            if(pd.isnull(MCC_class)):
                raise Exception()
        except:
            MCC_class = "1.0"
        MCCs_vector.append(MCC_class)
        
    return(precisions_vector,recalls_vector,specificities_vector,f1_scores_vector,MCCs_vector,matrix_confusion_tables_by_class)
    
def compute_macro_avg_values_of_metrics(list_names_metrics,precisions_vector, recalls_vector,specificity_vector,f1_score_vector,mcc_vector,target_recodified_values,report_dict,model_name,number_decimals = 4):
    
    '''General parameters'''
    number_targets = len(target_recodified_values)    
    
    '''Computation of macro avg'''
    precision_macro_avg = 0.0
    recall_macro_avg = 0.0
    specificity_macro_avg = 0.0
    f1_score_macro_avg = 0.0
    mcc_macro_avg = 0.0    
    
    report_dict[model_name]['metrics'] = {}
    for ind in range(number_targets):
        temp_dict = {'target_'+str(target_recodified_values[ind]):{list_names_metrics[0]:str(precisions_vector[ind]),
                                                                                          list_names_metrics[1]:str(recalls_vector[ind]),
                                                                                          list_names_metrics[2]:str(specificity_vector[ind]),
                                                                                          list_names_metrics[3]:str(f1_score_vector[ind]),
                                                                                          list_names_metrics[4]:str(mcc_vector[ind])}}
        report_dict[model_name]['metrics'].update(temp_dict)
        
        precision_macro_avg+=float(precisions_vector[ind])
            
        recall_macro_avg+=float(recalls_vector[ind])

        specificity_macro_avg+=float(specificity_vector[ind])

        f1_score_macro_avg+=float(f1_score_vector[ind])

        mcc_macro_avg+=float(mcc_vector[ind])

    
    #Calculate macro avg of each metric
    try:
        precision_macro_avg = round(precision_macro_avg/number_targets,number_decimals)
    except:
        precision_macro_avg = "0.0"
        
    try:
        recall_macro_avg = round(recall_macro_avg/number_targets,number_decimals)
    except:
        recall_macro_avg= "0.0"
    
    try:
        specificity_macro_avg = round(specificity_macro_avg/number_targets,number_decimals)
    except:
        specificity_macro_avg ="0.0"
        
    try:
        f1_score_macro_avg = round(f1_score_macro_avg/number_targets,number_decimals)
    except:
        f1_score_macro_avg = "0.0"
        
    try:
        mcc_macro_avg=round(mcc_macro_avg/number_targets,number_decimals)
    except:
        mcc_macro_avg = "0.0"
        
    report_dict[model_name]['metrics_macro_avg'] ={list_names_metrics[0]:str(precision_macro_avg),
                                                   list_names_metrics[1]:str(recall_macro_avg),
                                                   list_names_metrics[2]:str(specificity_macro_avg),
                                                   list_names_metrics[3]:str(f1_score_macro_avg),
                                                   list_names_metrics[4]:str(mcc_macro_avg)}
    
    return (precision_macro_avg,recall_macro_avg,specificity_macro_avg,f1_score_macro_avg,mcc_macro_avg,report_dict)

def compute_micro_avg_values_of_metrics(list_names_metrics,confusion_tables_by_class,target_recodified_values,report_dict,model_name,number_decimals = 4):
    
    '''General parameters'''
    number_targets = len(target_recodified_values)    
    
    '''Computation of micro avg'''
    #confusion_tables_by_class[0] = [[TP,FN],[FP,TN]]
    '''sensitivity (recall) micro avg'''
    recall_micro_avg = 0.0
    num_recall = 0.0
    den_recall = 0.0
    
    '''specificity micro avg'''
    specificity_micro_avg = 0.0
    num_specificity = 0.0
    den_specificity = 0.0
    
    '''precision micro avg'''
    precision_micro_avg = 0.0
    num_precision = 0.0
    den_precision = 0.0
    
    '''f1-score micro avg'''
    f1_score_micro_avg = 0.0
    
    '''mcc micro avg'''
    total_TP = 0.0
    total_FN = 0.0
    total_FP = 0.0
    total_TN = 0.0
    
    
    for ind in range(number_targets):
        #recall
        num_recall+=float(confusion_tables_by_class[ind][0,0]) #TP each class
        den_recall+=float(confusion_tables_by_class[ind][0,0]) + float(confusion_tables_by_class[ind][0,1]) #TP + FN each class
        
        #specificity
        num_specificity+=float(confusion_tables_by_class[ind][1,1]) #TN each class        
        den_specificity+=float(confusion_tables_by_class[ind][1,1]) + float(confusion_tables_by_class[ind][1,0]) #TN + FP each class
        
        #precision
        num_precision+=float(confusion_tables_by_class[ind][0,0]) #TP each class
        den_precision+=float(confusion_tables_by_class[ind][0,0]) + float(confusion_tables_by_class[ind][1,0]) #TP + FP each class
        
        #mcc
        total_TP += float(confusion_tables_by_class[ind][0,0])
        total_FN += float(confusion_tables_by_class[ind][0,1])
        total_FP += float(confusion_tables_by_class[ind][1,0])
        total_TN += float(confusion_tables_by_class[ind][1,1])
        
    
    recall_micro_avg = round(num_recall/den_recall,number_decimals)
    specificity_micro_avg = round(num_specificity/den_specificity,number_decimals)
    precision_micro_avg = round(num_precision/den_precision,number_decimals)
    
    f1_score_micro_avg = 2*( (precision_micro_avg*recall_micro_avg)/ (precision_micro_avg + recall_micro_avg) )
    f1_score_micro_avg = round(f1_score_micro_avg,number_decimals)
    
    mcc_micro_avg = ( (total_TP*total_TN) - (total_FP*total_FN) ) / np.sqrt( (total_TP+total_FP) * (total_TP+total_FN) * (total_TN+total_FP) * (total_TN+total_FN) )
    mcc_micro_avg = round(mcc_micro_avg,number_decimals)
    
    
    report_dict[model_name]['metrics_micro_avg'] ={list_names_metrics[0]:str(precision_micro_avg),
                                                   list_names_metrics[1]:str(recall_micro_avg),
                                                   list_names_metrics[2]:str(specificity_micro_avg),
                                                   list_names_metrics[3]:str(f1_score_micro_avg),
                                                   list_names_metrics[4]:str(mcc_micro_avg)}
    
    return (precision_micro_avg,recall_micro_avg,specificity_micro_avg,f1_score_micro_avg,mcc_micro_avg,report_dict)

def get_confusion_matrix(labels_true,model_predictions,labels):
    '''It allows to generate the confusion matrix
    Parameters
    :param list labels_true: original values for the labels
    :param list model_predictions: predicted vales for the labels
    :param list labels: list of possible labels
    
    :return confusion matrix
    :rtype numpy array
    '''
    
    matriz_confusion=confusion_matrix(labels_true,model_predictions,labels)
    return matriz_confusion


def save_confusion_matrix(confusion_matrix,customized_labels,customized_name,output_format):
    '''
    It allows to obtain a graphical representation of the numpay array with the confusion matrix
    in the specified format
    
    Params:
    :param numpy_array cofusion_matrix: confusion matrix calculated previously
    :param customized_labels: list of labels
    :param customized_name: path to the directory where to store the file
    :param output_format: output format for the image with the confusion matrix
    
    :return None
    '''
    
    x_axis=list(range(len(customized_labels)))
    y_axis=list(range(len(customized_labels)))
    plt.figure()
    plt.imshow(confusion_matrix, cmap=plt.cm.Blues, interpolation='nearest')#cmap=plt.cm.GnBu
    #plt.xticks([0,1,2], etiquetas, rotation='vertical')
    #plt.yticks([0,1,2], etiquetas)
    plt.xticks(x_axis, customized_labels, rotation='vertical')
    plt.yticks(y_axis, customized_labels)
    #fmt = '.2f' 
    #thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        #plt.text(j, i, format(confusion_matrix[i, j], fmt), horizontalalignment="center", color="white" if confusion_matrix[i, j] > thresh else "black")
        plt.text(j, i, confusion_matrix[i, j], horizontalalignment="center", color="black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.colorbar()
    plt.savefig(customized_name + '.' + output_format)


def compute_learning_curve(estimator, X, y, score_method, cv_it, n_jobs=1, train_sizes=np.linspace(.1,1.0,10)):
    '''It allows to generate the learning curve
    Parameters
    :param skleanr_model estimator: original values for the labels
    :param pandas_dataframe X: dataframe with values for the features excluding target feature
    :param pandas_dataframe y:dataframe with values only for target feature
    :param score_method: scoring method used
    :param iterator cv_it: iterator used
    
    :return train_sizes
    :return train_scores
    :return test_scores
    :rtype numpy array
    '''

    score_method = score_method[1]
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv_it, n_jobs=n_jobs, train_sizes=train_sizes,scoring=score_method)
    
    return(train_sizes, train_scores, test_scores)
    

def save_learning_curve(train_sizes, train_scores, test_scores, title, name_of_the_file,output_format,ylim=None):
    '''
    It allows to obtain a graphical representation of the numpay array with the learning curve
    in the specified format
    
    Params:
    
    '''
    fig = plt.figure()
    #plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Training sample')
    plt.ylabel('Success rate')
    #print('Computing scores')
    train_scores_mean=np.mean(train_scores,axis=1)
    train_scores_std=np.std(train_scores,axis=1)
    test_scores_mean=np.mean(test_scores,axis=1)
    test_scores_std=np.std(test_scores,axis=1)
    plt.grid
    #print('Plot')
    plt.fill_between(train_sizes,train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,color='r')
    plt.fill_between(train_sizes,test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,color='g')
    plt.plot(train_sizes, train_scores_mean,'o-', color='r', label='succes rate in-sample')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='cross-validation rate')
    plt.legend(loc='best')   
    fig.savefig(name_of_the_file,format=output_format)
    
    return(train_scores_mean,test_scores_mean)

'''Begin of Mutual Information functions'''

def column_categorization(data,column):
    data['aux'] = data[column].apply(lambda x: 'nan' if pd.isnull(x) else x)
    data[column] = data['aux']
    data = data.drop('aux',axis=1)
    
    column_data=data[column]
    data_in_column = list(set(column_data.values)) 
    dictionary={}
    dictionary.fromkeys(data_in_column)
    values=list(range(len(data_in_column))) 
    i=0
    for key in data_in_column:
        dictionary[key]=values[i]
        i=i+1
    column_data=column_data.apply(lambda x: dictionary[x])
    return(column_data)


def get_variables_inside_percentil(report,percent,list_important_features_for_the_user):
    lista_variables_scores = []
    for element in report:            
        current_feature = element[0]        
        score = element[1]       
        
        if((score >= percent) or (current_feature in list_important_features_for_the_user)):            
            lista_variables_scores.append([current_feature,score])
    return lista_variables_scores


def get_scores_and_percentil_threshold_using_mutual_information_function(df_data,useless_features,target,input_percentil):
    '''
    It allows to obtain the scores of each variable facing the objective variable and the percentile

    Parameters:
    :param pandas_dataframe df_data: dataframe with all data of the event
    :param str target: objective feature to contrast other features
    :param float umbral: percentil of variable taht will be used    
    
    :return: list of variables and scores
    :rtype: list<list<str,float>>
    :return: percetil value with the minimun score to be between the variables in the desired pecentage
    :rtype: float   
    '''
                
    features = list(df_data.columns) #todas las variables que no son vacias o constantes
    
    for feat in useless_features:
        features.remove(feat)
    
    scores=[]
    report=[]
    for i in range(len(features)):
        variable=features[i]

        ''' Facing feature with target feature'''
        true=list(df_data[target])
        pred=list(df_data[variable])
        score=normalized_mutual_info_score(true,pred)
        score= float(round(score,6))        

        report.append([variable,score])
        scores.append(score)  
        
    percentile = np.percentile(np.array(scores),input_percentil)

    return report,percentile



def apply_mutual_information_function_to_current_features_using_percentil(event,target_variable,percentil_specified_by_user,path_to_directory_current_target,df_catalogued_data,list_relevant_variables_user,useless_features,log_file,enco):
    '''
    Permite obtener el porcentaje de influencia de cada una de las features en otra feature objetivo
    para un determinado evento. Dado que los datasets pueden tener un elevado numero de registros, se
    puede establecer el maximo numero limitando el numero de observaciones que se usaran en el calculo.

    Parameters:
    :param str evento: es el evento para el cual se quiere calcular la funcion de informacion mutua con respecto al parametro target.
    :param str target_variable: es la variable objetivo contra la que se contrasta cada una de las features
    :param list lista_rutas_ficheros_csv: lista con las rutas completas a los ficheros csv con los datos
    :param dict diccionario_variables_relevantes: diccionario con las variables que se usaran a lo largo del programa
    :param int numero_maximo_registros: numero maximo de registros que se examinaran en la variable de informacion mutua
    :param str ruta_directorio_target_actual: ruta al directorio de salida de los resutados de la funcion de informacion mutua
    
    :return: lista con  de listas de la forma [feature, porcentaje_influencia] | o una lista vacia
    :rtype: list<list<str,float>> | empty list
    '''
        
    returned_list_variables_over_percentil=[]
                
    '''Giving name of event to mif report'''
    filename_generated_reports=event         

    '''Getting scores for each possible feature for train models applying MIF'''    
    list_possible_features_mif_scores,percent = get_scores_and_percentil_threshold_using_mutual_information_function(df_catalogued_data,useless_features,target_variable,percentil_specified_by_user)
    
    '''Generating file with features and scores'''
    repg.create_mif_report(path_to_directory_current_target,filename_generated_reports,list_possible_features_mif_scores,percent,enco)
    
    '''Returning features inside percentil'''
    returned_list_variables_over_percentil= get_variables_inside_percentil(list_possible_features_mif_scores,percent,list_relevant_variables_user)    

    return returned_list_variables_over_percentil

