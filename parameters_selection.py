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

import auxiliary_functions as auxf
import os
import shutil
import numpy as np


from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV

import supervised as sup
import reports_generation as repg

import sys
sys.path.insert(0, './RHOASo')

import seleccion_nueva as rhoaso

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


def move_file_to_folder(file_name, base_path_folder, rute_folder_destination):
  rute_file=os.path.join(base_path_folder,file_name)
  if os.path.exists(rute_folder_destination+'/'+file_name):
      os.remove(rute_folder_destination+'/'+file_name)
  shutil.move(rute_file,rute_folder_destination)
    
def transform_dic_to_matrix(ac, max_pars):
    matrix = np.zeros([max(max_pars)+1] * 2)
    for point in ac:
        matrix[point] = ac[point]
    return matrix

def plot_matrix(matrix,max_pars,name,path_to_supervised_models_directory,output_format):
    '''
    imput type = 2D array (matrix of accuracies)
                 size of the matrix
                 name of the model respecto to which the accuracies have been computed
                 folder in which the figure will be saved
    return = name of the figure
             figure
             outputs of the function stable_accurate_surface_nodes
    '''
    xx=range(max_pars)
    yy=range(max_pars)
    fig=plt.figure()
    ax=Axes3D(fig)
    xx, yy=np.meshgrid(xx,yy)
    
    ax.plot_surface(xx,yy,np.transpose(matrix))
    fname='figure_for_'+name+'.' + output_format
    plt.savefig(os.path.join(path_to_supervised_models_directory,fname))
    
    return os.path.join(path_to_supervised_models_directory,fname)
    
def plot_graphic(parameter_variation,accuracy_vector,name,path_to_supervised_models_directory,output_format):
    '''
    imput type = 2D array (matrix of accuracies)
                 size of the matrix
                 name of the model respecto to which the accuracies have been computed
                 folder in which the figure will be saved
    return = name of the figure
             figure
             outputs of the function stable_accurate_surface_nodes
    '''
    plt.clf()
    plt.plot(parameter_variation,accuracy_vector)
    plt.xlabel('Value of parameter')
    plt.ylabel('Accuracy')
    fname='figure_for_'+name+'.' + output_format
    plt.savefig(os.path.join(path_to_supervised_models_directory,fname))

    return(os.path.join(path_to_supervised_models_directory,fname))


def get_depth(model_name,dictionary_supervised_models,max_depth,features,target,df_catalogued_data,ruta_directorio_modelo_supervisado,formato):

    '''Computes dept or number of estimtors for the model'''
    
    default_iter = 30
    if default_iter >= max_depth:
        default_iter = max_depth - 1
    X = df_catalogued_data[features]
    Y = df_catalogued_data[target]
    mi_modelo = sup.initialize_model(model_name,[1],dictionary_supervised_models)
    parameter_to_compute = ''
    if(model_name == dictionary_supervised_models[1]): #Tree
        parameter_to_compute = 'max_depth'
    elif(model_name == dictionary_supervised_models[2]): #Ada
        parameter_to_compute = 'n_estimators'
        
    rand = RandomizedSearchCV(mi_modelo, {parameter_to_compute: range(1, max_depth)}, n_iter=default_iter, cv=10).fit(X, Y)
    depth = rand.best_params_[parameter_to_compute]        
    depth_vector = rand.cv_results_['param_' + parameter_to_compute].tolist()
    scores_vector = rand.cv_results_['mean_test_score'].tolist()
    ruta_imagen = plot_graphic(depth_vector,scores_vector,model_name,ruta_directorio_modelo_supervisado,formato)
    return (depth,ruta_imagen)



def select_optimal_parameters_current_model(model_name,diccionario_modelos_sup,diccionario_modelos_no_sup,df_parameter_selection,features,target,fichero_log,ruta_directorio_modelo_supervisado,report_dict,formato,enco,max_pars):
    '''
    The function allows to create and train a model which names has been specified

    Parameters:
    :param str model_name: name of the model to be trained
    :param pandas-dataframe train_data: Data to train de model. It includes the target column
    :param pandas-dataframe test_data: Data to test de model. It includes the target column
    :param list features: List with the relevant features to train the model
    :param str target: Target feature
    :param list params_array: Array with the specific parameters of the model to be trained
    
    :return: specified model trained
    :rtype: sklearn-model
    '''
    
    max_depth=max_pars[0]    
    selected_parameters = []
    max_pars_ensemble = {'max_depth':max_depth,'n_estimators':max_depth}
    max_pars_mlp = {'hidden_layer_sizes':(max_depth,max_depth)}
    
   
    if(model_name == diccionario_modelos_sup[1]):#Tree
        try:
            depth,ruta_imagen = get_depth(model_name,diccionario_modelos_sup,max_depth,features,target,df_parameter_selection,ruta_directorio_modelo_supervisado,'png')
            selected_parameters = [int(depth)]        
            report_dict[model_name]['parameters_plot'] = "'" + ruta_imagen + "'"        
        except Exception as e:
            print e.message
            depth = 0
            selected_parameters = [depth]
            report_dict[model_name]['parameters_plot'] = "''"
                        
        repg.register_log(fichero_log,">>>> Selected parameters for " + model_name + ":\n\t->" \
                           + "max-depth " + str(depth) + "\n",'',enco)   
        report_dict = repg.update_model_parameters(report_dict,model_name,"Max-depth",str(depth))
        
    elif(model_name == diccionario_modelos_sup[2]):#Ada
        try:
            depth,ruta_imagen = get_depth(model_name,diccionario_modelos_sup,max_depth,features,target,df_parameter_selection,ruta_directorio_modelo_supervisado,'png')
            selected_parameters = [int(depth)]        
            report_dict[model_name]['parameters_plot'] = "'" + ruta_imagen + "'"            
        except Exception as e:
            print e.message
            depth = 0
            selected_parameters = [depth]
            report_dict[model_name]['parameters_plot'] = "''"
        
        repg.register_log(fichero_log,">>>> Selected parameters for " + model_name + ":\n\t->" \
                           + "estimators-number " + str(depth) + "\n",'',enco)      
        report_dict = repg.update_model_parameters(report_dict,model_name,"estimators-number",str(depth))
                                                   
    elif(model_name == diccionario_modelos_sup[3]): #Boosting
        try:
            name='accuracy_matrix_'+model_name            
            parameters, accuracy, ac, num_evals= rhoaso.selection_n(GradientBoostingClassifier, df_parameter_selection[features], df_parameter_selection[target],max_pars_ensemble,1)# num_neurons=None, random=False,n_layers=None)            
            depth = parameters['max_depth']
            estimators = parameters['n_estimators']
            acc=transform_dic_to_matrix(ac,max_pars)
            resultados = plot_matrix(acc,max(max_pars)+1,name,ruta_directorio_modelo_supervisado,formato)
            selected_parameters = [int(depth),int(estimators)]
            report_dict[model_name]['parameters_plot'] = "'" + resultados + "'"            
        except Exception as e:
            print e.message
            depth = 0
            estimators = 0
            accuracy = 0
            selected_parameters = [depth,estimators]
            report_dict[model_name]['parameters_plot'] = "''"            
            
        repg.register_log(fichero_log,">>>> Selected parameters for " + model_name + ":\n" \
                           + "\t-> acc " + str(accuracy) +"\n"\
                           + "\t-> depth " + str(depth) + "\n"\
                           + "\t-> estimators " + str(estimators) +"\n" ,'',enco)
       
        report_dict = repg.update_model_parameters(report_dict,model_name,"Acc",str(accuracy))
        report_dict = repg.update_model_parameters(report_dict,model_name,"Depth",str(depth))
        report_dict = repg.update_model_parameters(report_dict,model_name,"Estimators",str(estimators))

    elif(model_name == diccionario_modelos_sup[4]):#RandomForest
        try:
            name='accuracy_matrix_'+model_name
            parameters,accuracy, ac, num_evals=rhoaso.selection_n(RandomForestClassifier, df_parameter_selection[features], df_parameter_selection[target],max_pars_ensemble,1)
            depth = parameters['max_depth']
            estimators = parameters['n_estimators']
            acc=transform_dic_to_matrix(ac,max_pars)
            resultados = plot_matrix(acc,max(max_pars)+1,name,ruta_directorio_modelo_supervisado,formato)
            selected_parameters = [int(depth),int(estimators)]
            report_dict[model_name]['parameters_plot'] = "'" + resultados + "'"
        except Exception as e:
            print e.message
            depth = 0
            estimators = 0
            accuracy = 0
            report_dict[model_name]['parameters_plot'] = "''"
            
        repg.register_log(fichero_log,">>>> Selected parameters for " + model_name + ":\n" \
                           + "\t-> acc " + str(accuracy) +"\n"\
                           + "\t-> depth " + str(depth) + "\n"\
                           + "\t-> estimators " + str(estimators) +"\n" ,'',enco)
        
        report_dict = repg.update_model_parameters(report_dict,model_name,"Acc",str(accuracy))
        report_dict = repg.update_model_parameters(report_dict,model_name,"Depth",str(depth))
        report_dict = repg.update_model_parameters(report_dict,model_name,"Estimators",str(estimators))
    
    elif(model_name == diccionario_modelos_sup[5]):#MLPerceptron
        algorithm = 'tanh'
        try:
            name='accuracy_matrix_'+model_name
            parameters,accuracy, ac, num_evals=rhoaso.selection_n(MLPClassifier, df_parameter_selection[features], df_parameter_selection[target],max_pars_mlp,1)#, num_neurons=None, random=False,n_layers=None)
            len_layer1 = parameters['hidden_layer_sizes'][0]
            len_layer2 = parameters['hidden_layer_sizes'][1]
            acc=transform_dic_to_matrix(ac,max_pars)
            resultados = plot_matrix(acc,max(max_pars)+1,name,ruta_directorio_modelo_supervisado,formato)
            selected_parameters = [[int(len_layer1),int(len_layer2)],algorithm]
            report_dict[model_name]['parameters_plot'] = "'" + resultados + "'"
        except Exception as e:
            print e
            len_layer1 = 0
            len_layer2 = 0
            accuracy = 0
            selected_parameters = [[int(len_layer1),int(len_layer2)],algorithm]
            
        repg.register_log(fichero_log,">>>> Selected parameters for " + model_name + ":\n" \
                           + "\t-> acc " + str(accuracy) +"\n"\
                           + "\t-> len_layer_1 " + str(len_layer1) + "\n"\
                           + "\t-> len_layer_2 " + str(len_layer2) +"\n" ,'',enco)
                
        report_dict = repg.update_model_parameters(report_dict,model_name,"Acc",str(accuracy))
        report_dict = repg.update_model_parameters(report_dict,model_name,"Length layer 1",str(len_layer1))
        report_dict = repg.update_model_parameters(report_dict,model_name,"Length layer 2",str(len_layer2))

    elif(model_name == diccionario_modelos_no_sup[1]): #Kmeans
        numero_grupos = auxf.get_number_different_values_objective_target(df_parameter_selection[target].values)
        numero_inicializaciones = 10
        selected_parameters = [numero_grupos,numero_inicializaciones]
        repg.register_log(fichero_log,">>>> Selected parameters for " + model_name + ":\n" \
                           + "\t-> numero grupos " + str(numero_grupos) +"\n"\
                           + "\t-> numero_inicializaciones " + str(numero_inicializaciones) +"\n" ,'',enco)
        
        report_dict = repg.update_model_parameters(report_dict,model_name,"Number groups",str(numero_grupos))
        report_dict = repg.update_model_parameters(report_dict,model_name,"Number initializations",str(numero_inicializaciones))
                
    return selected_parameters,report_dict
