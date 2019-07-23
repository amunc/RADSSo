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

import codecs
import copy
import datetime
import json
import numpy as np
import os
import pandas as pd
import pickle
import shutil
import simplejson
import time

import data_request as datr

import reports_generation as repg

def print_initial_license_message():
    print ''' 
    
     RIASC Automated Decision Support Software (RADSSo) generates the best supervised/unsupervised model,
    in an automated process, based on some input features and one target feature, to solve a multi-CASH problem.

    Copyright (C) 2018  by RIASC Universidad de Leon (Angel Luis Munoz Castaneda, Mario Fernandez Rodriguez, Noemi De Castro Garcia y Miguel Carriegos Vieira)
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
	
	 You can find more information about the project at https://github.com/amunc/RADSSo\n'''


def decodify_using_enconding(text,enco):
    '''
    This function allows decode text using a concrete enconding

    Parameters:    
    :param str text: String text to decode    
    :return: decoded text
    :rtype: str
    '''
    text_dec = ''
    try:
        text_dec=text.decode(enco)
    except:
        text_dec=text
    return text_dec


def codify_using_enconding(text,enco):
    '''
    This function allows encode text using a concrete enconding

    Parameters:    
    :param str text: String text to encode    
    :return: encoded text
    :rtype: str
    '''
    text_enco = ''
    try:
        text_enco = text.encode(enco)
    except:
        text_enco = text
    return text_enco


def load_json_file(fullpath_to_filename,enco):
    '''
    This function allows to retrieve a dictionary in json format stored in
    a json file

    Parameters:    
    :param str fullpath_to_filename: String with the full path to the location of the
    json file
    :exception IOError is json file is not found
    :return: dictionary, json parameters in dictionary form
    :rtype: dict
    '''
    
    if(os.path.lexists(fullpath_to_filename)):
        read_file = file(fullpath_to_filename,"r").read()        
        dictionary = simplejson.loads(read_file)        
    else:
        raise IOError('File ' + fullpath_to_filename.encode(enco) + ' does not exist')
    return dictionary

	
def check_input_target(vector_with_input_arguments):
    '''
    This function allows to check if the number of invocation parameters introduced by command line is correct.

    Parameters:    
    :param list vector_with_input_arguments: List with the arguments introduced by command line
    
    :exception Exception if target is not correct or if invocation is not correct
    
    :return: target,Name of the target
    :rtype: str
    '''
    target = ''
    if(len(vector_with_input_arguments) > 1):
        target = vector_with_input_arguments[1]
    else:
        print 'main_train.py invocation is not correct: main_train.py target_feature [semi-automated] [build_files] '
        raise Exception('main_train.py invocation is not correct: main_train.py target_feature [semi-automated] [build_files] ')
    return target
	

def compute_matrix_weights(dimensions):
    '''
    This function allows compute the matrix of weights by using
    an uniform distribution for each weight

    Parameters:    
    :param dimensions: number of values in the target feature
    
    
    :return: matrix of weights in list form
    :rtype: list
    '''
    value_to_fullfil = float(1.0/dimensions)
    np_matrix = np.full((dimensions,dimensions),value_to_fullfil)
    return np_matrix.tolist()

def check_matrix_weights(matrix_of_weights,dimensions):
    '''
    This function allows to check if the matrix of weights in the conf.ini
    is correct in shape and in values.

    Parameters:    
    :param list matrix_of_weights: Matrix with the weights introduced by the user. Must be a 3x3 matrix which values of rows
    sum a total amount of 1.0
    
    :exception ValueError if shape ot he matrix or distribution of values is not correct
    
    :return: checked_matrix_of_weights,matrix with weights
    :rtype: numpy_array
    '''
    matrix_of_weights = np.array(matrix_of_weights)    
    checked_matrix_of_weights =[]
    if(matrix_of_weights.shape[0]== dimensions):
        correct_number_elements = dimensions*dimensions
        number_elements = 0
        for indice in range(0,dimensions):
            number_elements+=len(matrix_of_weights[indice])        
        if(number_elements == correct_number_elements):          
            correct_distribution = True
            contador = 0
            while(contador <= dimensions-1 and correct_distribution):
                row_distribution = round(np.sum(matrix_of_weights[contador]),0)                
                if(row_distribution != 1.0):
                    correct_distribution = False
                contador+=1
            if(correct_distribution):
                checked_matrix_of_weights = matrix_of_weights.tolist()
            else:
                raise ValueError('Check conf.ini - matrix_of_weights_fp_fn parameter:\n The distribution in some row of the matrix of weights is not correct (the addition of elements in the rows must be 1.0).')            
        else:
            raise ValueError("Check conf.ini - matrix_of_weights_fp_fn parameter:\n The shape of the weighted matrix must be num_target_values(" + str(dimensions) + ") rows by num_target_values(" + str(dimensions) + ") columns.Number of columns incorrect")
    else:
        raise ValueError("Check conf.ini - matrix_of_weights_fp_fn parameter:\n The shape of the weighted matrix must be num_target_values(" + str(dimensions) + ")rows by num_target_values(" + str(dimensions) + ") columns. Number of rows incorrect")
    
    return checked_matrix_of_weights
	
	
def check_train_test_division_percent(percent):
    '''
    This function allows to check if the division percentage for the train
    and test datasets is correct

    Parameters:    
    :param float percent: pecentage of division introduced by the user
    
    :exception ValueError if the percentage is out of range(0.6,0.9)
    
    :return: checked_percent, percent of division
    :rtype: float
    '''
    if(percent < 0.6 or percent > 0.9):
        raise ValueError("Check conf.ini - Parameter with the percentage of division in train and test datasets is not in the range[0.6,0.9]")
    else:
        return percent


def save_dictionary_to_disk_pickle_format(dictionary_to_store,handler):
    '''
    This function allows to store the dict structure that receives as parameter
    in the specified path with the specified name in pickle format

    Parameters:
    :param dict dictionary_to_store: Dictionary structure to store in disk.
    :param str ruta_diccionario: String with the full path to the location of the
    dictionary where it will be stored.
    :param str nombre_diccionario: String with the name that will be given to the
    dictionary
    
    :return: None    
    '''
    
    pickle_protocol=pickle.HIGHEST_PROTOCOL
    pickle.dump(dictionary_to_store, handler, protocol=pickle_protocol)
    handler.close()


def open_dictionary_pickle_format_for_reading(full_path_to_stored_dictionary):
    '''
    This function allows to store the dict structure that receives as parameter
    in the specified path with the specified name

    Parameters:    
    :param str full_path_to_stored_dictionary: String with the full path to the location of the
    dictionary that will be loaded.
    
    :return: retrieved_dictionary,Dictionary previously stored
    :rtype: dict<python_hashable_type:python_type>
    '''
    handler = open(full_path_to_stored_dictionary, 'rb')
    retrieved_dictionary = pickle.load(handler)
    return retrieved_dictionary,handler


def get_dictionary_in_pickle_format_handler(fullpath_to_dictionary):
    '''
    This function allows to obtain dictionary handler from the
    the specified path with the specified name in order to modify it

    Parameters:    
    :param str fullpath_to_dictionary: String with the full path to the location of the
    dictionary that will be loaded.
    
    :return: handler, the handler of the dictionary previously stored
    :return: handler, handler for the specified file
    :rtype: file handler
    '''
    handler = open(fullpath_to_dictionary, 'wb')
    return handler


def retrieve_dictionary_and_handler(fullpath_to_dictionary):
    '''
    This function allows to retrieve the dictionary stored in disk

    Parameters:    
    :param str ruta_diccionario: String with the full path to the location of the
    dictionary that will be loaded.
    
    :return: Dictionary previously stored
    :rtype: dict<python_hashable_type:python_type>
    '''
    returned_dictionary = {}
    handler=None
    if(os.path.exists(fullpath_to_dictionary)):
        returned_dictionary,handler = open_dictionary_pickle_format_for_reading(fullpath_to_dictionary)
    else: #creation of dictionary
        pickle_protocol=pickle.HIGHEST_PROTOCOL
        handler =  open(fullpath_to_dictionary, 'wb')
        pickle.dump(returned_dictionary, handler, protocol=pickle_protocol)
        handler.close()        
        returned_dictionary,handler = open_dictionary_pickle_format_for_reading(fullpath_to_dictionary)
    return returned_dictionary,handler


def restore_dictionary_to_previous_version(fullpath_to_dictionary,previous_version_of_dict):
    '''
    This function allows to restore the previous safe version of the dictionary when
    the last modification was being saved and it was interrupted

    Parameters:    
    :param str fullpath_to_dictionary: String with the full path to the location of the
    dictionary that will be loaded.
    :param previous_version_of_dict: State of the before the last changes in the dictionary
    
    :return: Dictionary previously stored
    :rtype: dict<python_hashable_type:python_type>
    '''
    pickle_protocol=pickle.HIGHEST_PROTOCOL
    handler =  open(fullpath_to_dictionary, 'wb')
    pickle.dump(previous_version_of_dict, handler, protocol=pickle_protocol)
    handler.close()    


def check_number_of_values_objective_target(number_different_values):
    '''
    This function allows to check the number if targets to determine if the problem
    has a solution applying machine learning.

    Parameters:    
    :param int number_different_values: number of targets found
    
    :return: True | False
    :rtype: boolean
    '''
    classification_problem_already_solved = False    
    if(number_different_values < 2):
        classification_problem_already_solved = True
    return classification_problem_already_solved

def load_dictionary_of_variables(fullpath_to_json_file,json_key,enco):
    '''
    It allows to obtain a dictionaray with the variables hardcoded, by this way if the name of any of the variables is modified, 
    it would be only necessary to change the value for that key in the dictionary.
     
    :param str fullpath_to_json_file: full path to the json where the dictionray with the variables and the name that receive in the current
    execution is stored
    :exception IOError: If the json file is not found
    :exception KeyError: If the dictionray with the variables is not found

    :return: dictionary with features with known recodification and their actual name
    :rtype: dict<str:str>
    '''
    
    json_variables_known_recodification = ''
    json_variables_known_recodification = load_json_file(fullpath_to_json_file,enco)
        
    return json_variables_known_recodification[json_key]


def get_number_different_values_objective_target(list_with_all_values_of_objective_target):
    '''
    This function allows to obtain the number of targets for the current event.

    Parameters:    
    :param list list_with_all_values_of_objective_target: Colum with all the values for the objective feature
    
    :return: Number of targets
    :rtype: int
    '''
    return len(list(set(list_with_all_values_of_objective_target)))


def compare_lists(first_list,second_list):
    '''
    This function allows to compare two lists of elements and for each element that is located at the same position in both lists,
    in increases the counter variable number_of_coincident_elements. Both lists must have the same length

    Parameters:
    :param list first_list: First list with the elements to compare.
    :param list second_list: Second list with the elements to compare.  
    
    :return: number of elements that are the same in the same location at both lists
    :rtype: int
    '''
   
    number_of_coincident_elements=0
    for i in range(len(first_list)):
        if (first_list[i] == second_list[i]):            
            number_of_coincident_elements=number_of_coincident_elements+1
    return(number_of_coincident_elements)


def get_complementary_list(full_list,sublist):
    '''
    This function allows to obtain the complementary list of a sublist recevied as parameter
    and that is a sublist of the another list received as parameter from which are extracted the
    complementary features

    Parameters:
    :param list full_list: List with all the elements
    :param list sublist: List with some elements of full list (a sublist of full_list)
    
    :return: complementary_list,number of elements that are only in full_list
    :rtype: list'''
   
    complementary_list=list(set(full_list).difference(sublist))
    return(complementary_list)


def get_all_files_in_dir_with_extension(fullpath_to_directory_with_files,maximum_number_files_to_read,files_extension):
    '''
    This function allows to create a list(vector) with all the files in a directory that
    has an specific files_extension
    
    Parameters:
    :param str fullpath_to_directory_with_files: Path to the root directory with the files.
    :param int maximo_numero_ficheros: Number that indicates the maximum number of files that will be loaded.   
    :param files_extension: Extension of the files to look for into the directory

    :return: vector_with_fullpaths_to_input_files,list with strings that refers to the full path of each file to be read orderer by name    
    :rtype: list<str>
    '''
    
    vector_with_fullpaths_to_input_files = []
    for root, directories, available_files in os.walk(fullpath_to_directory_with_files):  
        for filename in available_files:
            if('.'+files_extension in filename):                
                vector_with_fullpaths_to_input_files.append(os.path.join(fullpath_to_directory_with_files,filename))
                
    if (maximum_number_files_to_read !=0 ): # if maximun number of files was specified
        if(len(vector_with_fullpaths_to_input_files) > maximum_number_files_to_read):
            vector_with_fullpaths_to_input_files = vector_with_fullpaths_to_input_files[0:maximum_number_files_to_read]
    return sorted(vector_with_fullpaths_to_input_files)


def checking_events_reading_method(vector_input_arguments,vector_of_fullpaths_to_input_files,fullpath_to_json_with_events_features,event_name,input_files_delimiter,execution_log,time_log,list_of_not_valid_characters,enco):
    '''
    This function allows several things depending on the invocation of main.py:
        main.py target (2 arguments): it reads the events and variables from the path fullpath_to_json_with_events_features (a file created by the user is necesary)
        main.py target build_files(3 arguments): it drives the user to build a file with events and variables that will be used during the process
        main.py target auto (3 arguments): it tries to extract the events automatically from the input files
    
    
    Parameters:
    :param list vector_input_arguments: list with the parameters introduced by command line
    :param list vector_of_fullpaths_to_input_files: list with the path to the files that will be read
    :param str fullpath_to_json_with_events_features: full path to the file with events, relevant features and discarded features
    :param str feature_events_name: feature inside the input files where the name of the events is stored
    :param str target: objective feature
    :param str input_files_delimiter: field delimiter for input files 
    :param str execution_log: path to file log to write 
    :param str time_log: path to file time log to write
    
    :return list lista_eventos: list of events
    :rtype: list<str>
    :return list_list_of_relevant_features_for_the_user: list of relevant features for each event specified by the user
    :rtype: list<list<str>>
    :return list_list_discarded_features_by_the_user: list of discarded variables for each event specified by the user
    :rtype: list<list<str>>    
    '''
    
    substep_init_time = datetime.datetime.fromtimestamp(time.time())
    list_of_events_to_process = []
    list_list_of_relevant_features_for_the_user = []
    list_list_discarded_features_by_the_user = []
    
    '''Checking if automatic mode has been selected'''    
    if(len(vector_input_arguments) > 2 and vector_input_arguments[2] == 'semi-automated'): #Manual reading (building files or using existing ones)
        repg.register_log([execution_log],'>>>>>>Substep 1.3: Recognizing events manually '+ substep_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n\n" \
                       ,'',enco)
        if(len(vector_input_arguments) > 3 and vector_input_arguments[3] == 'build_files'): #Building files option specified
            repg.register_log([execution_log],'>>>>>>Substep 1.3: Building files by user interaction '+ datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n\n" \
                       ,'',enco)          
            diccionario_eventos_variables = datr.get_events_and_variables_from_user_interaction(vector_of_fullpaths_to_input_files,event_name,input_files_delimiter,list_of_not_valid_characters,enco)            
            with codecs.open(fullpath_to_json_with_events_features, 'w',encoding=enco) as outfile:
                json.dump(diccionario_eventos_variables, outfile,ensure_ascii=False)
                                
        '''Reading json with events and variables'''        
        list_of_events_to_process,list_list_of_relevant_features_for_the_user,list_list_discarded_features_by_the_user = datr.get_list_events_features_json_file(fullpath_to_json_with_events_features,list_of_not_valid_characters)
                                
        substep_finish_time = datetime.datetime.fromtimestamp(time.time())
        repg.register_log([time_log],'>>>>>>Substep 1.3 Recognizing events manually elapsed time: '+ str(substep_finish_time - substep_init_time) + "\n",'',enco)
        repg.register_log([execution_log],'>>>>>>Substep 1.3 (manually) ends '+ substep_finish_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',enco)
    else:        
        repg.register_log([execution_log],'>>>>>>Substep 1.3: Recognizing events automatically '+ substep_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n\n" \
                       ,'',enco)
        #list_of_events_to_process = [event_name]
        list_of_events_to_process,list_of_features = datr.recognize_available_events_and_variables_automatically(vector_of_fullpaths_to_input_files,codify_using_enconding(event_name,enco),input_files_delimiter,list_of_not_valid_characters,enco)
        #list_of_features = datr.recognize_available_variables_automatically(vector_of_fullpaths_to_input_files,input_files_delimiter,list_of_not_valid_characters,enco)
    
        list_of_events_to_process=sorted(list_of_events_to_process)
        for event in list_of_events_to_process:
            list_list_of_relevant_features_for_the_user.append([])
            list_list_discarded_features_by_the_user.append([])
        substep_finish_time = datetime.datetime.fromtimestamp(time.time())
        repg.register_log([time_log],'>>>>>>Substep 1.3 Recognizing events automatically elpased time: '+ str(substep_finish_time - substep_init_time) + "\n",'',enco)
        repg.register_log([execution_log],'>>>>>>Substep 1.3 (automatically) ends '+ substep_finish_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',enco)
            
    return list_of_events_to_process,list_list_of_relevant_features_for_the_user,list_list_discarded_features_by_the_user


def create_directory(fullpath_to_directory):
    '''
    This function allows to create a non-existent directory in the path specified.If the directory already exists no action is performed.    
    If the directory cannot be created a ValueException is raised.
    
    Parameters:
    :param str fullpath_to_directory: Full path to the location of the new directory that includes the name of the directory to be created.    
    :exception ValueError: If the directory could not be created
    
    :return: None
    '''
    
    try:
        os.makedirs(fullpath_to_directory)
        
    except OSError: 
        pass
    
    except Exception:
        raise ValueError("Directory could no be created " + str(fullpath_to_directory))


def delete_directory_with_content(fullpath_to_directory):
    '''
    This function allows to remove the specified directory
    
    Parameters:
    :param str fullpath_to_directory: Full path to the location of the directory to erase    
    :return: None
    '''
    
    shutil.rmtree(fullpath_to_directory,ignore_errors=True)


def remove_feature_from_list(feature_to_remove,list_of_features):
    '''
    This function allows to delete the variable, that is passed as parameter, from the list

    Parameters:
    :param str feature_to_remove: variable to delete from the list
    :param list<str> list_of_features: list of variables where to find the variable to delete
    
    :return: list with the non-deleted variables
    :rtype: list<str>
    '''
    
    if feature_to_remove in list_of_features:
        list_of_features.remove(feature_to_remove)    
    return list_of_features


def count_number_of_observations_by_target(df_data,target):
    '''
    The function allows to count the number of observations that have each one of the existent value for the target variable in df_datos

    Parameters:
    :param pandas_dataframe df_datos: dataframe with all data available where the target feature is included
    :param str target: objective variable
    
    :return: dictionary_with_value_for_target_and_occurrences,dictionary where the keys are the differents values of the targets and the values are the number of observations under the particular target
    :rtype: dict<int:int>
    '''
    
    lista_targets = df_data[target]
    dictionary_with_value_for_target_and_occurrences = {}
    for current_target in lista_targets:
        if(current_target not in dictionary_with_value_for_target_and_occurrences):
            dictionary_with_value_for_target_and_occurrences[current_target] = 1
        else:
             valores = dictionary_with_value_for_target_and_occurrences[current_target]   
             valores+=1
             dictionary_with_value_for_target_and_occurrences[current_target] = valores
    
    return dictionary_with_value_for_target_and_occurrences


def deleting_empty_and_constant_features(df_data):
    '''
    This function allows to remove empty and constant features from a dataframe

    Parameters:
    :param pandas_Dataframe df_data: Dataframe with all availabel data
    :param list<str> lista_features: list with relevant features excluding the target variable
    
    :return: list of non-empty and non-constant variables
    :rtype: list<str>
    '''

    '''Full list of available features to check'''
    list_features_to_check = list(df_data.columns)
    
    '''Looping through the list to find empty or constant features'''
    list_of_empty_or_constant_features = []
    for i in range(len(list_features_to_check)):        
        current_feature = list_features_to_check[i]
        
        list_of_elements = pd.unique(df_data[current_feature])
        list_of_elements = list_of_elements.tolist()
        if len(list_of_elements) == 1:                                    
            list_of_empty_or_constant_features.append(current_feature)
    
    '''Obtaining the list with the variable non-empty and non-constant'''
    list_valid_features= get_complementary_list(list_features_to_check,list_of_empty_or_constant_features)
    
    return sorted(list_valid_features), sorted(list_of_empty_or_constant_features)


def get_features_to_train_models(list_relevant_features_and_scores,features_discarded_mif_process):
    '''
    Allows to obtaind the relevant features without those that were specified by the user
    to be discarded

    Parameters:
    :param list<str> list_relevant_features_and_scores: list of variables and scores from MIF of each one
    :param list<str> features_discarded_mif_process: lsit of variables specified by the user to be discarded
    
    :return: list_mif_features, list with relevant features after mif
    :rtype: list<str>
    :return: dictionary_relevant_features_scores, dictionary with relevant features and their corresponding score
    :rtype: list<str>
    '''
    
    dictionary_relevant_features_scores = {}
    list_mif_features = []
    for element in list_relevant_features_and_scores:
        dictionary_relevant_features_scores[element[0]] = round(float(element[1]),6)
        list_mif_features.append(element[0])
        
    for feature in features_discarded_mif_process:
        if (feature in dictionary_relevant_features_scores):
            list_mif_features.remove(feature)
            del dictionary_relevant_features_scores[feature]
                        
    return list_mif_features,dictionary_relevant_features_scores

	
def split_dataset_catalogued_non_catalogued(vector_with_fullpath_to_input_files,feature_events_name,event,target,maximum_number_of_registers,non_catalogued_target_value,input_files_delimiter,log_file,list_of_user_discarded_features,enco):
    '''
    It allows to obtain all the observations for a specific event which target has a known value,
    it is to say, it is different than int(1). The number of observations to process van be limited

    Parameters:
    :param list<str> vector_with_fullpath_to_input_files: lista con las rutas completas a los ficheros de los cuales se obtendran las observaciones
    :param dict dictionary_with_variables_known_recodification: diccionario con las variables que se usaran a lo largo del programa
    :param str event: event to filter to obtain known observations
    :param str target: feature objetivo
    :param int64 maximum_number_of_registers: numero maximo de observaciones a concatenar
    :param list threshold_split: intervals to recodify target feature values
    :param list list_of_objective_feature_recodified_values: values to recodify target feature original values
    :param str input_files_delimiter: field delimiter for the input files
    :param str fichero_log: log to register information about current process
    :param list_of_user_discarded_features: list of variables established by the user to be discarded
    :param encon: encoding
        
    
    :return: un dataframe con las observaciones que tienen un target conocido
             si no se obtienen observaciones validas, devuelve un Dataframe vacio
    :rtype: tuple(list<str>,pandas Dataframe) | empty Dataframe
    '''   
    event = codify_using_enconding(event,enco)
    name = feature_events_name
    name = codify_using_enconding(name,enco)
    
    target = codify_using_enconding(target,enco)
    mandatory_features = [name]    
    list_of_catalogued_registers=[] 
    list_of_non_catalogued_registers=[] 
    current_amount_catalogued_observations = 0
    current_amount_non_catalogued_observations = 0
    condition_catalogued_registers = True
    condition_non_catalogued_registers = True
    if(maximum_number_of_registers > 0):
        repg.register_log(log_file,'>> It will be read a maximum number of: ' + str(maximum_number_of_registers) + ' ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',enco)
    else:
        repg.register_log(log_file,'>> It will be read the maximum number of registers '+ datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',enco)
    
    for i in range(len(vector_with_fullpath_to_input_files)):
        if(maximum_number_of_registers != 0):
            condition_catalogued_registers = current_amount_catalogued_observations < maximum_number_of_registers
            condition_non_catalogued_registers = current_amount_non_catalogued_observations < maximum_number_of_registers
        
        '''Checking if maximum number of catalogued and/or not catalogued observations reached'''
        if((i<vector_with_fullpath_to_input_files) and ((condition_catalogued_registers) or (condition_non_catalogued_registers)) ):
            repg.register_log(log_file,'>> Reading csv ' + str(i) + ': ' + vector_with_fullpath_to_input_files[i] + ' ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',enco)
            
            original_data=pd.read_csv(vector_with_fullpath_to_input_files[i],sep=input_files_delimiter)            
            try:
                
                original_data = original_data[original_data[name] == event]
            except KeyError:
                raise Exception('Feature selected that should contain the events is not available in datasets')
        
            '''Recodifying objective target values'''
            try:
                original_data[target]
            except KeyError:
                raise Exception('Feature selected as target is not available in datasets')
                                    
            '''Separating catalogued and non catalogued observations'''                         
            data_not_catalogued = original_data[original_data[target] == non_catalogued_target_value]
            data_catalogued = original_data[original_data[target] != non_catalogued_target_value]
        
            data_not_catalogued = data_not_catalogued.reset_index()
            data_catalogued = data_catalogued.reset_index()
                                
            if(maximum_number_of_registers != 0): # if maximum limit
                current_amount_catalogued_observations = 0
                for ind in range(len(list_of_catalogued_registers)):
                    current_amount_catalogued_observations += len(list_of_catalogued_registers[ind])
                number_of_registers_to_add = (maximum_number_of_registers - (current_amount_catalogued_observations))
                if (number_of_registers_to_add > 0):# if maximum limit was not reached
                    if(len(data_catalogued) < number_of_registers_to_add):            
                        list_of_catalogued_registers.append(data_catalogued)
                    else: 
                        good_data_mod = data_catalogued.iloc[0:number_of_registers_to_add]
                        list_of_catalogued_registers.append(good_data_mod)
                
                current_amount_non_catalogued_observations = 0
                for ind in range(len(list_of_non_catalogued_registers)):
                    current_amount_non_catalogued_observations += len(list_of_non_catalogued_registers[ind])
                number_of_registers_to_add = (maximum_number_of_registers - (current_amount_non_catalogued_observations))
                if (number_of_registers_to_add > 0):
                    if(len(data_not_catalogued) < number_of_registers_to_add):            
                        list_of_non_catalogued_registers.append(data_not_catalogued)
                    else: 
                        bad_data_mod = data_not_catalogued.iloc[0:number_of_registers_to_add]
                        list_of_non_catalogued_registers.append(bad_data_mod)
                        
            else: 
                
                if(not data_catalogued.empty):
                    list_of_catalogued_registers.append(data_catalogued)
                if(not data_not_catalogued.empty):
                    list_of_non_catalogued_registers.append(data_not_catalogued)
    
    '''Checking if valid catalogued data'''
    df_result_for_catalogued_data = pd.DataFrame()
    df_result_for_not_catalogued_data = pd.DataFrame()
    if list_of_catalogued_registers != []:
        data_catalogued=pd.concat(list_of_catalogued_registers)        
        del list_of_catalogued_registers
        coded_list_of_features = []        
        for elemento in data_catalogued.columns:            
            elemento = elemento.decode(enco)
            coded_list_of_features.append(elemento)        
        data_catalogued.columns = coded_list_of_features
        data_catalogued[decodify_using_enconding(name,enco)] = data_catalogued[decodify_using_enconding(name,enco)].apply(lambda x: x.decode(enco))
        head = list(data_catalogued)
        full_list_of_features = remove_feature_from_list('Unnamed: 0',head)
        full_list_of_features = remove_feature_from_list('index',full_list_of_features)
        for variable_to_discard in list_of_user_discarded_features:            
            remove_feature_from_list(variable_to_discard,full_list_of_features)
        df_result_for_catalogued_data = data_catalogued[full_list_of_features]
    else:        
        df_result_for_catalogued_data = pd.DataFrame(columns=mandatory_features) 
        
    if list_of_non_catalogued_registers != []:
        data_not_catalogued=pd.concat(list_of_non_catalogued_registers)        
        del original_data
        del list_of_non_catalogued_registers
        head = list(data_not_catalogued)
        full_list_of_features = remove_feature_from_list('Unnamed: 0',head)
        full_list_of_features = remove_feature_from_list('index',full_list_of_features)
        for variable_to_discard in list_of_user_discarded_features:
            remove_feature_from_list(variable_to_discard,full_list_of_features)        
        df_result_for_not_catalogued_data = data_not_catalogued[full_list_of_features]
    else:        
        df_result_for_not_catalogued_data = pd.DataFrame(columns=mandatory_features)
    
    return df_result_for_catalogued_data,df_result_for_not_catalogued_data


def get_two_subsets_using_random_split(dataframe,percentaje):
    '''
    It allows to obtain two new dataframes trai and test dataframe according to the 
    specified percentaje

    Parameters:
    :param pandas_dataframe dataframe: dataframe to split between train and test data
    :param int percentaje: percentaje to divide the original dataframe
        
    
    :return: a new tuple of dataframes 
             if no valid observations,an empty Dataframe vacio
    :rtype: tuple(list<str>,pandas Dataframe) | empty Dataframe
    '''
    
    msk = np.random.rand(len(dataframe))
    bound=np.percentile(msk,percentaje*100)
    msk=msk<bound
    train=dataframe[msk]
    test=dataframe[~msk]
    return(train,test)


def split_train_test_datasets(df_catalogued_data,target,percentaje):   
    '''
    It allows to obtain two new dataframes of observations according to a percentaje

    Parameters:
    :param pandas_dataframe df_catalogued_data: dataframe to split between train and test data
    :param str target: objective feature
    :param int percentaje: percentaje to divide the original dataframe
        
    
    :return: train_data, dataframe with observations to train the models
    :rtype: pandas Dataframe   
    :return: test_data, dataframe with observations to test the models
    :rtype: pandas Dataframe   
    '''    
    
    current_different_targets = list(set(df_catalogued_data[target].values))    
    train_data = []
    test_data = []        
    
    for current_target in current_different_targets:        
        df_registers_current_target = df_catalogued_data[df_catalogued_data[target] == current_target]
        current_train_data, current_test_data=get_two_subsets_using_random_split(df_registers_current_target, percentaje)
        if(len(current_train_data) < len(current_test_data)):
            aux = current_test_data.copy()
            current_test_data = current_train_data.copy()            
            current_train_data = aux.copy()            
        train_data.append(current_train_data)
        test_data.append(current_test_data)
        
    train_data = pd.concat(train_data)
    test_data = pd.concat(test_data)
    
    train_data=train_data.reset_index()
    test_data=test_data.reset_index()
         
    train_data=train_data.drop('index',axis=1)
    test_data=test_data.drop('index',axis=1)

    return train_data,test_data


def get_models_to_train(learning,list_of_models):
    '''
    It allows to obtain the full list of available models to apply

    Parameters:
    :param learning: type of learning, supervised or unsupervised
    :param list_of_models: list of available models for the type of learning
            
    :return: models_to_apply,dictionary with model to apply
    :rtype: dict<learning:model>
    '''
    
    dictionary_with_learnings_related_models = datr.get_dictionary_with_learnings_related_to_models()
    
    models_to_apply = [] #list with the names of the models that will be applied to the event
    for current_model in list_of_models:
        if current_model == 0: #it means that all the supervised/unsupervised models available will be applied
            for key in dictionary_with_learnings_related_models[learning]:
                if(key != 0):
                        selected_model = dictionary_with_learnings_related_models[learning][key]
                        models_to_apply.append(selected_model)
        else:
            selected_model = dictionary_with_learnings_related_models[learning][current_model]
            models_to_apply.append(selected_model)
        
    return models_to_apply


def angle_two_points(a,b):
    '''
    It allows to calculate the angle between a line and the horizontal axis
    Parameters:
    :param list a: a list of two coordinates that defines one point
    :param list b: a list of two coordinates that defines one point
    
    :return alpha: angle in radians
    :rtype float
    '''
    xa=float(a[0])
    ya=float(a[1])
    xb=float(b[0])
    yb=float(b[1])
    tangent=(yb-ya)/float(xb-xa)
    alpha=np.arctan(tangent) #radianes
    return(alpha)

def angle_two_lines(ts,ss,cvs):
    '''
    It allows to calculate the mean convergence velocity
    Parameters:
    :param list ts: horizontal axis points
    :param list ss: points corresponding to one path
    :param list cvs: points corresponding to one path
    
    :return alpha: angle in radians
    :rtype float
    '''
    ass=[ts[0],ss[0]]
    bss=[ts[-1],ss[-1]]
    acv=[ts[0],cvs[0]]
    bcv=[ts[-1],cvs[-1]]
    alphass=angle_two_points(ass,bss)
    alphacv=angle_two_points(acv,bcv)
    alpha=alphacv-alphass #radianes
    return(alpha)
    
 
def transformation_learning_curve(ts,ss,cvs):
    '''
    It allows to calculate plain transformation
    Parameters:
    :param list ts: horizontal axis points
    :param list ss: points corresponding to one path
    :param list cvs: points corresponding to one path
    
    :return ts: horizontal axis points, ss: points corresponding to one transformed path, cvs: points corresponding to one transformed path
    :rtype list<list<float,float>>    
    '''
    new_ss=[]
    new_cvs=[]
    for i in range(len(ts)):
        new_ss.append(ss[i]+(1-ss[i])/float(10))
        new_cvs.append(cvs[i]-cvs[i]/float(10))
    return(ts,new_ss,new_cvs)


def weight_for_decision_function(ts,ss,cvs,cm,w,main_metric):
    '''It allows to calculate plain transformation
    Parameters:
    :param list ts: horizontal axis points
    :param list ss: points corresponding to one path
    :param list cvs: points corresponding to one path
    
    :return ts: horizontal axis points, ss: points corresponding to one transformed path, cvs: points corresponding to one transformed path
    :rtype list<list<float,float>>'''
	
    nts=transformation_learning_curve(ts,ss,cvs)[0]
    nss=transformation_learning_curve(ts,ss,cvs)[1]
    ncvs=transformation_learning_curve(ts,ss,cvs)[2]
    alpha=float(angle_two_lines(nts,nss,ncvs)) #radianes
    pi_final=float(np.abs(nss[-1]-ncvs[-1]))
    pi_zero=float(np.abs(nss[0]-ncvs[0]))
    cw=0
    
    total=cm/float(np.sum(cm))
    for i in range(len(cm)):
        for j in range(len(cm)):
            cw=cw+total[i][j]*w[i][j]
    cwfin=cw
    W=main_metric*10 + float(cwfin) - np.abs(float(pi_final)) - 2*(np.pi/2-np.abs(alpha))*np.abs(float(pi_zero))/np.pi
    return(W)


def random_selection(df,percentaje):
    msk = np.random.rand(len(df))
    bound=np.percentile(msk,percentaje)
    msk=msk<bound
    df_selection=df[msk]
    return(df_selection)


def random_selection_same_distribution(df,target,percentaje):
    target_values=list(set(list(df[target].values)))
    df_per_value=[]
    for value in target_values:
        df_value=df[df[target]==value]
        df_value=random_selection(df_value,percentaje)
        df_per_value.append(df_value)
        del df_value
    df_random_selection=pd.concat(df_per_value)
    return(df_random_selection.reset_index())


'''Functions related to the selection of the best model and the predictions part'''

def get_dictionary_of_models_score_time(report_dict_evento,generic_list_of_models):
    '''
    It allows to get the dictionary of models with their score and time
    Parameters:
    :param dict report_dict_evento: current information about the event in dict format
    :param list generic_list_of_models: list with the avilable models to be apply    
    
    :return dictionary_models_score_time, dictionary wit the models and their current score and time
    :rtype list<list<str,score,time>>  
    '''
    
    score_key = 'current_score'
    time_key='current_time'
    dictionary_models_score_time = {}
    for model in generic_list_of_models:
        if (model in report_dict_evento):            
            index_dec_modelo = round(float(report_dict_evento[model]['Decision_index']),4)
            training_elapsed_time = report_dict_evento[model]['Time_parameters']['time_train_finish'] - report_dict_evento[model]['Time_parameters']['time_train_init']
            dictionary_models_score_time[model] = {score_key:index_dec_modelo,time_key: training_elapsed_time}
            
    return dictionary_models_score_time #dictionary of models with their score and time


def order_models_by_score_and_time(report_dict_evento,generic_list_of_models):
    '''
    It allows to get the list of models ordered by score and time
    Parameters:
    :param dict report_dict_evento: current information about the event in dict format
    :param list generic_list_of_models: list with the avilable models to be apply    
    
    :return list_models_decision_index_ordered_by_score, list of models ordered by score and time
    :rtype list<list<str,score,time>>  
    '''
    score_key = 'current_score'
    time_key='current_time'
    dictionary_models_scores_time = get_dictionary_of_models_score_time(report_dict_evento,generic_list_of_models)
    resultant_list_ordered_by_models_time = []    
    
    while (dictionary_models_scores_time != {}):                
        maximum_score_available = ''
        list_models_maximum_score = []
        for model in dictionary_models_scores_time:
            if(maximum_score_available == ''):
                maximum_score_available = dictionary_models_scores_time[model][score_key]
            else:
                current_score = dictionary_models_scores_time[model][score_key]
                if(current_score > maximum_score_available):
                    maximum_score_available = current_score
                    
        '''Get models with maximum available score'''
        for model in dictionary_models_scores_time:
            score_to_check = dictionary_models_scores_time[model][score_key]
            if(score_to_check == maximum_score_available):
                list_models_maximum_score.append(model)
        
        '''Get list with models that share maximun score'''
        if(len(list_models_maximum_score) > 1):
            '''Checking times to get the best'''
            cp_models_max_score = copy.deepcopy(list_models_maximum_score)
            while(list_models_maximum_score != []):
                best_time = ''
                fastest_model_name = ''
                for model in list_models_maximum_score:
                    if (best_time == ''):
                        best_time = dictionary_models_scores_time[model][time_key]
                        fastest_model_name = model
                        
                    else:
                        current_time = dictionary_models_scores_time[model][time_key]
                        if(current_time < best_time):
                            best_time = current_time
                            fastest_model_name = model
                
                resultant_list_ordered_by_models_time.append([fastest_model_name,maximum_score_available,best_time])
                list_models_maximum_score.remove(fastest_model_name)
            for modelo_eliminar in cp_models_max_score:
                del dictionary_models_scores_time[modelo_eliminar]                
            
        else:
            model = list_models_maximum_score[0]            
            time_for_current_model = dictionary_models_scores_time[model][time_key]
            resultant_list_ordered_by_models_time.append([model,maximum_score_available,time_for_current_model])
            del dictionary_models_scores_time[model]
            
    return resultant_list_ordered_by_models_time


def get_model_with_highest_decision_index(report_dict,generic_list_of_models):
    '''
    It allows to get the model with the highest decision index
    Parameters:
    :param dict report_dict: current information about the event in dict format
    :param list generic_list_of_models: list with the avilable models to be apply    
    
    :return model_max_dec, name of the best model
    :rtype str
    :return max_dec_index, maximun decision index
    :rtype float
    '''
    ordered_list_model_score_time = order_models_by_score_and_time(report_dict,generic_list_of_models)
    
    model_max_dec_index = ordered_list_model_score_time[0][0]
    max_dec_index = ordered_list_model_score_time[0][1]
    return model_max_dec_index,max_dec_index


def add_entry_for_event_in_prediction_dictionary(prediction_dictionary,event):
    '''
    It allows to add a new entry for the event in the prediction dictionary
    Parameters:
    :param dict prediction_dictionary: current information about events and 
    :param event: name of the event to add
    
    :return prediction_dictionary, dictionary with events and prediction models info
    :rtype dict
    '''
    if(event not in prediction_dictionary):
        prediction_dictionary[event] = {}    
    return prediction_dictionary


def get_original_list_features(list_features_derived_and_original,reserved_character):
    '''
    It allows to retreieve the list wiht the original names of derived features
    Parameters:
    :param list list_features_derived_and_original: list of the features
    :param reserved_character: special character to obtain derived fatures
    
    :return original_list_of_features, list of the original features
    :rtype dict
    '''
    original_list_of_features = []
    for feature in list_features_derived_and_original:
        if reserved_character in feature:
            feature = feature.split(reserved_character)
            feature = feature[0]
            original_list_of_features.append(feature)
        else:
            original_list_of_features.append(feature)
        original_list_of_features = sorted(list(set(original_list_of_features)))
    return original_list_of_features


def update_prediction_models_original_features_for_event_in_prediction_dictionary(prediction_dictionary,event,target,model_characteristics,reserved_character,list_of_parameters):
    '''
    It allows to store in the predictino dictionary the list with original features
    Parameters:
    :param dict prediction_dictionary: current information about events and 
    :param event: name of the event to add
    
    :return prediction_dictionary, dictionary with events and prediction models info
    :rtype dict
    '''
    prediction_dictionary[event][target] = {}
    prediction_dictionary[event][target][list_of_parameters[0]] = model_characteristics[0]
    prediction_dictionary[event][target][list_of_parameters[1]] = model_characteristics[1]
    prediction_dictionary[event][target][list_of_parameters[2]] = model_characteristics[2]
    prediction_dictionary[event][target][list_of_parameters[3]] = model_characteristics[3]
    '''Recover original list of features to check if derived ones were created'''
    original_features = get_original_list_features(model_characteristics[3],reserved_character)
    prediction_dictionary[event][target][list_of_parameters[4]] = original_features #list with original features
    return prediction_dictionary

# ---------------------------------------------------------
# natsort.py: Natural string sorting. https://github.com/ActiveState/code/tree/master/recipes/Python/285264_Natural_string_sorting  
#By Connelly Barnes.
# ---------------------------------------------------------

def try_int(s):
    "Convert to integer if possible."
    try: return int(s)
    except: return s

def natsort_key(s):
    "Used internally to get a tuple by which s is sorted."
    import re
    return map(try_int, re.findall(r'(\d+|\D+)', s))

def natcmp(a, b):
    "Natural string comparison, case sensitive."
    return cmp(natsort_key(a), natsort_key(b))

def natcasecmp(a, b):
    "Natural string comparison, ignores case."
    return natcmp(a.lower(), b.lower())

def natsort(seq, cmp=natcmp):
    "In-place natural string sort."
    seq.sort(cmp)
    
def natsorted(seq, cmp=natcmp):
    "Returns a copy of seq, sorted by natural string sort."
    import copy
    temp = copy.copy(seq)
    natsort(temp, cmp)
    return temp

