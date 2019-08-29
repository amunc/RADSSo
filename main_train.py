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

import ast
import copy
import ConfigParser as conp
import datetime
import numpy as np
import os
import sys
import time
import traceback
import shutil

import auxiliary_functions as auxf
import data_request as datr
import metrics as metr
import parameters_selection as spar

import supervised as supe
import reports_generation as repg
import unsupervised as unsu

from sklearn.model_selection import train_test_split



class InvalidFeatureException(Exception):
    pass

class FollowFlowException(Exception):
    pass

class FinishFlowException(Exception):
    pass


#############################################################################
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Init of the porgram<<<<<<<<<<<<<<<<<<<<<<<<<<<#
#############################################################################
auxf.print_initial_license_message()
'''BASE_PATH contains the path where the file .py is being executed '''
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

'''Creation of the variable that parses the configuration file. If the file is
not found, an exception is thrown and finishes the execution'''
path_to_configuration_file =os.path.join(BASE_PATH,"conf.ini")
config_parser = conp.ConfigParser()
if(os.path.exists(path_to_configuration_file)):
    config_parser.read(path_to_configuration_file)
else:
    raise Exception('Configuration file (conf.ini) was not found')

'''Reading the different sections fo the log'''
logs_section = 'Logs section'
input_data_section = 'Input data section'
auxiliary_data_section = 'Auxiliary data section'
output_data_section = 'Output data section'
testing_section = 'Validation'

'''Reading field separator for the input and ouput files, the format of the images
and the execution mode'''
'''Mandatory execution parameters - Do not change if not sure. It can cause impredictable behaviour'''

char_encoding = 'utf-8' #encoding used along the script

lc_output_format = 'png'#learning curves output format
input_files_extension = 'csv'
reserved_derived_features_character = '_'
list_of_not_valid_characters = ['\\','/',':','*','?','<','>','|']
max_lenght_output_names = 180
list_of_parameters_models_events_dict = ['learning','model_path','dict_reassignment','features','original_features']
list_names_metrics = ['precision','recall','specificity','f1_score','mcc']
target = 'generic' #to create generic error log if error in objective input target
max_pars=[30,30]

'''Mandatory execution parameters - Do not change if not sure. It can cause impredictable behaviour'''
    
'''Creating directory of logs and declaring error_log file'''
log_path = os.path.join(BASE_PATH,auxf.decodify_using_enconding(config_parser.get(logs_section,'logs_directory_name'),char_encoding))
auxf.create_directory(log_path)

errors_file = ''

input_files_delimiter = config_parser.get(input_data_section,'input_files_delimiter')
output_files_delimiter = config_parser.get(output_data_section,'output_files_delimiter')
file_with_features_delimiter = ' '
main_metric = config_parser.get(input_data_section,'main_metric')
default_matrix = config_parser.get(input_data_section,'default_matrix')
validation_mode = config_parser.get(testing_section,'validation_mode') # data is separated also in validation data when True

execution_init_time = datetime.datetime.fromtimestamp(time.time())

try: 
    
    '''Step 0: It starts the execution of the script and creates log files'''
    observation_number = config_parser.get(input_data_section,'obsnumber') # data is separated also in validation data when True
    start_time=time.time()
    
    
    '''Step 0.1: Creating log files: Errors, Execution and Execution-Time log '''    
    errors_file = os.path.join(log_path,auxf.decodify_using_enconding(config_parser.get(logs_section,'log_error_filename'),char_encoding)+target+'.log')
    repg.register_log([errors_file],"Errors registered:\n",0,char_encoding)    
            
    '''Setp 0.2: reading name feature (feature with events names)'''
    name_variable_with_events = config_parser.get(input_data_section,'event_name_feature')
    
    target = auxf.check_input_target(sys.argv)
    target = auxf.decodify_using_enconding(target,char_encoding)
    
    label_non_cataloged=int(config_parser.get(input_data_section,'label_non_catalogued'))
    
    errors_file = os.path.join(log_path,auxf.decodify_using_enconding(config_parser.get(logs_section,'log_error_filename'),char_encoding)+target+'.log')    
    repg.register_log([errors_file],"Errors registered:\n",0,char_encoding)   
    log_file = os.path.join(log_path,auxf.decodify_using_enconding(config_parser.get(logs_section,'execution_log_filename'),char_encoding)+target+'.log')    
    time_log = os.path.join(log_path,auxf.decodify_using_enconding(config_parser.get(logs_section,'time_log_filename'),char_encoding)+target+'.log')        
    
    
    '''Phase 1'''
    '''Step 1: Creating lists of input files, reading execution parameters, reading json with events and variables (eventos),
    creating output directory and creating a summary report with all the details of the events, learnings and variables to be processed'''
    repg.register_log([log_file],'Start of execution\n>>>>>>Step 1: Setting base_path and reading execution parameters(events, learning methods, requested models) '+ datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n\n" \
                       ,0,char_encoding)
    repg.register_log([time_log],'>>>>Step 1 starts:\n',0,char_encoding)
    step_init_time = datetime.datetime.fromtimestamp(time.time())

    '''Substep 1.1: Creating a list of input files to process, the maximun number of files y and the maximun number of observations'''
    repg.register_log([log_file],'>>>>>> Substep 1.1: Creating array with paths to files '+ step_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
    substep_init_time = datetime.datetime.fromtimestamp(time.time())
                
    '''Path to files and number of files to process'''
    path_to_input_files = os.path.join(BASE_PATH,auxf.decodify_using_enconding(config_parser.get(input_data_section,'path_to_root_directory_input_files'),char_encoding))
    path_to_dir_user_files = os.path.join(BASE_PATH,auxf.decodify_using_enconding(config_parser.get(input_data_section,'user_files_directoryname'),char_encoding))
    auxiliary_directory_filename = auxf.decodify_using_enconding(config_parser.get(auxiliary_data_section,'auxiliary_directory_filename'),char_encoding)
    path_to_dir_auxiliary_files = os.path.join(BASE_PATH,auxiliary_directory_filename)    
        
    maximum_number_input_files= int(config_parser.get(input_data_section,'maximum_number_files_to_read'))
    
    vector_full_paths_input_files= auxf.get_all_files_in_dir_with_extension(path_to_input_files, maximum_number_input_files,input_files_extension)
        
    repg.register_log([log_file],'>> The next files will be read:\n','',char_encoding)
    for input_file_in_list in vector_full_paths_input_files:
        repg.register_log([log_file],"\t"+ input_file_in_list+ "\n",'',char_encoding)
                
    '''Number of observations to process'''
    maximum_number_of_observations_to_process = int(config_parser.get(input_data_section,'maximum_number_observations_to_read'))
    
    substep_finish_time = datetime.datetime.fromtimestamp(time.time())    
    repg.register_log([log_file],'>>>>>>Substep 1.1 ends '+ substep_finish_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
    repg.register_log([time_log],'>>Substep 1.1 Concatenating files elapsed time: ' + str(substep_finish_time - substep_init_time) + "\n",'',char_encoding)
    
    
    '''Substep 1.2 Reading conf.ini parameters (path to json with event, variables specified by user and discarded variables)'''
     
    substep_init_time = datetime.datetime.fromtimestamp(time.time())
    repg.register_log([log_file],'>>>>>>Substep 1.2: Reading events and variables (mandatory and discarded) '+ substep_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)

    '''Path to json with events''' 
    events_filename = auxf.decodify_using_enconding(config_parser.get(input_data_section,'events_filename'),char_encoding)
    path_to_input_file_with_events = os.path.join(path_to_dir_user_files,events_filename)


    '''Path to file with commom discarded variables'''
    discarded_variables_filename = auxf.decodify_using_enconding(config_parser.get(input_data_section, 'user_discarded_variables_filename'),char_encoding)    
    path_to_discarded_variables_file = os.path.join(path_to_dir_user_files,discarded_variables_filename)
    
    substep_finish_time = datetime.datetime.fromtimestamp(time.time())
    repg.register_log([log_file],'>>>>>>Substep 1.2 ends '+ substep_finish_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
    repg.register_log([time_log],'>>Substep 1.2 Reading parameteres from conf.ini elpased time: '+ str(substep_finish_time - substep_init_time) + "\n",'',char_encoding)

    '''Retrieving train-test division percentage from conf.ini'''
    data_division_percentaje= float(config_parser.get(input_data_section,'train_test_division_percentaje'))                         
    try:
        auxf.check_train_test_division_percent(data_division_percentaje)
    except ValueError as e:
        raise ValueError(e)
            
    '''Substep 1.3 Loading events to process and their features: Depending on the third parameter, the reading
    of the events will be done automatically or manually (from a file constructed by the user)'''
    
    '''Checking automatic or manual events and variables lecture'''
    list_of_events = []
    list_of_user_relevant_variables_event = []
    list_of_discarded_variables_events = []
    try:
        list_of_events,list_of_user_relevant_variables_event,list_of_discarded_variables_events = auxf.checking_events_reading_method(sys.argv,vector_full_paths_input_files,path_to_input_file_with_events,name_variable_with_events,input_files_delimiter,log_file,time_log,list_of_not_valid_characters,char_encoding)        
    except KeyError as e:        
        raise KeyError('Selected feature for events is not available in the datasets')
    
    '''Substep 1.4: List with common discarded variables for all the events '''
    substep_init_time = datetime.datetime.fromtimestamp(time.time())    
    repg.register_log([log_file],'>>>>>>Substep 1.4: Loadings list with discarded variables '+ substep_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n\n" \
                       ,'',char_encoding)
    
    list_common_discarded_variables = auxf.load_dictionary_of_variables(path_to_discarded_variables_file,'discarded_common_features',char_encoding)
    substep_finish_time = datetime.datetime.fromtimestamp(time.time())
    repg.register_log([log_file],'>>>>>>Substep 1.4 ends '+ substep_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
    repg.register_log([time_log],'>>Substep 1.4 Loading discarded variables by the user elapsed time: '+ str(substep_finish_time - substep_init_time) + "\n",'',char_encoding)
             
    
    '''Substep 1.5: Loading lists with learnings and models, no user interaction,
     all learning and models are loaded '''
    substep_init_time = datetime.datetime.fromtimestamp(time.time())
    repg.register_log([log_file],'>>>>>>Substep 1.5: Loadings lists with learnings and models '+ substep_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n\n" \
                       ,'',char_encoding)
    ''' Getting static dict with learnings '''
    available_learnings_dict = datr.generate_learnings_dict()
        
    ''' Getting static dict with models '''        
    separador_modelos = ' '
    supervised_models_dictionary = datr.generate_dictionary_available_supervised_learning_models()
    list_of_supervised_models_av = supervised_models_dictionary.values()
    list_of_supervised_models_av.remove('All')
    unsupervised_models_dictionary = datr.generate_dictionary_available_unsupervised_learning_models()
    list_of_unsupervised_models_av = unsupervised_models_dictionary.values()
    list_of_unsupervised_models_av.remove('All')
    full_list_of_available_models = [list_of_supervised_models_av,list_of_unsupervised_models_av]
    
    list_available_learnings = []
    list_of_mod = []
    for event in list_of_events:        
        list_available_learnings.append('All')
        list_of_mod.append(full_list_of_available_models)

    substep_finish_time = datetime.datetime.fromtimestamp(time.time())
    repg.register_log([log_file],'>>>>>>Substep 1.5 ends '+ substep_finish_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
    repg.register_log([time_log],'>>Substep 1.5 Loading learnings and models elapsed time: '+ str(substep_finish_time - substep_init_time) + "\n",'',char_encoding)
             
    
    '''Checking correspondencies in number between events, relevant variables and discarded variables.'''    
    datr.check_lists_of_features_specified_by_user(list_of_events,list_of_user_relevant_variables_event,list_of_discarded_variables_events)   
    
    '''Checking correspondencies in number between events and models'''    
    datr.check_correspondency_input_lists_events_learnings_and_models(list_of_events,list_available_learnings,list_of_mod)                       
        
    
    '''Substep 1.6: Creating static dictionary for variables with known recodification, 
    the ouput directory and one report with all the details about actual execution'''
     
    substep_init_time = datetime.datetime.fromtimestamp(time.time())
    repg.register_log([log_file],'>>>>>>Substep 1.6: Loading dictionary with known recodifications and generating general execution report '+ substep_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)

    '''Obtaining percentil'''
    percentil = datr.checking_percentil_value(config_parser.get(input_data_section,'percentaje_relevant_variables'))    
    
    ''' Stablishing output directories '''
    ruta_directorio_raiz_salida = os.path.join(BASE_PATH,auxf.decodify_using_enconding(config_parser.get(output_data_section,'output_directory_rootname'),char_encoding)) #directorio raiz de salida    
    auxf.create_directory(ruta_directorio_raiz_salida)
    
    '''Generating actual execution report'''
    execution_info = datr.generate_information_about_current_execution(list_of_events,list_available_learnings,list_of_mod,target,percentil,list_of_user_relevant_variables_event,list_common_discarded_variables)
    repg.register_log([log_file], execution_info +"\n",'',char_encoding)  
    report_data = repg.create_basic_report_data_dict(percentil,target,main_metric,list_common_discarded_variables,"'" + os.path.join(path_to_dir_auxiliary_files,"logo.jpg") + "'")
    repg.create_report_current_execution(report_data,list_of_events,list_of_user_relevant_variables_event,list_of_discarded_variables_events,list_available_learnings,list_of_mod, available_learnings_dict,auxiliary_directory_filename, ruta_directorio_raiz_salida,char_encoding)
    
    substep_finish_time = datetime.datetime.fromtimestamp(time.time())    
    repg.register_log([log_file],'>>>>>>Substep 1.6 ends '+ substep_finish_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
    repg.register_log([time_log],'>>Substep 1.6 - Loading dictionary with known recodifications and generating general execution report: ' + str(substep_finish_time - substep_init_time)  + "\n",'',char_encoding)
    
    step_finish_time = datetime.datetime.fromtimestamp(time.time())
    repg.register_log([log_file],'>>>>>>Step 1 ends '+ step_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
    total_time_step_1 = step_finish_time - step_init_time
    repg.register_log([time_log],'>>>>Step 1 - Reading events and parameters total elapsed time: ' + str(total_time_step_1)  + "\n\n",'',char_encoding)
    
    
    '''Phase 2 - Creating models: Process of obtaining information about the features of each event to construct the datasets'''    
    
    '''Step 2: Obtaining prediction_models-events dict, creating individual event log and working with the variables of the current event'''
    repg.register_log([log_file],'>>>>>>Creating models phase: Processing events and getting models '+ datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
    repg.register_log([log_file],'>>>>>>Step 2: Creating models-event dict, events and getting models '+ datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding) 
    repg.register_log([time_log],'>>>>Step 2 starts:\n','',char_encoding)
        
    '''Substep 2.1: Retrieving the dictionary that stores the relationship between the events and the best prediction model for each one '''
    step_init_time = datetime.datetime.fromtimestamp(time.time())
    substep_init_time = datetime.datetime.fromtimestamp(time.time())
    repg.register_log([log_file],'>>>>>>Substep 2.1: Getting prediction model-event dictionary '+ substep_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
    
    path_to_directory_prediction_models = os.path.join(BASE_PATH,auxf.decodify_using_enconding(config_parser.get(output_data_section,'output_directory_name_prediction_models'),char_encoding))
    auxf.create_directory(path_to_directory_prediction_models)
    prediction_models_for_events_dictionary,handler_prediction_models_for_events_dictionary = auxf.retrieve_dictionary_and_handler(os.path.join(path_to_directory_prediction_models,config_parser.get(output_data_section,'prediction_models_dictionary_filename')))
    
    substep_finish_time = datetime.datetime.fromtimestamp(time.time())
    repg.register_log([log_file],'>>>>>>Substep 2.1 ends '+ substep_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
    repg.register_log([time_log],'>>Substep 2.1 Reading prediction models-event dictionary elapsed time: ' + str(substep_finish_time - substep_init_time)  + "\n",'',char_encoding)
       
    ''' Step time counters initialization'''
    total_time_step_2 = substep_finish_time - substep_init_time
    total_time_step_3 = 0
    total_time_step_4 = 0
    total_time_step_5 = 0
    total_model_phase_time = 0

    '''  Beginning of processing '''
    repg.register_log([time_log],'\n###############################################\n############   Processing events   ############\n###############################################\n\n','',char_encoding)
    for event_index in range(len(list_of_events)):        
        current_event = list_of_events[event_index]
        print '\nCreating models for ' + current_event
        
        repg.register_log([time_log],'############################################################################################\n#################################  '+ current_event + '\n############################################################################################\n','',char_encoding)
        repg.register_log([time_log],'>>>>Step 2 starts\n>>Substep 2.1 - Common step to all events\n','',char_encoding)
        
        '''Substep 2.2: Creating the temporary dictionary with the current models and the paths of the outputs for the current event '''
        substep_init_time = datetime.datetime.fromtimestamp(time.time())
        repg.register_log([log_file],'>>>>>>Substep 2.2: Creating temporary model dictionary and directories of results for '+ current_event  +" " + substep_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
        
        '''Temporary dict to store models created during the process'''
        dictionary_of_selected_models_for_events = {}
        
        '''Output directories for current event '''
        path_to_current_event_directory = os.path.join(ruta_directorio_raiz_salida,current_event)
        path_to_current_event_directory = os.path.join(path_to_current_event_directory,target)
        auxf.create_directory(path_to_current_event_directory)        
        
        '''Output report subdirectory for current event'''
        path_to_reports_current_event_directory = os.path.join(path_to_current_event_directory,auxf.decodify_using_enconding(config_parser.get(output_data_section,'output_directory_name_report'),char_encoding))            
        auxf.create_directory(path_to_reports_current_event_directory)       
        
        substep_finish_time = datetime.datetime.fromtimestamp(time.time())
        total_time_step_2+= substep_finish_time - substep_init_time
        repg.register_log([time_log],'>>Substep 2.2 Creating temporary model dictionary and directories of results for ' + current_event + ' elapsed time: ' + str(substep_finish_time - substep_init_time)  + "\n",'',char_encoding)
        repg.register_log([log_file],'>>>>>>Substep 2.2 ends '+ substep_finish_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
        
        '''Substep 2.3: Creating individual log individual for the current event'''
        substep_init_time = datetime.datetime.fromtimestamp(time.time())
        repg.register_log([log_file],'>>>>>>Substep 2.3: Creating individual log for '+ current_event  +" " + substep_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                
        current_event_individual_log = "Log_" + current_event + ".log"
        path_to_current_event_individual_log = os.path.join(path_to_current_event_directory,current_event_individual_log)
        
        try:            
            if((len(path_to_current_event_individual_log) >= max_lenght_output_names) or (len(log_file) >= max_lenght_output_names) or (len(path_to_reports_current_event_directory) + len(current_event+'****************') >= max_lenght_output_names)):
                print 'Maximum path length exceeded for event ' + current_event  
                raise FollowFlowException('')
                       
            array_of_paths_log_files = [log_file,path_to_current_event_individual_log]
                
            repg.register_log([path_to_current_event_individual_log],"####### Starts execution #####\n",0,char_encoding)
        
            repg.register_log(array_of_paths_log_files,'############################################################################################################################\n' + \
                     '###########################>>>>>> Working with '+  list_of_events[event_index] +' ' +\
                      datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') +'<<#################################' + "\n" + \
                     '############################################################################################################################\n'
                     ,'',char_encoding)
        
            substep_finish_time = datetime.datetime.fromtimestamp(time.time())
            total_time_step_2+= substep_finish_time - substep_init_time       
            repg.register_log([time_log],'>>Substep 2.3 Creating individual log for ' + current_event + ' elapsed time: ' + str(substep_finish_time - substep_init_time)  + "\n\n",'',char_encoding)
            repg.register_log([log_file],'>>>>>>Substep 2.3 ends '+ substep_finish_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
        
            '''Substep 2.4: Processing relevant and discarded user variables for the current event '''
            substep_init_time = datetime.datetime.fromtimestamp(time.time())                
            repg.register_log([log_file],'>>>>>>Substep 2.4: Getting mandatory and discarded variables by the user for '+ current_event  +" " + substep_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
        
            list_user_relevant_variables_actual_event = list_of_user_relevant_variables_event[event_index] 
            list_discarded_variables_current_event =  list_common_discarded_variables 
            list_discarded_variables_current_event+= list_of_discarded_variables_events[event_index] #lista variables descartadas especificas
        
            '''Cheking variables list for current event'''
            '''--Removing duplicate, if any, discarded variables'''
            list_discarded_variables_current_event = list(set(list_discarded_variables_current_event))
            list_discarded_variables_current_event = sorted(list_discarded_variables_current_event)
        
            '''--Removing target variable if one of the deleted variables'''        
            if(target in list_discarded_variables_current_event):
                list_discarded_variables_current_event.remove(target)

            '''--Removing duplicate, if any, relevant features'''
            list_user_relevant_variables_actual_event = list(set(list_user_relevant_variables_actual_event))
            list_user_relevant_variables_actual_event = sorted(list_user_relevant_variables_actual_event)

            '''--Removing discarded variable from important list variables'''
            for discarded_variable in list_discarded_variables_current_event:
                if discarded_variable in list_user_relevant_variables_actual_event:
                    list_user_relevant_variables_actual_event.remove(discarded_variable)
                        
            if(target in list_user_relevant_variables_actual_event):
                list_user_relevant_variables_actual_event.remove(target)
        
            repg.register_log(array_of_paths_log_files,'>>>>>> Variables discarded from de init \n','',char_encoding)
            for discarded_variable in list_discarded_variables_current_event:
                repg.register_log(array_of_paths_log_files,"\t\t->" + discarded_variable +"\n",'',char_encoding)            
        
            substep_finish_time = datetime.datetime.fromtimestamp(time.time())
            total_time_step_2+= substep_finish_time - substep_init_time          
            repg.register_log([time_log],'>>Substep 2.4 Getting mandatory and discarded variables for ' + current_event + ' elapsed time: ' + str(substep_finish_time - substep_init_time)  + "\n",'',char_encoding)
            repg.register_log([log_file],'>>>>>>Substep 2.4 ends '+ substep_finish_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
        
            '''Substep 2.5: Creating report template with basic data for the current_event'''
            substep_init_time = datetime.datetime.fromtimestamp(time.time())
            repg.register_log([log_file],'>>>>>>Substep 2.5: Creating report template for '+ current_event  +" " + substep_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
        
            report_data = repg.create_report_data_dict(current_event,percentil,target,list_discarded_variables_current_event,"'" + os.path.join(path_to_dir_auxiliary_files,"logo.jpg") + "'")
            
            substep_finish_time = datetime.datetime.fromtimestamp(time.time())                
            total_time_step_2+= substep_finish_time - substep_init_time
            repg.register_log([time_log],'>>Substep 2.5 Creating report template for ' + current_event + ' elapsed time: ' + str(substep_finish_time - substep_init_time)  + "\n",'',char_encoding)
            repg.register_log([log_file],'>>>>>>Substep 2.5 ends '+ substep_finish_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                
            repg.register_log([log_file],'>>>>>>Step 2 ends '+ step_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)        
            repg.register_log([time_log],'>>>>Step 2 - Creating models-event dict, events and getting models total elapsed time: ' + str(total_time_step_2)  + "\n\n",'',char_encoding)
        
        
            '''Step 3: Setting up dataset for current event '''        
            step_init_time = datetime.datetime.fromtimestamp(time.time())
            repg.register_log([time_log],'>>>>Step 3 starts:\n','',char_encoding)
        
            path_to_mif_output_directory = os.path.join(path_to_current_event_directory,auxf.decodify_using_enconding(config_parser.get(output_data_section,'output_directory_name_mif'),char_encoding))
            auxf.create_directory(path_to_mif_output_directory)
    
        
            if (maximum_number_of_observations_to_process != 0):
                repg.register_log(array_of_paths_log_files,'>>>>>>Substep 3.1 Separating default target (1) from the rest of data to obtain a total of ' + str(maximum_number_of_observations_to_process*2) + " registers (" + str(maximum_number_of_observations_to_process) +" catalogued and "+ str(maximum_number_of_observations_to_process) + ' not catalogued, if possible, for the event ' + current_event + ' ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
            else:
                repg.register_log(array_of_paths_log_files,'>>>>>>Substep 3.1 Separating default target (1) from the rest of data to obtain the maximum number of registers catalogued and not catalogued for the event ' + current_event + ' ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
        
            '''Substep 3.1 Splitting catalogued and non catalogued observations, limiting maximun number if desired '''        
            substep_init_time = datetime.datetime.fromtimestamp(time.time())
            repg.register_log([log_file],'>>>>>> Substep 3.1: Separating catalogued data from non-catalogued data '+ current_event  +" " + substep_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
            
            df_catalogued_data,df_datos_no_catalogados = auxf.split_dataset_catalogued_non_catalogued(vector_full_paths_input_files,name_variable_with_events,current_event,target,maximum_number_of_observations_to_process,label_non_cataloged,input_files_delimiter,[log_file], list_discarded_variables_current_event,char_encoding)           
                        
            existen_datos_catalogados = not df_catalogued_data.empty
            catalogar_no_validos = list(df_datos_no_catalogados.columns) == list(df_catalogued_data.columns) #si hay datos tendran las mismas columnas
        
            substep_finish_time = datetime.datetime.fromtimestamp(time.time())                
            repg.register_log([time_log],'>>Substep 3.1 Separating catalogued data from non-catalogued data and recodification of the target for ' + current_event + ' elapsed time: ' + str(substep_finish_time - substep_init_time)  + "\n",'',char_encoding)
            repg.register_log(array_of_paths_log_files,">> The label for the non catalogued data is: \n\t-> " +str(label_non_cataloged)+ "\n",'',char_encoding)
        
        
            if (not existen_datos_catalogados):                
                step_finish_time = datetime.datetime.fromtimestamp(time.time())
                repg.register_log(array_of_paths_log_files,'>>>>>> Creating models phase ends because the event ' + current_event + ' has not valid data (no catalogued data was found) '  + step_finish_time.strftime('%Y-%m-%d %H:%M:%S') + "\n\n",'',char_encoding)
                total_time_step_3 = step_finish_time - step_init_time
                repg.register_log([time_log],'>>>>Step 3 - Separating catalogued data and applying mutual information function: ' + str(total_time_step_3)  + "\n\n",'',char_encoding)
                repg.register_log([time_log],'>>>>Creating models phase ends(catalogued data was not found) - Processing events and getting models total elapsed time: ' + str(total_time_step_2 + total_time_step_3)  + "\n\n",'',char_encoding)
                report_data = repg.upddate_report_warning_info(report_data, current_event + ' has not valid data. Models for this event could not be calculated because catalogued data was not found. ')
                repg.create_report_current_model(report_data,[],auxiliary_directory_filename,path_to_reports_current_event_directory,char_encoding)
                raise FollowFlowException
            
            else:                        
                '''Inserting observation number'''
                df_catalogued_data[observation_number] = range(1,len(df_catalogued_data)+1)
                
                '''3.2 Obtaining statistics about current number of values for target feature'''
                substep_init_time = datetime.datetime.fromtimestamp(time.time())
                repg.register_log(array_of_paths_log_files,'>>>>>>Substep 3.2 Checking the number of different values for target (' + target + ') and event: ' + current_event + step_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n\n",'',char_encoding)
                
                report_data=repg.register_target_values_distribution(auxf.count_number_of_observations_by_target(df_catalogued_data, target),">>>>Substep 3.2.1 Distribution for the values of the CATALOGUED data for the target feature("+ target +"):\n", array_of_paths_log_files,report_data,char_encoding)
                
                if(catalogar_no_validos):
                    report_data = repg.register_target_values_distribution(auxf.count_number_of_observations_by_target(df_datos_no_catalogados, target),">>>>Substep 3.2.2 Distribution for the values of the NON-CATALOGUED data for the target feature("+ target +"):\n", array_of_paths_log_files,report_data,char_encoding,'None') 
                else:
                    repg.register_log(array_of_paths_log_files,">>>>Substep 3.2.2 INFO: No NON-CATALOGUED data for the target feature("+ target +") was found\n", '',char_encoding)                
                                         
                num_targets_evento = auxf.get_number_different_values_objective_target(df_catalogued_data[target].values)
                list_targets_current_event = sorted(list(set(df_catalogued_data[target].values)))
                target_recodified_values = list_targets_current_event
                solved_problem = auxf.check_number_of_values_objective_target(num_targets_evento)                                
                
                substep_finish_time = datetime.datetime.fromtimestamp(time.time())
                repg.register_log(array_of_paths_log_files,'>>>>>>Substep 3.2 ends ' + step_finish_time.strftime('%Y-%m-%d %H:%M:%S') + "\n\n",'',char_encoding)
                repg.register_log([time_log],'>>Substep 3.2 Checking all the different values for the target elapsed time: ' + str(substep_finish_time - substep_init_time)  + "\n",'',char_encoding)
                            
                if(solved_problem):#if only one target is available
                    step_finish_time = datetime.datetime.fromtimestamp(time.time())
                    repg.register_log(array_of_paths_log_files,'>>>>Creating models phase ends - because the event ' + current_event + ' has an only target. The classification problem is already solved '  + step_finish_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                    total_time_step_3 = step_finish_time - step_init_time
                    repg.register_log([time_log],'>>>>Step 3 - Separating catalogued data and applying mutual information function: ' + str(total_time_step_3)  + "\n\n",'',char_encoding)
                    repg.register_log([time_log],'>>>>Creating models phase ends(only one target) - Processing events and getting models total elapsed time: ' + str(step_finish_time - step_init_time)  + "\n\n",'',char_encoding)
                    report_data = repg.upddate_report_warning_info(report_data, current_event + ' has only one target. The classification problem is already solved ')                    
                    repg.create_report_current_model(report_data,[],auxiliary_directory_filename,path_to_reports_current_event_directory,char_encoding)
                    raise FollowFlowException
                
                else:#there is, at least, two known values for the target
                    
                    '''Creating path for generated data'''
                    path_to_directory_event_data = os.path.join(path_to_current_event_directory,'generated_data')            
                    auxf.create_directory(path_to_directory_event_data)
                    
                    if (validation_mode != 'True'):
                        #move observation number column to front
                        ordered_columns = list(df_catalogued_data.columns)
                        ordered_columns.remove(observation_number)
                        ordered_columns = [observation_number] + ordered_columns
                        
                        #save input data                        
                        df_catalogued_data = df_catalogued_data[ordered_columns]
                        df_catalogued_data = df_catalogued_data.sort_values(by=['obsnumber'])
                        df_catalogued_data.to_csv(os.path.join(path_to_directory_event_data,'numbered_' + current_event + '_data_'+ target +'.csv'),index=False,sep=",",encoding='utf-8')
                    
                    elif (validation_mode == 'True'):
                        '''Splitting data in Training-Test and Validation datasets if validation_mode is True'''
                        validation_division_percentaje = float(config_parser.get(testing_section,'validation_division_percentaje'))                                
                        df_catalogued_data,df_validation = auxf.split_train_test_datasets(df_catalogued_data,target,validation_division_percentaje)
                        
                        #move observation number column to front
                        ordered_columns = list(df_catalogued_data.columns)
                        ordered_columns.remove(observation_number)
                        ordered_columns = [observation_number] + ordered_columns
                        
                        #save input data                        
                        df_catalogued_data = df_catalogued_data[ordered_columns]
                        df_catalogued_data = df_catalogued_data.sort_values(by=['obsnumber'])
                        df_catalogued_data.to_csv(os.path.join(path_to_directory_event_data,'numbered_' + current_event + '_data_'+ target +'.csv'),index=False,sep=",",encoding='utf-8')
                        
                        #save validation data
                        nombre_fichero_datos_validacion = os.path.join(path_to_directory_event_data,target+'validation_data_'+current_event+'.csv')                        
                        df_validation = df_validation[ordered_columns]
                        
                        df_validation = df_validation.sort_values(by=['obsnumber'])
                        df_validation.to_csv(nombre_fichero_datos_validacion,sep=',',index=False,encoding='utf-8')
                        report_data = repg.register_target_values_distribution(auxf.count_number_of_observations_by_target(df_validation, target),">>>>Substep 3.2 Extra :Distribution for the values of VALIDATION data for the target feature("+ target +"):\n", array_of_paths_log_files,report_data,char_encoding,'Validation')
                        name_validation_data_directory =config_parser.get(output_data_section,'validation_data_directory_name')
                        ruta_directorio_datos_validacion = os.path.join(path_to_directory_prediction_models,name_validation_data_directory)
                        auxf.create_directory(ruta_directorio_datos_validacion)
                        shutil.move(nombre_fichero_datos_validacion,os.path.join(ruta_directorio_datos_validacion,target+'validation_data_'+current_event+'.csv'))
                    
                    '''Substep 3.3 Deleting empty and constant variables and updating dictionary with report template'''                    
                    substep_init_time = datetime.datetime.fromtimestamp(time.time())
                    repg.register_log(array_of_paths_log_files,'>>>>>> Substep 3.3 Deleting empty and constant features of the event ' + current_event + ' ' + substep_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                    
                    '''Substep 3.3.1 Deleting empty or constant variables for current event'''
                    list_valid_variables_for_models_current_event,list_empty_const_features_current_event = auxf.deleting_empty_and_constant_features(df_catalogued_data)
                    
                    '''Correcting list of relevant feature if any of the was erased for being constant or empty'''
                    report_data=repg.update_report_relevant_user_features(report_data,sorted(list_user_relevant_variables_actual_event))
                    for deleted_feature_emp_cte in list_empty_const_features_current_event:
                        if deleted_feature_emp_cte in list_user_relevant_variables_actual_event:
                            list_user_relevant_variables_actual_event.remove(deleted_feature_emp_cte)
                    
                    substep_finish_time = datetime.datetime.fromtimestamp(time.time())                
                    repg.register_log([time_log],'>>Substep 3.3.1 Deleting empty and constant variables for ' + current_event + ' elapsed time: ' + str(substep_finish_time - substep_init_time)  + "\n",'',char_encoding)
                    repg.register_log(array_of_paths_log_files,'>>>>>>Substep 3.3.1 ends' + substep_finish_time.strftime('%Y-%m-%d %H:%M:%S') + "\n\n",'',char_encoding)
                    
                    
                    '''Substep 3.3.2 Updating current event report'''
                    substep_init_time = datetime.datetime.fromtimestamp(time.time())
                    
                    report_data=repg.update_report_user_discarded_features(report_data,sorted(list_of_discarded_variables_events[event_index]))
                    report_data=repg.update_report_empty_constant_features(report_data,sorted(list_empty_const_features_current_event))
                    
                                        
                    substep_finish_time = datetime.datetime.fromtimestamp(time.time())                
                    repg.register_log([time_log],'>>Substep 3.3.2 Updating report with empty and constant variables elapsed time: ' + str(substep_finish_time - substep_init_time)  + "\n",'',char_encoding)
                    repg.register_log(array_of_paths_log_files,'>>Deleted empty or constant features of the event ' + current_event + ':  '+ str(list_empty_const_features_current_event) + ' ' + substep_finish_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                    repg.register_log(array_of_paths_log_files,'>>>>>>Substep 3.3.2 ends' + substep_finish_time.strftime('%Y-%m-%d %H:%M:%S') + "\n\n",'',char_encoding)
                                                                                
                    existence_possible_features_train_models = list_valid_variables_for_models_current_event != []       
                    
                    if(not existence_possible_features_train_models):
                        step_finish_time = datetime.datetime.fromtimestamp(time.time())
                        repg.register_log(array_of_paths_log_files,'>>>>>>Creating models phase ends - because the event ' + current_event + ' has not relevant features '+ str(list_valid_variables_for_models_current_event) +' ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n\n",'',char_encoding)
                        total_time_step_3 = step_finish_time - step_init_time
                        repg.register_log([time_log],'>>>>Step 3 - Separating catalogued data and applying mutual information function: ' + str(total_time_step_3)  + "\n\n",'',char_encoding)
                        repg.register_log([time_log],'>>>>Creating models phase ends(no relevant features) - Processing events and getting models total elapsed time: ' + str(step_finish_time - step_init_time)  + "\n\n",'',char_encoding)
                        raise FollowFlowException
                        
                    else:
                        
                        '''Substep 3.4 Filtering data I, Catalogued Data: Maintaining non-empty and non-constant variables'''
                        substep_init_time = datetime.datetime.fromtimestamp(time.time())
                                
                        df_catalogued_data = df_catalogued_data.loc[:, list_valid_variables_for_models_current_event]                                                
    
                        repg.register_log(array_of_paths_log_files,'>>>>Substep 3.4 Features not empty and not constant:\n','',char_encoding)
                        for feat_not_emp_not_cte in list_valid_variables_for_models_current_event:
                            repg.register_log(array_of_paths_log_files,"\t\t->" + feat_not_emp_not_cte +"\n",'',char_encoding)
                        repg.register_log(array_of_paths_log_files,datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + '\n','',char_encoding)
                        
                        substep_finish_time = datetime.datetime.fromtimestamp(time.time())                
                        repg.register_log([time_log],'>>Substep 3.4 Filtering initial dataset with non-empty and non-constant variables: ' + str(substep_finish_time - substep_init_time)  + "\n",'',char_encoding)
                        repg.register_log(array_of_paths_log_files,'>>>> Substep 3.4 ends' + str(substep_finish_time - substep_init_time) + "\n",'',char_encoding)
                        
                        
                        '''Substep 3.5 Getting available features'''
                        substep_init_time = datetime.datetime.fromtimestamp(time.time())
                        repg.register_log(array_of_paths_log_files,'>>>>>>Substep 3.5 Recodifying features of the event ' + current_event + ' ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
        
                        #get the list of features to work with                        
                        lista_variables_utilizables = list(df_catalogued_data.columns)
                        
                        
                        substep_finish_time = datetime.datetime.fromtimestamp(time.time())
                        repg.register_log([time_log],'>>Substep 3.5  Getting available features for ' + current_event + ' total elapsed time: ' + str(substep_finish_time - substep_init_time)  + "\n",'',char_encoding)
                        repg.register_log(array_of_paths_log_files,'>>>>Features that are available:\n','',char_encoding)
                        for feat_reco in lista_variables_utilizables:
                            repg.register_log(array_of_paths_log_files,"\t\t->" + feat_reco + "\n",'',char_encoding)
                        repg.register_log(array_of_paths_log_files,'>>>>>>Substep 3.5 ends' + substep_finish_time.strftime('%Y-%m-%d %H:%M:%S') + "\n\n",'',char_encoding) 
                
                
                        ''' Substep 3.6 Applying Mutual Information Function'''
                        substep_init_time = datetime.datetime.fromtimestamp(time.time())                        
                        
                        useless_features = [observation_number]
                        lista_features_scores = metr.apply_mutual_information_function_to_current_features_using_percentil(current_event,target,percentil,path_to_mif_output_directory,df_catalogued_data,list_user_relevant_variables_actual_event,useless_features,log_file,char_encoding)
                        
                        substep_finish_time = datetime.datetime.fromtimestamp(time.time())                
                        repg.register_log([time_log],'>>Substep 3.6 Calculating mutual information function elapsed time: ' + str(substep_finish_time - substep_init_time)  + "\n",'',char_encoding)
                        repg.register_log(array_of_paths_log_files,'>>>>Substep 3.6 Relevant features obtained after MIF process:\n','',char_encoding)
                        for rel_feat in lista_features_scores:
                            repg.register_log(array_of_paths_log_files,"\t\t-> [" + rel_feat[0] + ',' +  str(rel_feat[1])  +"]\n",'',char_encoding)
                        repg.register_log(array_of_paths_log_files,'>>>>Substep 3.6 ends' + str(list_valid_variables_for_models_current_event) + substep_finish_time.strftime('%Y-%m-%d %H:%M:%S') +"\n",'',char_encoding)
                                                                        
                                                
                        '''Substep 3.7 Filtering variables exclusively to train the models and the full list of variables used in the process'''
                        
                        substep_init_time = datetime.datetime.fromtimestamp(time.time())  
                        repg.register_log(array_of_paths_log_files,'>>>>>>Substep 3.7 Filtering variables exclusively to train the models and the full list of variables used in the process ' + current_event + ' ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                        
                        features_discarded_mif_process = [target] + useless_features #removing target and observation_number(if present) in relevant variables
                        features_used_during_process,dic_relevantfeat_scores = auxf.get_features_to_train_models(lista_features_scores,features_discarded_mif_process)
                        features_used_during_process = list(set(features_used_during_process))
                        
                        report_data=repg.update_report_training_models_features(report_data,dic_relevantfeat_scores)                        
                        report_data = repg.update_report_full_list_features_used_in_process(report_data,features_used_during_process)
                        
                        substep_finish_time = datetime.datetime.fromtimestamp(time.time())                
                        repg.register_log([time_log],'>>Substep 3.7  Filtering variables exclusively to train the models and the full list of variables used in the process elapsed time: ' + str(substep_finish_time - substep_init_time)  + "\n",'',char_encoding)
        
                        if(features_used_during_process == []): #process ends if no relevant variables were found
                            step_finish_time = datetime.datetime.fromtimestamp(time.time())
                            repg.register_log(array_of_paths_log_files,'>>>>>>Substep 3.7 ends because the event ' + current_event + ' has not relevant features (all scores ar under computed percentil value). The event is excluded'  + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n\n",'',char_encoding)
                            total_time_step_3 = step_finish_time - step_init_time
                            repg.register_log([time_log],'>>>>Step 3 - Separating catalogued data and applying mutual information function: ' + str(total_time_step_3)  + "\n\n",'',char_encoding)
                            repg.register_log([time_log],'>>>>Creating models phase ends(relevant features were not found) - Processing events and getting models total elapsed time: ' + str(step_finish_time - step_init_time)  + "\n\n",'',char_encoding)
                            report_data = repg.upddate_report_warning_info(report_data, current_event + ' has not relevant features (all scores ar under computed percentil value). The event is excluded')                            
                            repg.create_report_current_model(report_data,[],auxiliary_directory_filename,path_to_reports_current_event_directory,char_encoding)
                            raise FollowFlowException
                            
                        else: #Spliting catalogued data in train data and test data
                            repg.register_log(array_of_paths_log_files,'>>>>Step 3.7 Relevant features after using MIF scorings inside the percentage specified \n','',char_encoding)
                            for rel_feat in features_used_during_process:
                                repg.register_log(array_of_paths_log_files,"\t\t->"+ rel_feat +"\n",'',char_encoding)    
                            repg.register_log(array_of_paths_log_files,'>>>>Step 3.7 ends' + substep_finish_time.strftime('%Y-%m-%d %H:%M:%S') +"\n",'',char_encoding)
                            
                            '''Substep 3.8 Filtering data II: Filtering dataset using relevant features (those to train the models + target + necessary ones)'''
                            substep_init_time = datetime.datetime.fromtimestamp(time.time())
                            repg.register_log(array_of_paths_log_files,'>>>>>>Step 3.8 Filtering relevant data (those to train the models + target + necessary ones) to apply learning models to event ' +  current_event +' '+ substep_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                            
                                
                            '''Copy the features that will be used TO TRAIN the models'''
                            features_used_train_models = []
                            for relevant_feature in features_used_during_process:                                
                                features_used_train_models.append(relevant_feature)
                                                            

                            '''Adding to the full list of variables used in the process the ones that were discarded to obtain the features to train models'''
                            for discarded_feature in features_discarded_mif_process:
                                features_used_during_process.append(discarded_feature)

                            '''features_used_during_process.append(observation_number) #necesitamos introducirlo para determinar aquellas observaciones mal clasificadas al predecir'''
 
                            
                            repg.register_log(array_of_paths_log_files,'>>>>Step 3.8 Relevant features to filter catalogued/non-cataloged data:\n','',char_encoding)
                            for rel_feat in features_used_during_process:
                                repg.register_log(array_of_paths_log_files,"\t\t->" + rel_feat +"\n",'',char_encoding)
                            
                            repg.register_log(array_of_paths_log_files,'>>>>Step 3.8 Relevant features to operate with the models:\n','',char_encoding)
                            for feat_operate in features_used_train_models:
                                repg.register_log(array_of_paths_log_files,'\t\t->'+ feat_operate +'\n','',char_encoding)
                                                        
                            '''Filtering dataset of catalogued data using relevant variables (those to train the models + target + necessary ones)'''                                                            
                            df_catalogued_data = df_catalogued_data.loc[:, features_used_during_process]                            
                            
                            substep_finish_time = datetime.datetime.fromtimestamp(time.time())                
                            repg.register_log([time_log],'>>Substep 3.8 Filtering data using MIF variables elapsed time: ' + str(substep_finish_time - substep_init_time) + "\n",'',char_encoding)
                            
                            repg.register_log(array_of_paths_log_files,'>>>>Step 3.8 Catalogued data has: ' + str(df_catalogued_data.shape[0]) + " observations and " + str(df_catalogued_data.shape[1]) + " features. " +datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                            
                            repg.register_log(array_of_paths_log_files,'>>>>Step 3.8 INFO: At this point catalogued data and not catalogued data has re-codified values and only the relevant features' + current_event +' ' +datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                            repg.register_log(array_of_paths_log_files,'>>>>>>Step 3.8 ends ' + substep_finish_time.strftime('%Y-%m-%d %H:%M:%S') + "\n\n",'',char_encoding)
                            
                            step_finish_time = datetime.datetime.fromtimestamp(time.time())              
                            total_time_step_3 = step_finish_time - step_init_time
                            repg.register_log([time_log],'>>>>Step 3 - Separating catalogued data and applying mutual information function: ' + str(total_time_step_3)  + "\n\n",'',char_encoding)
                            
                                                        
                            '''Step 4 Creating models for the catalogued data of the event'''
                            repg.register_log([time_log],'>>>>Step 4 starts:\n','',char_encoding)
                            step_init_time = datetime.datetime.fromtimestamp(time.time())
                            
                            '''Step 4.0 Creating subdatasets to select parameters if number of observations in train+test datasets exceed 10000 observations'''
                            y = df_catalogued_data[target]
                            if (len(df_catalogued_data)>5000):                                
                                df_seleccion_parametros, X_test, y_train, y_test = train_test_split(df_catalogued_data[features_used_train_models], y, stratify=y, test_size=0.75)
                                df_seleccion_parametros[target] = y_train                                
                            else:
                                df_seleccion_parametros=df_catalogued_data
                            
                           

                            '''Substep 4.1 Spliting catalogued dataset in training and test datasets'''
                            substep_init_time = datetime.datetime.fromtimestamp(time.time())                                                        
                            repg.register_log(array_of_paths_log_files,'>>>>>>Substep 4.1 Splitting train('+str(data_division_percentaje*100) +'%) and test('+str((1-data_division_percentaje)*100)+'%) data for '+ current_event + ' ' + substep_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                                                            
                            df_train,df_test = auxf.split_train_test_datasets(df_catalogued_data,target,data_division_percentaje)                                                        
                            
                
                            '''Register target distribution values for the train and test datasets'''       
                            report_data = repg.register_target_values_distribution(auxf.count_number_of_observations_by_target(df_train, target),">>>>Substep 4.1 Distribution for the values of TRAIN data for the target feature("+ target +"):\n", array_of_paths_log_files,report_data,char_encoding,'Train')
                            report_data = repg.register_target_values_distribution(auxf.count_number_of_observations_by_target(df_test, target),">>>>Substep 4.1 Distribution for the values of the TEST data for target feature("+ target +"):\n", array_of_paths_log_files,report_data,char_encoding,'Test')  
                            
                            substep_finish_time = datetime.datetime.fromtimestamp(time.time())
                            repg.register_log(array_of_paths_log_files,'>>>>>>Substep 4.1 ends ' + substep_finish_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                            repg.register_log([time_log],'>>Substep 4.1 Splitting train and test data elapsed time: ' + str(substep_finish_time - substep_init_time) + "\n",'',char_encoding)
                            
                            '''Substep 4.2 Creating and applying supervised models'''
                            substep_init_time = datetime.datetime.fromtimestamp(time.time())                                    
                            
                            list_available_supervised_models = supervised_models_dictionary.values()                            
                            list_non_supervised_available_models = unsupervised_models_dictionary.values()
                                                        
                            list_available_supervised_models.remove('All')
                            list_non_supervised_available_models.remove('All') 
                            list_available_models_for_the_event = list_available_supervised_models + list_non_supervised_available_models
                            list_available_models_for_the_event = sorted(list_available_models_for_the_event)
                                                                                                                                                                        
                            list_available_learnings_current_event = list_available_learnings[event_index]
                    
                    
                            '''Retrieving weighted matrix from conf.ini'''
                            if(default_matrix == 'True'):#user specified                                
                                w = auxf.compute_matrix_weights(len(target_recodified_values))
                            else:
                                w = ast.literal_eval(config_parser.get(input_data_section,'matrix_of_weights_fp_fn'))
                                try:        
                                    auxf.check_matrix_weights(w,len(target_recodified_values))
                                except ValueError as e:
                                    raise ValueError(e)                                                        
                                    
                            if(list_available_learnings_current_event != available_learnings_dict[2]):# if supervised or both
                                repg.register_log(array_of_paths_log_files,'>>>>Substep 4.1 ends ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n\n" + \
                                                   '>>>>>>Substep 4.2 Applying '+ str(available_learnings_dict[1]) + ' learning to '+  current_event +' ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",\
                                       '',char_encoding)
                                
                                '''Creating output directory for supervised learning'''
                                current_learning = available_learnings_dict[1]#'supervised'
                                repg.register_log([time_log],'>>> Supervised <<< \n','',char_encoding)
                                path_to_directory_supervised_models = os.path.join(path_to_current_event_directory,current_learning)
                                auxf.create_directory(path_to_directory_supervised_models) 
                                
                                '''Substep 4.2.1 Getting supervised models'''
                                subsubstep_init_time = datetime.datetime.fromtimestamp(time.time())
                                if(available_learnings_dict[1] == list_available_learnings_current_event):
                                    list_supervised_models_current_event = list_of_mod[event_index]    
                                    list_supervised_models_current_event = sorted(list_supervised_models_current_event)
                                else:
                                    list_supervised_models_current_event = list_of_mod[event_index][0]
                                    list_supervised_models_current_event = sorted(list_supervised_models_current_event)                
                
                                subsubstep_finish_time = datetime.datetime.fromtimestamp(time.time())
                                repg.register_log([time_log],'>>Substep 4.2.1 Getting supervised models elapsed time: ' + str(subsubstep_finish_time - subsubstep_init_time) + "\n",'',char_encoding)
                                repg.register_log(array_of_paths_log_files,'>>>>Substep 4.2.1 It will be applied the next supervised models:\n\t-> ' + str(list_supervised_models_current_event) + "\n",'',char_encoding)                                
                                for supervised_model_name in list_supervised_models_current_event:
                                    print 'Computing... ' + supervised_model_name
                                    try:
                                        '''Subsubstep 4.2.1.1 Selecting optimal parameters for the model'''
                                        model_init_time = datetime.datetime.fromtimestamp(time.time())
                                        subsubstep_init_time = datetime.datetime.fromtimestamp(time.time())
                                        report_data = repg.add_model_to_report(report_data,supervised_model_name)
                                        report_data = repg.update_report_model_time(report_data,supervised_model_name,'time_model_init',datetime.datetime.fromtimestamp(time.time()))
                                        path_to_supervised_model_directory = os.path.join(path_to_directory_supervised_models,supervised_model_name)
                                        auxf.create_directory(path_to_supervised_model_directory) 
                                                                        
                                        repg.register_log(array_of_paths_log_files,'>>>>Substep 4.2.1.1 Selecting the optimal parameters for the model |||' + supervised_model_name +'||| '+ datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                                            
                                        report_data = repg.update_report_model_time(report_data,supervised_model_name,'time_sel_init',datetime.datetime.fromtimestamp(time.time()))
                                        
                                        parameters_current_model,report_data = spar.select_optimal_parameters_current_model(supervised_model_name,supervised_models_dictionary,unsupervised_models_dictionary,df_seleccion_parametros,features_used_train_models,target,array_of_paths_log_files,path_to_supervised_model_directory,report_data,lc_output_format,char_encoding,max_pars)  
                                        report_data = repg.update_report_model_time(report_data,supervised_model_name,'time_sel_finish',datetime.datetime.fromtimestamp(time.time()))
                            
                                        subsubstep_finish_time = datetime.datetime.fromtimestamp(time.time())
                                        repg.register_log(array_of_paths_log_files,'>>>>Substep 4.2.1.1 ends ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)                                                       
                                        repg.register_log([time_log],'>>Substep 4.2.1.1 Selecting the optimal parameters for the model ' + supervised_model_name + ' elapsed time: ' + str(subsubstep_finish_time - subsubstep_init_time) + "\n",'',char_encoding)
                                        
                                        
                                        '''Subsubstep 4.2.1.2 Training the model using optimal parameters'''
                                        subsubstep_init_time = datetime.datetime.fromtimestamp(time.time())
                                        repg.register_log(array_of_paths_log_files,'>>>>Substep 4.2.1.2 Training the model '+ supervised_model_name  + ' '+ datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                                                              
                                        report_data = repg.update_report_model_time(report_data,supervised_model_name,'time_train_init',datetime.datetime.fromtimestamp(time.time()))
                                        current_trained_model = supe.get_trained_model(supervised_model_name,df_train,features_used_train_models,target,parameters_current_model,supervised_models_dictionary)
                                        report_data = repg.update_report_model_time(report_data,supervised_model_name,'time_train_finish',datetime.datetime.fromtimestamp(time.time()))
                                                                                
                                        repg.register_log(array_of_paths_log_files,'>>>>Substep 4.2.1.2 Charateristics:\n\t '+ str(current_trained_model) + ' '+ datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                                        repg.register_log(array_of_paths_log_files,'>>>>Substep 4.2.1.2 ends '+ datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                                        subsubstep_finish_time = datetime.datetime.fromtimestamp(time.time())
                                        repg.register_log([time_log],'>>Substep 4.2.1.2 Training the model ' + supervised_model_name + ' elapsed time: ' + str(subsubstep_finish_time - subsubstep_init_time) + "\n",'',char_encoding)
                                        
                                        
                                        '''Subsubstep 4.2.1.3 Obtaining model accuracy'''
                                        subsubstep_init_time = datetime.datetime.fromtimestamp(time.time())
                                        repg.register_log(array_of_paths_log_files,'>>>>Substep 4.2.1.3 Obtaining accuracy for the model ' + supervised_model_name + ' ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                                                                                
                                        accuracy_current_model= supe.calculate_model_accuray(current_trained_model,df_test,features_used_train_models,target,0)
                                        report_data = repg.actualizar_accuracy_modelo(report_data,supervised_model_name,accuracy_current_model)
                                        
                                        numpy_array_accuracy_current_model=np.array([accuracy_current_model])
                                        repg.register_log(array_of_paths_log_files,'>>>>Substep 4.2.1.3 Accuracy obtained: ' + str(accuracy_current_model) + "\n",'',char_encoding)                                                                                
        
                                        subsubstep_finish_time = datetime.datetime.fromtimestamp(time.time())
                                        repg.register_log(array_of_paths_log_files,'>>>>Substep 4.2.1.3 ends '+ datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                                        repg.register_log([time_log],'>>Substep 4.2.1.3 Obtaining accuracy for the model ' + supervised_model_name + ' elapsed time: ' + str(subsubstep_finish_time - subsubstep_init_time) + "\n",'',char_encoding)
                                        
                                        
                                        '''Subsubstep 4.2.1.4 Obtaining learning curve for the model'''
                                        subsubstep_init_time = datetime.datetime.fromtimestamp(time.time())
                                        repg.register_log(array_of_paths_log_files,'>>>>Substep 4.2.1.4 Obtaining learning curve for the model ' + supervised_model_name + ' ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                                        
                                        learning_curve_current_model=supe.initialize_model(supervised_model_name,parameters_current_model,supervised_models_dictionary)
                                        lista_scorings = metr.get_all_available_scoring_methods()
                                        cv_iterators = metr.get_all_available_cross_validation_iterators(5,0.2)
                                        train_sizes_def=np.linspace(.1,1.0,10)
                                        for indice_sc in range(len(lista_scorings)):                                            
                                            score_curve = lista_scorings[indice_sc]
                                            score_name = score_curve[0]                                                
                                            for index_cv_it in range(len(cv_iterators)):                                                    
                                                cv_iterator_name = cv_iterators[index_cv_it][0]
                                                cv_it = cv_iterators[index_cv_it][1]
                                                train_sizes, train_scores, test_scores = metr.compute_learning_curve(learning_curve_current_model,df_train[features_used_train_models],df_train[target],score_curve,cv_it,train_sizes=train_sizes_def)
                                                                                        
                                                name_of_the_curve = "Learning curve for It_" + str(cv_iterator_name) + "_Sc_"+ score_name + "_" + supervised_model_name + ".png"
                                                full_path_to_learning_curve_file = os.path.join(path_to_supervised_model_directory,name_of_the_curve)                                                    
                                                train_scores_mean,test_scores_mean = metr.save_learning_curve(train_sizes, train_scores, test_scores, "Learning curve(" +str(cv_iterator_name) + " and " +score_name +") for "+ supervised_model_name,full_path_to_learning_curve_file,lc_output_format)
                                                report_data = repg.update_model_feature(report_data,supervised_model_name,'lc_'+str(indice_sc)+"_"+str(index_cv_it),full_path_to_learning_curve_file)
                            
                                        subsubstep_finish_time = datetime.datetime.fromtimestamp(time.time())
                                        repg.register_log([time_log],'>>Substep 4.2.1.4 Calculating learning curve for the model ' + supervised_model_name + ' elapsed time: ' + str(subsubstep_finish_time - subsubstep_init_time) + "\n",'',char_encoding)
                                        repg.register_log(array_of_paths_log_files,'>>>>Substep 4.2.1.4 ends ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                                        
                                        
                                        '''Subsubstep 4.2.1.5 Obtaining confusion matrices and decision index for the model'''
                                        subsubstep_init_time = datetime.datetime.fromtimestamp(time.time())
                                        repg.register_log(array_of_paths_log_files,'>>>>Substep 4.2.1.5 Generating confusion matrix and decision index for '+ supervised_model_name  + ' '+ datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                            
                                        confusion_matrix_delimiter =','                                        
                                        train_true=df_train[target]
                                        test_true=df_test[target]
                        
                                        train_predictions_model=current_trained_model.predict(df_train[features_used_train_models])
                                        test_predictions_model=current_trained_model.predict(df_test[features_used_train_models])
                    
                                        confusion_matrix_model_train=metr.get_confusion_matrix(train_true,train_predictions_model,target_recodified_values)
                                        tr_precisions,tr_recalls,tr_specificity,tr_f1_score,tr_mcc,confusion_tables_by_class_train=metr.metrics_cm(confusion_matrix_model_train)
                                        confusion_matrix_name_train='confusion_matrix'+ supervised_model_name +'_train'                                        
                                        
                                        matrix_summary_clasifications_train = metr.confusion_matrix_gather_feature_incorrect_values(df_train[features_used_during_process], train_true, train_predictions_model, observation_number)
                                        metr.confusion_matrix_to_csv(matrix_summary_clasifications_train,list_targets_current_event, os.path.join(path_to_supervised_model_directory,'confusion_matrix_incorrect_classifications_train' + "." + 'csv'))

                                        confusion_matrix_model_test=metr.get_confusion_matrix(test_true,test_predictions_model,target_recodified_values)                                        
                                        
                                        #getting metrics
                                        test_precisions,test_recalls,test_specificity,test_f1_score,test_mcc,confusion_tables_by_class_test=metr.metrics_cm(confusion_matrix_model_test)

                                        #compute macro and micro avgs                                        
                                        precision_macro_avg_test,recall_macro_avg_test,specificity_macro_avg_test,f1_score_macro_avg_test,mcc_macro_avg_test,report_data = metr.compute_macro_avg_values_of_metrics(list_names_metrics,test_precisions,test_recalls,test_specificity,test_f1_score,test_mcc,target_recodified_values,report_data,supervised_model_name)                                        
                                        precision_micro_avg_test,recall_micro_avg_test,specificity_micro_avg_test,f1_score_micro_avg_test,mcc_micro_avg_test,report_data = metr.compute_micro_avg_values_of_metrics(list_names_metrics,confusion_tables_by_class_test,target_recodified_values,report_data,supervised_model_name)                                        
                                        
                                        #get incorrect classifications
                                        matrix_summary_clasifications_test = metr.confusion_matrix_gather_feature_incorrect_values(df_test[features_used_during_process], test_true, test_predictions_model, observation_number)
                                        metr.confusion_matrix_to_csv(matrix_summary_clasifications_test,list_targets_current_event,os.path.join(path_to_supervised_model_directory,'confusion_matrix_incorrect_classifications_test' + "." + 'csv'))
                                        repg.register_log(array_of_paths_log_files,'>>>>Confusion matrix incorrect classifications \n','',char_encoding)                                        
                                        
                                        '''Obtaining decision index for the current model'''
                                        #ts=train_sizes,ss=sample_scores,cvs=cv_scores,cm=confusion_matrix,w=weights,ac=accuracy                                        
                                        try:
                                            if(main_metric == 'mcc'):                                                
                                                decision_index_current_model=auxf.weight_for_decision_function(train_sizes,train_scores_mean,test_scores_mean,confusion_matrix_model_test,w,mcc_micro_avg_test)
                                            else:                                                
                                                decision_index_current_model=auxf.weight_for_decision_function(train_sizes,train_scores_mean,test_scores_mean,confusion_matrix_model_test,w,accuracy_current_model)
                                        except Exception as e:
                                            raise Exception(' the decision index has no value due to an internal error along the process')
                                        repg.update_report_current_model_decision_index(report_data,supervised_model_name,str(decision_index_current_model))
                        
                                        '''Saving confusion matrices to file'''
                                        repg.save_data_to_file(confusion_matrix_model_train.astype(int),path_to_supervised_model_directory,confusion_matrix_name_train,confusion_matrix_delimiter,'numpy')
                                        metr.save_confusion_matrix(confusion_matrix_model_train,target_recodified_values,os.path.join(path_to_supervised_model_directory,confusion_matrix_name_train),lc_output_format)
                                        report_data = repg.update_model_feature(report_data,supervised_model_name,"Confussion_matrix_train_path",os.path.join(path_to_supervised_model_directory,confusion_matrix_name_train + "." + lc_output_format))
                                        repg.register_log(array_of_paths_log_files,'>>>>Substep 4.2.1.5 Confusion matrix: train \n '+ str(confusion_matrix_model_train.tolist())  + ' '+ datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                                
                                        confusion_matrix_name_test='confusion_matrix_'+supervised_model_name+'_test'
                                        repg.save_data_to_file(confusion_matrix_model_test.astype(int),path_to_supervised_model_directory,confusion_matrix_name_test,confusion_matrix_delimiter,'numpy')
                                        metr.save_confusion_matrix(confusion_matrix_model_test,target_recodified_values,os.path.join(path_to_supervised_model_directory,confusion_matrix_name_test),lc_output_format)
                                        report_data = repg.update_model_feature(report_data,supervised_model_name,"Confussion_matrix_test_path",os.path.join(path_to_supervised_model_directory,confusion_matrix_name_test + "." + lc_output_format))
                                        repg.register_log(array_of_paths_log_files,'>>>>Substep 4.2.1.5 Confusion matrix test:\n '+ str(confusion_matrix_model_test.tolist())  + ' '+ datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                                                                    
                                        subsubstep_finish_time = datetime.datetime.fromtimestamp(time.time())
                                        repg.register_log([time_log],'>>Substep 4.2.1.5 Generating confusion matriz and decision index for the model ' + supervised_model_name + ' elapsed time: ' + str(subsubstep_finish_time - subsubstep_init_time) + "\n",'',char_encoding)
                                        repg.register_log(array_of_paths_log_files,'>>>>Step 4.2.1.5 ends ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                                        
                                        '''Subsubstep 4.2.1.6 Generating model pkl'''
                                        subsubstep_init_time = datetime.datetime.fromtimestamp(time.time())
                                        repg.register_log(array_of_paths_log_files,'>>>>Step 4.2.1.6 Generating model pkl for '+ supervised_model_name  + ' ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                                        
                                        repg.save_data_to_file(numpy_array_accuracy_current_model,path_to_supervised_model_directory, 'informe_accuracy_'+supervised_model_name, output_files_delimiter,'txt')
                                        path_to_store_current_generated_model = supe.save_current_model_to_file(current_trained_model, path_to_supervised_model_directory,supervised_model_name +'.pkl')                                        
                                        dictionary_of_selected_models_for_events[supervised_model_name] = [current_learning,path_to_store_current_generated_model,{},features_used_train_models]
                                        
                                        subsubstep_finish_time = datetime.datetime.fromtimestamp(time.time())
                                        repg.register_log([time_log],'>>Substep 4.2.1.6 Generating model pkl for ' + supervised_model_name + ' elapsed time: ' + str(subsubstep_finish_time - subsubstep_init_time) + "\n",'',char_encoding)
                                        repg.register_log(array_of_paths_log_files,'>>>>Step 4.2.1.6 ends ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)        
                                        
                                                                                    
                                    except Exception as e:
                                        if supervised_model_name in report_data:
                                            del report_data[supervised_model_name]
                                            auxf.delete_directory_with_content(path_to_supervised_model_directory)
                                        repg.register_log([errors_file],'\n>>'+supervised_model_name +' model was not created for the event '+ current_event + ' because ' + auxf.decodify_using_enconding(str(e),char_encoding) + ' ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n\n",'',char_encoding)
                                        repg.register_log(array_of_paths_log_files,'>> An error has happened check error log #######' + "\n",'',char_encoding)
                                        error_trace = "Full trace:\n" + str(traceback.format_exc())
                                        repg.register_log([errors_file], error_trace,'',char_encoding)                                        
                                    
                                    model_finish_time = datetime.datetime.fromtimestamp(time.time())
                                    repg.register_log([time_log],'>>Substep Creation of model '+ supervised_model_name +' total elapsed time: ' + str(model_finish_time - model_init_time) + "\n",'',char_encoding)
                                    repg.register_log(array_of_paths_log_files,'>>>>>>Substep 4.2 ends for model '+ supervised_model_name +' ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n\n",'',char_encoding)
                                    
                            substep_finish_time = datetime.datetime.fromtimestamp(time.time())
                            repg.register_log([time_log],'>>Substep 4.2 Creating and applying models for supervised learning total elapsed time: ' + str(substep_finish_time - substep_init_time) + "\n",'',char_encoding)
                            
                            
                            '''Substep 4.3 Creating and applying unsupervised models'''
                            substep_init = datetime.datetime.fromtimestamp(time.time())
                            subsubstep_init_time = datetime.datetime.fromtimestamp(time.time())
                            if(list_available_learnings_current_event != available_learnings_dict[1]):# if unsupervised or both
                                repg.register_log(array_of_paths_log_files,'>>>>>> Step 4.3 Applying unsupervised learning to '+  current_event +' ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                                
                                
                                '''Creating output directory for unsupervised learning'''
                                current_learning = available_learnings_dict[2]#'unsupervised'
                                repg.register_log([time_log],'>>> Unsupervised <<< \n','',char_encoding)
                                path_to_unsupervised_directory = os.path.join(path_to_current_event_directory,current_learning)
                                auxf.create_directory(path_to_unsupervised_directory)
                                if(available_learnings_dict[2] == list_available_learnings_current_event):
                                    list_unsupervised_models_current_event = list_of_mod[event_index]    
                                    list_unsupervised_models_current_event = sorted(list_unsupervised_models_current_event)
                                else:
                                    list_unsupervised_models_current_event = list_of_mod[event_index][1]
                                    list_unsupervised_models_current_event = sorted(list_unsupervised_models_current_event)                                
                                
                                subsubstep_finish_time = datetime.datetime.fromtimestamp(time.time())
                                repg.register_log([time_log],'>>Substep 4.3.1 Getting unsupervised models elapsed time: ' + str(subsubstep_finish_time - subsubstep_init_time) + "\n",'',char_encoding)              
                                repg.register_log(array_of_paths_log_files,'>>>> Step 4.3.1 It will be applied the next unsupervised models ' + str(list_unsupervised_models_current_event) + "\n",'',char_encoding)
                
                                for unsupervised_model_name in list_unsupervised_models_current_event:
                                    print 'Computing... ' + unsupervised_model_name
                                    try:
                                        '''Subsubstep 4.3.1.1 Getting optimal parameters unsupervised model'''
                                        model_init_time = datetime.datetime.fromtimestamp(time.time())                                    
                                        subsubstep_init_time = datetime.datetime.fromtimestamp(time.time())
                                        
                                        '''Creating output directory for current unsupervised model'''
                                        report_data = repg.add_model_to_report(report_data,unsupervised_model_name)
                                        report_data = repg.update_report_model_time(report_data,unsupervised_model_name,'time_model_init',datetime.datetime.fromtimestamp(time.time()))
                                        path_to_current_unsupervised_model_output_directory = os.path.join(path_to_unsupervised_directory,unsupervised_model_name)
                                        auxf.create_directory(path_to_current_unsupervised_model_output_directory)
                                        
                            
                                        repg.register_log(array_of_paths_log_files,'>>>>Substep 4.3.1.1 Selecting the optimal parameters for the model |||' + unsupervised_model_name +'||| '+ datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                                        
                                        report_data = repg.update_report_model_time(report_data,unsupervised_model_name,'time_sel_init',datetime.datetime.fromtimestamp(time.time()))
                                        parameters_current_model,report_data = spar.select_optimal_parameters_current_model(unsupervised_model_name,supervised_models_dictionary,unsupervised_models_dictionary,df_catalogued_data,features_used_train_models,target,array_of_paths_log_files,path_to_current_unsupervised_model_output_directory,report_data,lc_output_format,char_encoding,max_pars)                                        
                                        report_data = repg.update_report_model_time(report_data,unsupervised_model_name,'time_sel_finish',datetime.datetime.fromtimestamp(time.time()))
                                        
                                        subsubstep_finish_time = datetime.datetime.fromtimestamp(time.time())
                                        repg.register_log([time_log],'>>Substep 4.3.1.1 Selecting optimal parameters for the model ' + unsupervised_model_name + ' elapsed time: ' + str(subsubstep_finish_time - subsubstep_init_time) + "\n",'',char_encoding)
                                        repg.register_log(array_of_paths_log_files,'>>>>Substep 4.3.1.1 ends ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                                        
                                        '''Subsubstep 4.3.1.2 Training the model using optimal parameters'''
                                        subsubstep_init_time = datetime.datetime.fromtimestamp(time.time())
                                        repg.register_log(array_of_paths_log_files,'>>>>Substep 4.3.1.2 Training the model '+ unsupervised_model_name  + ' '+ datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                                        
                                        report_data = repg.update_report_model_time(report_data,unsupervised_model_name,'time_train_init',datetime.datetime.fromtimestamp(time.time()))
                                        current_trained_model = unsu.create_trained_model(unsupervised_model_name,df_catalogued_data,features_used_train_models,target,parameters_current_model,unsupervised_models_dictionary)
                                        report_data = repg.update_report_model_time(report_data,unsupervised_model_name,'time_train_finish',datetime.datetime.fromtimestamp(time.time()))
                                            
                                        subsubstep_finish_time = datetime.datetime.fromtimestamp(time.time())
                                        repg.register_log([time_log],'>>Substep 4.3.1.2 Training unsupervised model ' + unsupervised_model_name +' elapsed time: ' + str(subsubstep_finish_time - subsubstep_init_time) + "\n",'',char_encoding)
                                        repg.register_log(array_of_paths_log_files,'>>>>Substep 4.3.1.2 ends '+ datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                                            
                                        '''Subsubstep 4.3.1.3 Obtainin model accuracy'''
                                        subsubstep_init_time = datetime.datetime.fromtimestamp(time.time())
                                        repg.register_log(array_of_paths_log_files,'>>>> Step 3.8.3 Obtaining accuracy and learning curve for the model ' + unsupervised_model_name + ' ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                                        
                                        predictions = current_trained_model.predict(df_catalogued_data[features_used_train_models])
                                            
                                        diccionario_reasignacion = unsu.get_dictionary_of_reasignation_of_labels(current_trained_model,df_catalogued_data,target)                                        
                                            
                                        repg.register_log([log_file],'>> Dictionary of reasignation of values:\n\t-> ' + str(diccionario_reasignacion) + "\n",'',char_encoding)
                                                                                                                            
                                        for prediction_index in range(len(predictions)):
                                            predictions[prediction_index] = diccionario_reasignacion[predictions[prediction_index]]
                                        
                                        train_true=list(df_catalogued_data[target])
                                        train_predictions_model=predictions                                                                                        
                                                                                    
                                        accuracy_current_model= unsu.get_accuracy(train_true,predictions)
                                        report_data = repg.actualizar_accuracy_modelo(report_data,unsupervised_model_name,accuracy_current_model)
                                        numpy_array_accuracy_current_model=np.array([accuracy_current_model])
                                        
                                        subsubstep_finish_time = datetime.datetime.fromtimestamp(time.time())
                                        repg.register_log([time_log],'>>Substep 4.3.1.3 Getting accuracy for the model ' + unsupervised_model_name +' elapsed time: ' + str(subsubstep_finish_time - subsubstep_init_time) + "\n",'',char_encoding)
                                        repg.register_log(array_of_paths_log_files,'>>Accuracy obtained: ' + str(accuracy_current_model) + "\n",'',char_encoding)
                                        repg.register_log(array_of_paths_log_files,'>>Substep 4.3.1.3 ends ' + "\n",'',char_encoding)
                                        
                                        '''Subsubstep 4.3.1.4 Obtaining learning curves for the model'''                                        
                                        subsubstep_init_time = datetime.datetime.fromtimestamp(time.time())
                                        #learning curve
                                        #1 - obtenemos el diccionario que asocia el target original a los clusters {target:cluster}
                                        dictionary_of_reasignation_of_targets_inverse = {}
                                        for cluster_generado in diccionario_reasignacion:
                                            target_original_asignado = diccionario_reasignacion[cluster_generado]
                                            dictionary_of_reasignation_of_targets_inverse[target_original_asignado] = cluster_generado
                                                                                    
                                        #2-Los target originales los recodificamos con las etiqueta de los cluster correspondientes
                                        train_true_learning_curve = []
                                        for index_of_target_original in range(len(train_true)):
                                            target_inicial = train_true[index_of_target_original]
                                            train_true_learning_curve.append(dictionary_of_reasignation_of_targets_inverse[target_inicial])
                                                                                
                                        
                                        kmeans = unsu.initialize_model(unsupervised_model_name,parameters_current_model,unsupervised_models_dictionary)
                                        lista_scorings = metr.get_scoring_methods_unsupervised()
                                        cv_iterators = metr.get_all_available_cross_validation_iterators(10,0.2)#obtenemos la lista de iteradores posibles                                            
                                        for indice_sc in range(len(lista_scorings)):
                                            score_curve = lista_scorings[indice_sc]
                                            score_name = score_curve[0]   

                                            for index_cv_it in range(len(cv_iterators)):
                                                cv_iterator_name = cv_iterators[index_cv_it][0]
                                                cv_it = cv_iterators[index_cv_it][1]
                                                train_sizes_def=np.linspace(.1,1.0,10)                                                
                                                train_sizes, train_scores, test_scores=metr.compute_learning_curve(kmeans,df_catalogued_data[features_used_train_models],train_true_learning_curve,score_curve,cv_it,train_sizes=train_sizes_def)
                                                    
                                                name_of_the_curve = "Learning curve for It_" + str(cv_iterator_name) + "_Sc_"+ score_name + "_" + unsupervised_model_name + ".png"
                                                full_path_to_learning_curve_file = os.path.join(path_to_current_unsupervised_model_output_directory,name_of_the_curve)                          
                                                train_scores_mean,test_scores_mean = metr.save_learning_curve(train_sizes, train_scores, test_scores, "Learning curve(Iterator-" +str(cv_iterator_name) + "and Score-" +score_name +") for "+ unsupervised_model_name, full_path_to_learning_curve_file,lc_output_format)
                                                report_data = repg.update_model_feature(report_data,unsupervised_model_name,'lc_'+str(indice_sc)+"_"+str(index_cv_it),full_path_to_learning_curve_file)
         
                                                
                                        subsubstep_finish_time = datetime.datetime.fromtimestamp(time.time())
                                        repg.register_log([time_log],'>>Substep 4.3.1.4 Calculating learning curve for the model ' + unsupervised_model_name + ' elapsed time: ' + str(subsubstep_finish_time - subsubstep_init_time) + "\n",'',char_encoding)
                                        repg.register_log(array_of_paths_log_files,'>>>> Step 4.3.1.4 ends ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                                            
                                        '''Subsubstep 4.3.1.5 Obtencion de las matrices de confusion y del indice de decision para el modelo'''
                                        subsubstep_init_time = datetime.datetime.fromtimestamp(time.time())
                                        repg.register_log(array_of_paths_log_files,'>>>>Substep 4.3.1.5 Generating confusion matrix for '+ unsupervised_model_name  + ' '+ datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                                            
                                        confusion_matrix_delimiter =',' 
                                        confusion_matrix_model_train=metr.get_confusion_matrix(train_true,train_predictions_model,target_recodified_values)
                                        
                                        #getting metrics                                        
                                        tr_precisions,tr_recalls,tr_specificity,tr_f1_score,tr_mcc,confusion_tables_by_class_train=metr.metrics_cm(confusion_matrix_model_train)
                                        
                                        #computing micro and 
                                        precision_macro_avg_train,recall_macro_avg_train,specificity_macro_avg_train,f1_score_macro_avg_train,mcc_macro_avg_train,report_data = metr.compute_macro_avg_values_of_metrics(list_names_metrics,tr_precisions,tr_recalls,tr_specificity,tr_f1_score,tr_mcc,target_recodified_values,report_data,unsupervised_model_name)                                        
                                        precision_micro_avg_train,recall_micro_avg_train,specificity_micro_avg_train,f1_score_micro_avg_train,mcc_micro_avg_train,report_data = metr.compute_micro_avg_values_of_metrics(list_names_metrics,confusion_tables_by_class_train,target_recodified_values,report_data,unsupervised_model_name)
                                        #repg.update_report_metrics(report_data,unsupervised_model_name,[tr_precisions,tr_recalls,tr_specificity,tr_f1_score,tr_mcc],target_recodified_values,list_names_metrics)
                                        
                                        #computing incorrect classifications                                        
                                        confusion_matrix_name_train='confusion_matrix'+ unsupervised_model_name                                                                                
                                        df_catalogued_data = df_catalogued_data.reset_index()
                                        matrix_summary_clasifications = metr.confusion_matrix_gather_feature_incorrect_values(df_catalogued_data[features_used_during_process], train_true, train_predictions_model, observation_number)                                        
                                        metr.confusion_matrix_to_csv(matrix_summary_clasifications,list_targets_current_event,os.path.join(path_to_current_unsupervised_model_output_directory,'confusion_matrix_incorrect_classifications' + "." + 'csv'))
                                                                                
                                        #ts=train_sizes,ss=sample_scores,cvs=cv_scores,cm=confusion_matrix,w=weights,ac=accuracy                                        
                                        try:
                                            if(main_metric == 'mcc'):                                                
                                                decision_index_current_model=auxf.weight_for_decision_function(train_sizes,train_scores_mean,test_scores_mean,confusion_matrix_model_train,w,mcc_micro_avg_train)
                                            else:
                                                decision_index_current_model=auxf.weight_for_decision_function(train_sizes,train_scores_mean,test_scores_mean,confusion_matrix_model_train,w,accuracy_current_model)
                                        except Exception:
                                            raise Exception(' the decision index has an incorrect value (inf)')                                        
                                        repg.update_report_current_model_decision_index(report_data,unsupervised_model_name,str(decision_index_current_model))
                                        
                                        #saving egenrated matrices
                                        repg.save_data_to_file(confusion_matrix_model_train.astype(int),path_to_current_unsupervised_model_output_directory,confusion_matrix_name_train,confusion_matrix_delimiter,'numpy')
                                        metr.save_confusion_matrix(confusion_matrix_model_train,target_recodified_values,os.path.join(path_to_current_unsupervised_model_output_directory,confusion_matrix_name_train),lc_output_format)
                                        report_data = repg.update_model_feature(report_data,unsupervised_model_name,"Confussion_matrix_train_path",os.path.join(path_to_current_unsupervised_model_output_directory,confusion_matrix_name_train + "." + lc_output_format))
                                        repg.register_log(array_of_paths_log_files,'>> Confusion matrix:\n '+ str(confusion_matrix_model_train.tolist())  + ' '+ datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)                                                                                
                                        
                                    
                                        subsubstep_finish_time = datetime.datetime.fromtimestamp(time.time())
                                        repg.register_log([time_log],'>>Substep 4.3.1.5 Generating confusion matriz and decision index for the model ' + unsupervised_model_name + ' elapsed time: ' + str(subsubstep_finish_time - subsubstep_init_time) + "\n",'',char_encoding)
                                        repg.register_log(array_of_paths_log_files,'>>>>Substep 4.3.1.5 ends ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                                        
                                        
                                        '''Subsubstep 4.3.1.6 Generating model pkl'''
                                        subsubstep_init_time = datetime.datetime.fromtimestamp(time.time())
                                        repg.register_log(array_of_paths_log_files,'>>>>Substep 4.3.1.6 generating reports and files for the model '+ unsupervised_model_name  + ' ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                                        repg.save_data_to_file(numpy_array_accuracy_current_model,path_to_current_unsupervised_model_output_directory, 'informe_accuracy_'+unsupervised_model_name, output_files_delimiter,'txt')
                                        path_to_store_current_generated_model = unsu.save_model_to_disk(current_trained_model, path_to_current_unsupervised_model_output_directory,unsupervised_model_name +'.pkl')                                            
                                        dictionary_of_selected_models_for_events[unsupervised_model_name] = [current_learning,path_to_store_current_generated_model,diccionario_reasignacion,features_used_train_models]
                                        
                                        subsubstep_finish_time = datetime.datetime.fromtimestamp(time.time())
                                        repg.register_log([time_log],'>>Substep 4.2.1.6 Generating model pkl for ' + unsupervised_model_name + ' elapsed time: ' + str(subsubstep_finish_time - subsubstep_init_time) + "\n",'',char_encoding)
                                        repg.register_log(array_of_paths_log_files,'>>>>Substep 4.2.1.6 ends ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                                            
                                    except Exception as e:
                                        if unsupervised_model_name in report_data:
                                            del report_data[unsupervised_model_name]
                                            auxf.delete_directory_with_content(path_to_current_unsupervised_model_output_directory)
                                        repg.register_log([errors_file],'\n>>'+unsupervised_model_name +' model was not created for the event '+ current_event + ' because ' + auxf.decodify_using_enconding(str(e),char_encoding) + "\n",'',char_encoding)
                                        repg.register_log([errors_file],'>> An error has happened '+ auxf.decodify_using_enconding(str(e),char_encoding) + "\n",'',char_encoding)
                                        repg.register_log(array_of_paths_log_files,'>> An error has happened check error log #######' + "\n",'',char_encoding)
                                        error_trace = "\n" + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')+ "\n Full trace:\n" + str(traceback.format_exc())
                                        repg.register_log([errors_file], error_trace,'',char_encoding)                                        
                                    
                                    model_finish_time = datetime.datetime.fromtimestamp(time.time())
                                    repg.register_log([time_log],'>>Substep Creation of model '+ unsupervised_model_name +' total elapsed time: ' + str(model_finish_time - model_init_time) + "\n",'',char_encoding)
                                        
                                repg.register_log(array_of_paths_log_files,'>>>>>>Substep 4.3 ends for model' +unsupervised_model_name  +' '+ datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n\n",'',char_encoding)
            
                            substep_finish_time = datetime.datetime.fromtimestamp(time.time())
                            repg.register_log([time_log],'>>Substep 4.3 Creating and applying models of unsupervised learning total elapsed time: ' + str(substep_finish_time - substep_init_time) + "\n",'',char_encoding)
                            step_finish_time = datetime.datetime.fromtimestamp(time.time())
                            total_time_step_4 = step_finish_time - step_init_time
                                                       
                            repg.register_log([time_log],'>>>>Step 4 - Creating supervised models for catalogued data total elapsed time: ' + str(total_time_step_4)  + "\n\n",'',char_encoding)
                            total_model_phase_time = total_time_step_2 + total_time_step_3 + total_time_step_4
                            repg.register_log([time_log],'>>>>Phase of creating models for catalogued data total elapsed time: ' + str(total_model_phase_time)  + "\n\n",'',char_encoding)
                                                        
            '''Step 5 Selecting best model for current event'''
            repg.register_log([time_log],'>>>>Step 5 starts:\n','',char_encoding)
            step_init_time = datetime.datetime.fromtimestamp(time.time())
            repg.register_log(array_of_paths_log_files,'>>>>Step 5 Selecting best model for '+ current_event + ' ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)

            '''Obtaining the best model and its characteristics'''
            try:                
                best_model = auxf.get_model_with_highest_decision_index(report_data,list_available_models_for_the_event)
            except Exception as e:
                raise Exception('There are no enough data. No model could be created ')
                
            best_model_characteristics = dictionary_of_selected_models_for_events[best_model[0]] # [tipo_aprendizaje,modelo,diccionario*]
            
            '''Copying the model to the destiny directory'''
            best_model_final_pkl_name = auxf.decodify_using_enconding(config_parser.get(output_data_section,'output_directory_name_prediction_models'),char_encoding) + '/' + current_event + "_" + target +"_"+best_model[0]+".pkl"
            
            
            pred_path = auxf.decodify_using_enconding(config_parser.get(output_data_section,'output_directory_name_prediction_models'),char_encoding)
            for root_pred,directory_pred,files_pred in os.walk(pred_path):
                for filename_pred in files_pred:                    
                    if(target in filename_pred) and(current_event in filename_pred) and('.pkl' in filename_pred):                
                        try:                            
                            os.remove(os.path.join(pred_path,filename_pred))
                        except OSError as e:
                            pass
            shutil.copy(best_model_characteristics[1],best_model_final_pkl_name)
                                    
            '''Updating general prediction_models-events dictionary'''            
            best_model_characteristics[1] = best_model_final_pkl_name                        
            prediction_models_for_events_dictionary = auxf.add_entry_for_event_in_prediction_dictionary(prediction_models_for_events_dictionary,current_event)
            prediction_models_for_events_dictionary = auxf.update_prediction_models_original_features_for_event_in_prediction_dictionary(prediction_models_for_events_dictionary,current_event,target,best_model_characteristics,reserved_derived_features_character,list_of_parameters_models_events_dict)
            handler_prediction_models_for_events_dictionary.close()
            try:
                '''Saving in memory actual state of dictionary'''
                path_to_prediction_models_dictionary = os.path.join(path_to_directory_prediction_models,auxf.decodify_using_enconding(config_parser.get(output_data_section,'prediction_models_dictionary_filename'),char_encoding))
                diccionario_modelos,handler = auxf.retrieve_dictionary_and_handler(path_to_prediction_models_dictionary) #recover dictionary with models                
                estado_inicial_diccionario = copy.deepcopy(diccionario_modelos) #storing initial state in case of error
                handler.close()
                
                '''Updating dictionary'''
                handler_prediction_models_for_events_dictionary = auxf.get_dictionary_in_pickle_format_handler(path_to_prediction_models_dictionary)                
                auxf.save_dictionary_to_disk_pickle_format(prediction_models_for_events_dictionary,handler_prediction_models_for_events_dictionary)                
                
                '''Generating reports with information about the models'''
                prediction_models_for_events_dictionary,handler = auxf.retrieve_dictionary_and_handler(path_to_prediction_models_dictionary)                
                repg.create_report_current_dictionary_models(prediction_models_for_events_dictionary,auxiliary_directory_filename,ruta_directorio_raiz_salida,list_of_parameters_models_events_dict,"'" + os.path.join(path_to_dir_auxiliary_files,"logo.jpg") + "'",char_encoding)
                handler.close()
                
            except KeyboardInterrupt:                
                handler_prediction_models_for_events_dictionary.close()
                auxf.restore_dictionary_to_previous_version(path_to_prediction_models_dictionary,estado_inicial_diccionario)
                raise Exception('Process interrumpted during the creation on the prediction_models_dict.pickle.\nPlease, restart de training process for the event ' + current_event + ' and target '+ target)
            
            repg.register_log(array_of_paths_log_files,'>>>> Extra Step  Dictionary with predicion models ' + str(prediction_models_for_events_dictionary) + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',char_encoding)
                        
            '''Generating reports with information about the models'''
            repg.create_report_current_model(report_data,list_available_models_for_the_event,auxiliary_directory_filename,path_to_reports_current_event_directory,char_encoding)
            repg.register_log([log_file],'>>>> Extra Step Report Data Info Gathered' + str(report_data)+ datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "<<<<<<<<<<<<<<<<\n\n",'',char_encoding)      
            repg.register_log([log_file],'>>>>>>>>>>>>>>>>>> Best model selected and report generated for ' + current_event + '  ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "<<<<<<<<<<<<<<<<\n\n",'',char_encoding)
            
            step_finish_time = datetime.datetime.fromtimestamp(time.time())
            total_time_step_5 = step_finish_time - step_init_time
            repg.register_log([time_log],'>>>>Step 5 - Selecting best model for '+ current_event +' total elapsed time: ' + str(total_time_step_5)  + "\n\n",'',char_encoding)
            repg.register_log([time_log],'>>>>Total time for event ' + current_event + ' (Step 2 + Step 3 + Step 4 + Step 5): ' + str(total_time_step_2 + total_time_step_3 + total_time_step_4 + total_time_step_5)  + "\n\n",'',char_encoding)
            
            
                                                                
        except FollowFlowException:
            print ''
                    
        except Exception as e:              
            repg.register_log([errors_file],u'>> An error has happened '+ auxf.decodify_using_enconding(str(e),char_encoding) + u"\n",'',char_encoding)
            repg.register_log(array_of_paths_log_files,'>> An error has happened check error log #######' + "\n",'',char_encoding)
            error_trace = "\n" + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')+ "\n Full trace:\n" + str(traceback.format_exc())
            repg.register_log([errors_file], auxf.decodify_using_enconding(error_trace,char_encoding)+"\n",'',char_encoding)
            report_data = repg.upddate_report_warning_info(report_data, auxf.decodify_using_enconding(str(e),char_encoding) + ' for ' + current_event + '.' )
            repg.create_report_current_model(report_data,[],auxiliary_directory_filename,path_to_reports_current_event_directory,char_encoding)            
            
        repg.register_log([errors_file],'','',char_encoding)
         
    execution_finish_time = datetime.datetime.fromtimestamp(time.time())
    repg.register_log([time_log],'>>>> Total elapsed time for the Training phase: ' + str(execution_finish_time - execution_init_time) + "\n",'',char_encoding)
    repg.register_log([log_file],'>>>>>>>>>>>>>>>>>> EXECUTION FINISHES FOR THE TRAINING PHASE '+ datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "<<<<<<<<<<<<<<<<\n\n",'',char_encoding)
    repg.register_log([log_file],'>> ALL PREPARED TO PREDICT. THANKS FOR TRAINING \n\n' ,'',char_encoding)
    print 'Training process finished. Check the logs and reports for more information.'
    print 'Thanks for training using RADSSo'
        
except KeyError as e:
    repg.register_log([errors_file],'>> Execution aborted by the user '+ auxf.decodify_using_enconding(e.message,char_encoding) + ' ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')+ '\n','',char_encoding)
    error_trace = "Full trace:\n" + auxf.decodify_using_enconding(str(traceback.format_exc()),char_encoding)
    repg.register_log([errors_file], error_trace,'',char_encoding)
    
    
except KeyboardInterrupt as e:
    repg.register_log([errors_file],'>> Execution aborted by the user '+ auxf.decodify_using_enconding(e.message,char_encoding) + ' ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')+ '\n','',char_encoding)
    error_trace = "Full trace:\n" + auxf.decodify_using_enconding(str(traceback.format_exc()),char_encoding)
    repg.register_log([errors_file], error_trace,'',char_encoding)
    

except IndexError as e:
    repg.register_log([errors_file],'>> An IndexError exception has happened '+ auxf.decodify_using_enconding(e.message,char_encoding) + ' ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')+ '\n','',char_encoding)
    error_trace = "Full trace:\n" + auxf.decodify_using_enconding(str(traceback.format_exc()),char_encoding)
    repg.register_log([errors_file], error_trace,'',char_encoding)
    
    
except ValueError as e:
    repg.register_log([errors_file],'>> A ValueError exception has happened '+ auxf.decodify_using_enconding(str(e.message),char_encoding) + ' ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')+ '\n','',char_encoding)    
    error_trace = "Full trace:\n" + auxf.decodify_using_enconding(str(traceback.format_exc()),char_encoding)
    repg.register_log([errors_file], error_trace,'',char_encoding)

    
except IOError as e:        
    repg.register_log([errors_file],u'>> An IOError has happened '+ auxf.decodify_using_enconding(e.message,char_encoding) + u' ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')+ u'\n','',char_encoding)
    error_trace = "Full trace:\n" + auxf.decodify_using_enconding(str(traceback.format_exc()),char_encoding)
    repg.register_log([errors_file], error_trace,'',char_encoding)
    
    
except Exception as e:
    repg.register_log([errors_file],'>> An Unknow Exception has happened '+ auxf.decodify_using_enconding(e.message,char_encoding) + ' ' +datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + '\n','',char_encoding)
    error_trace =  "Full trace:\n" + auxf.decodify_using_enconding(str(traceback.format_exc()),char_encoding)
    repg.register_log([errors_file], error_trace,'',char_encoding)    