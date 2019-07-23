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
import sys
import json
import pandas as pd

class FinishFlowException(Exception):
    pass


def get_items_dict(list_of_available_items):
    '''
    This funtion allows to generate the dictionary of elements for the user to be able to select them
    using the standard input. Options All and Exit/Continue are also added

    Parameters:
    :param list<str> list_of_available_items: list with items to create the list of available 
    options to select by the user    
    
    :return: dictionary_numeric_options_linked_to_items, dictionary with numeric options and the item linked to the number of the option
    :rtype: dict<int:str> 
    '''

    list_of_items_in_alphabetic_order = sorted(list_of_available_items)
        
    list_of_items_in_alphabetic_order.reverse()
    list_of_items_in_alphabetic_order.append('All')
    list_of_items_in_alphabetic_order.reverse()    
    
    dictionary_numeric_options_linked_to_items = {}
    for current_index in range(len(list_of_items_in_alphabetic_order)):
        dictionary_numeric_options_linked_to_items[len(dictionary_numeric_options_linked_to_items)] = list_of_items_in_alphabetic_order[current_index]
    dictionary_numeric_options_linked_to_items[-1] = 'Exit/Continue'

    return dictionary_numeric_options_linked_to_items


def show_elegible_items (items_dictionary,topic,enco):
    '''
    This funtion allows to show to the user and concrete menu of options for the current topicfor
    the user to select options from it.

    Parameters:
    :dictionary<int:str> items_dictionary, dictionary to generate the menu for with the available options for the current topic
    :str topic: area to which the displayed options belongs to
    
    :return: shown_string, menu in format string with all the information for the user to decide
    :rtype: str
    '''
    
    shown_string = "Please, select the " + topic + " you want:\n"
    for option in items_dictionary:
        if(enco != 'none'):            
            shown_string+= str(option) + " - "
            corrected = ''            
            for element in items_dictionary[option]:
                corrected +=  unichr(ord(element)).encode(enco)            
            shown_string+=corrected + "\n"
        else:
            shown_string+= option + " - " + items_dictionary[option] + "\n"
    
    return shown_string


def show_format_selected_items(list_of_numbers_related_to_items,items_dictionary,topic):
    '''
    This funtion allows to display to the user the items that he has selected prevously realted to a topic. As the user
    select numbers we need to recover th eoriginal item from the selected number

    Parameters:
    :list<int> list_of_numbers_related_to_items, list with the number of the items selected by the user
    :str topic: area to which the displayed options belongs to
    
    :return: list_of_selected_items, list of names of selected items for the current topic
    :rtype: list<str>
    :return: list_numbers_related_to_items_selected, list of numbers of selected items for the current topic
    :rtype: list<int>
    '''

    list_of_selected_items = []
    list_numbers_related_to_items_selected = []
    print "\nSelected " + topic + " by the user:\n"
    if(0 in list_of_numbers_related_to_items):
        list_of_numbers_related_to_items = [0] #0 options means all available items
        
    for item in list_of_numbers_related_to_items:
        if(item == 0):
            for option in items_dictionary:
                if(option != -1 and option !=0):
                    print items_dictionary[option]
                    list_of_selected_items.append(items_dictionary[option])
                    list_numbers_related_to_items_selected.append(option)
        else:
            print items_dictionary[item]
            list_of_selected_items.append(items_dictionary[item])
            list_numbers_related_to_items_selected.append(item)
    
    return list_of_selected_items,list_numbers_related_to_items_selected

	
def recognize_available_variables_automatically(vector_with_fullpath_to_inputs_files,input_files_delimiter,list_of_not_valid_characters,enco):
    '''
    It allows to recognize all the existing events in the specififed input file in an automatic way.

    Parameters:
    :param list<str> vector_with_fullpath_to_inputs_files: list with fullpaths to inputs files from which to obtain the events names
    :param str feature_with_events_names: name of the feature from which to obtain the names of the events looping through its values
    :param str input_files_delimiter: delimiter of the fields for the input files
    :param list<str> list_of_not_valid_characters: list with character not valid be contained by an event name
    :param str enco: string with the codification to apply
    
    
    :return: list_verified_events, list of events with valid names that has been recognized automatically
    :rtype: list<str>
    :return: list_verified_features, list of found features
    :rtype: list<str>
    '''
        
    list_verified_features = []
    print "\nReading input files..."
    for i in range(len(vector_with_fullpath_to_inputs_files)):          
        #looping through input files to get all possible events
        original_data=pd.read_csv(vector_with_fullpath_to_inputs_files[i],sep=input_files_delimiter)
        if(list_verified_features == []):
            for element in original_data.columns:                
                list_verified_features.append(element.decode(enco))

    #At least there is one variable    
    if(list_verified_features == []):
        raise Exception ('No features were found inside the specified input csv')
    return list_verified_features

def recognize_available_events_and_variables_automatically(vector_with_fullpath_to_inputs_files,feature_with_events_names,input_files_delimiter,list_of_not_valid_characters,enco):
    '''
    It allows to recognize all the existing events in the specififed input file in an automatic way.

    Parameters:
    :param list<str> vector_with_fullpath_to_inputs_files: list with fullpaths to inputs files from which to obtain the events names
    :param str feature_with_events_names: name of the feature from which to obtain the names of the events looping through its values
    :param str input_files_delimiter: delimiter of the fields for the input files
    :param list<str> list_of_not_valid_characters: list with character not valid be contained by an event name
    :param str enco: string with the codification to apply
    
    
    :return: list_verified_events, list of events with valid names that has been recognized automatically
    :rtype: list<str>
    :return: list_verified_features, list of found features
    :rtype: list<str>
    '''

    name = feature_with_events_names
    list_verified_events = []
    list_verified_features = []
    print "\nReading input files:"
    for i in range(len(vector_with_fullpath_to_inputs_files)):
        print "\tFile number ", str(i),"\n"        
        #looping through input files to get all possible events
        original_data=pd.read_csv(vector_with_fullpath_to_inputs_files[i],sep=input_files_delimiter)
        if(list_verified_features == []):
            for element in original_data.columns:                
                list_verified_features.append(element.decode(enco))
            
        found_events = list(set(original_data[name].values))        
        for event in found_events:
            add_event = True
            if pd.isnull(event):
                event = 'nan'
            if event not in list_verified_events :
                for not_valid_char in list_of_not_valid_characters:
                    if(not_valid_char in event):
                        add_event = False
                if(add_event):
                    list_verified_events.append(event.decode(enco))   
    #checking that at least there is one event and one variable
    if(list_verified_events == []):
        raise Exception ('No events were found inside the specified input csv')
    if(list_verified_features == []):
        raise Exception ('No features were found inside the specified input csv')
    return list_verified_events,list_verified_features
    
    
def get_events_and_variables_from_user_interaction(vector_with_fullpaths_of_input_files,event_name,input_files_delimiter,list_of_not_valid_characters,enco):
    '''
    It allows the user to select in an interactive way the events and the features(important and discarded) for each one of the events
    to use in the training process.

    Parameters:   
    :param list<str> vector_with_fullpaths_of_input_files: list with fullpaths to inputs files from which to obtain the events names
    :feature_events_name: feature with the values from which to obtain the names of the events
    :param str input_files_delimiter: delimiter of the fields for the input files
    :param str enco: string with the codification to apply
    
    
    :return: list_verified_events, list of events with valid names that has been recognized automatically
    :rtype: list<str>
    :return: list_verified_features, list of found features
    :rtype: list<str>
    '''   
                                                                                               
    elegible_features = recognize_available_variables_automatically(vector_with_fullpaths_of_input_files,input_files_delimiter,list_of_not_valid_characters,enco)    
    
    if 'Unnamed: 0' in elegible_features:
        elegible_features.remove('Unnamed: 0')
    dictionary_with_elegible_features = get_items_dict(elegible_features)
    
    
    list_of_selected_events_by_the_user = [event_name]
    '''        
    Features selection process(relevant ones)    
    '''
    
    topic = 'important features'
    dictionary_events_features = {}
    event_selected_index = 0
    while (event_selected_index < len(list_of_selected_events_by_the_user)):
        current_selected_event = list_of_selected_events_by_the_user[event_selected_index] 
        dictionary_events_features[current_selected_event] = {}
        
        corrected_event = ''            
        for elemento in current_selected_event:
            corrected_event +=  unichr(ord(elemento)).encode(enco)            
            
        
        print "\nSelecting " + topic + " for " + corrected_event
        sys.stdout.flush()
        reset_features_process = ''
        while(reset_features_process != 'N'):
            list_of_selected_features = []                  
            selected_option =''
            while(selected_option != -1 and selected_option !=0):
                sys.stdout.flush()
                try:                
                    print show_elegible_items(dictionary_with_elegible_features,topic,enco)
                    usr_selection = raw_input()
                    if(usr_selection == 'exit'):
                        raise KeyboardInterrupt
                    else: 
                        selected_option = int(usr_selection)                        
                except KeyboardInterrupt:
                    raise KeyboardInterrupt('Process aborted')
                except Exception as e:
                    print e
                    pass
                sys.stdout.flush()
                if (selected_option in dictionary_with_elegible_features and selected_option !=-1 and selected_option not in list_of_selected_features):
                    list_of_selected_features.append(selected_option)
                elif(selected_option != -1):
                    print "Incorrect variable number"
                    sys.stdout.flush()
                    selected_option = ''
            
            #>At this point we have the list with events and the corresponding list with relevant features
            numeric_list_selected_features = []
            if(list_of_selected_features != []):
                list_of_selected_features,numeric_list_selected_features = show_format_selected_items(list_of_selected_features,dictionary_with_elegible_features,topic)
                sys.stdout.flush()
                print "Are the selected variables correct for "+corrected_event+"?['Y','N']:"
                try:
                    user_confirmation = raw_input()
                    if(user_confirmation == 'exit'):
                        raise KeyboardInterrupt
                except KeyboardInterrupt:
                    raise KeyboardInterrupt('Process aborted')
                sys.stdout.flush()
            else:
                sys.stdout.flush()
                print "No " + topic +" were selected. Is correct for "+corrected_event+"?['Y','N']:"
                try:
                    user_confirmation = raw_input()
                    if(user_confirmation == 'exit'):
                        raise KeyboardInterrupt
                except KeyboardInterrupt:
                    raise KeyboardInterrupt('Process aborted')
                sys.stdout.flush()
            while(user_confirmation !='Y' and user_confirmation!='N' and user_confirmation != 'exit'):                
                sys.stdout.flush()
                print "Is correct for "+ corrected_event + "?['Y','N']:"
                try:
                    user_confirmation = raw_input()
                    if(user_confirmation == 'exit'):
                        raise KeyboardInterrupt
                except KeyboardInterrupt:
                    raise KeyboardInterrupt('Process aborted')
                sys.stdout.flush()
            if(user_confirmation == 'Y'):
                event_selected_index +=1
                reset_features_process = 'N'                
                dictionary_events_features[current_selected_event]['relevant_features'] = list_of_selected_features#corrected_relevant_features
                if(list_of_selected_features != []):
                    print "Important features typed by the user for event: " + corrected_event +"\n "                    
                    sys.stdout.flush()
                    for elemento in list_of_selected_features:                    
                        print elemento
                        sys.stdout.flush()
                else:
                    print "No " + topic + " were specified for " + corrected_event
                    sys.stdout.flush()
            else:
                reset_features_process = 'Y'                                        
        
    print ">>>>> Selecting important features <<<<<<<"
    sys.stdout.flush()
    
    '''        
    Features selection process (discarded ones)   
    '''
    
    topic = 'discarded features'    
    event_selected_index = 0
    while (event_selected_index < len(list_of_selected_events_by_the_user)):
        current_selected_event = list_of_selected_events_by_the_user[event_selected_index] #evento actual                        
        corrected_event = ''            
        for elemento in current_selected_event:
            corrected_event +=  unichr(ord(elemento)).encode(enco)
        
        print "\nSelecting " + topic + " for " + corrected_event
        sys.stdout.flush() 
        reset_features_process = ''
        while(reset_features_process != 'N'):
            list_of_selected_features = []                  
            selected_option =''
            while(selected_option != -1 and selected_option !=0):
                sys.stdout.flush() 
                try:
                    print show_elegible_items(dictionary_with_elegible_features,topic,enco)
                    usr_selection = raw_input()
                    if(usr_selection == 'exit'):
                        raise KeyboardInterrupt
                    else: 
                        selected_option = int(usr_selection)                    
                except KeyboardInterrupt:
                    raise KeyboardInterrupt('Process aborted')
                except Exception:
                    pass
                sys.stdout.flush() 
                if (selected_option in dictionary_with_elegible_features and selected_option !=-1 and selected_option not in list_of_selected_features): # la opcion es correcta
                    list_of_selected_features.append(selected_option)
                elif(selected_option != -1):
                    print "Incorrect variable number"
                    sys.stdout.flush() 
                    selected_option = ''
            
            #Events list
            numeric_list_selected_features = []
            if(list_of_selected_features != []):
                list_of_selected_features,numeric_list_selected_features = show_format_selected_items(list_of_selected_features,dictionary_with_elegible_features,topic)
                sys.stdout.flush() 
                print "Are the selected variables correct for "+corrected_event+ "?['Y','N']:"
                try:
                    user_confirmation = raw_input()
                    if(user_confirmation == 'exit'):
                        raise KeyboardInterrupt
                except KeyboardInterrupt:
                    raise KeyboardInterrupt('Process aborted')
                sys.stdout.flush() 
            else:
                sys.stdout.flush() 
                print "No " + topic + "were selected. Is correct for "+corrected_event+ "?['Y','N']:"
                user_confirmation = raw_input()                
                sys.stdout.flush() 
            while(user_confirmation !='Y' and user_confirmation!='N'):                
                sys.stdout.flush() 
                print "Is correct for "+corrected_event+ "?['Y','N']:"
                try:
                    user_confirmation = raw_input()
                    if(user_confirmation == 'exit'):
                        raise KeyboardInterrupt
                except KeyboardInterrupt:
                    raise KeyboardInterrupt('Process aborted')
                sys.stdout.flush() 
            if(user_confirmation == 'Y'):
                event_selected_index +=1
                reset_features_process = 'N'
                corrected_discarded_features = []
                for d_feature in list_of_selected_features:
                    corrected_feature = ''
                    for character in d_feature:
                        corrected_feature+= unichr(ord(character)).encode(enco)
                    corrected_discarded_features.append(corrected_feature)                               
                dictionary_events_features[current_selected_event]['discarded_features'] = list_of_selected_features#corrected_discarded_features
                if(list_of_selected_features != []):
                    print "Important features typed by the user for event: " + corrected_event +"\n "
                    sys.stdout.flush()
                    for elemento in list_of_selected_features:                    
                        print elemento
                        sys.stdout.flush()
                else:
                    print "No " + topic + " were specified for " + corrected_event
                    sys.stdout.flush()
            else:
                reset_features_process = 'Y'                                        
        
    print ">>>>> Relevant and discarded features stored successfully <<<<<<<"
    sys.stdout.flush()
    
    return dictionary_events_features
    
    
def get_list_events_features_json_file(fullpath_to_events_json_file,list_of_not_valid_characters):
    '''
    It allows to get the list of events with their corresponding relevant features and the discarded features for each one of
    the events.

    Parameters:   
    :param str fullpath_to_events_json_file: fullpath to the json file with the events, the relevant features and the discarded features
    :param list<str> list_of_not_valid_characters: list with the characters that are not valid to be in the events name
    
    :return: vector_with_valid_events, list of events with valid names that has been recognized automatically
    :rtype: list<str>
    :return: list_lists_specific_relevant_features, list of relevant features for the valid events
    :rtype:: list<str>
    :return: list_lists_specific_discarded_features, list of discarded features for the valid events
    :rtype: list<str>
    '''

    print 'Reading json with specific variables (semi-automated mode)'    
    vector_with_valid_events = []
    list_lists_specific_relevant_features = []
    list_lists_specific_discarded_features = []
    if(os.path.lexists(fullpath_to_events_json_file)):
        with open(fullpath_to_events_json_file, "r") as read_file:
            json_events_features = json.load(read_file)
            for event in json_events_features:
                add_event = True
                for not_valid_char in list_of_not_valid_characters:
                    if not_valid_char in event:
                        add_event =False
                if(add_event):
                    vector_with_valid_events.append(event)
                    #getting list with relevant features
                    try:
                        list_current_relevant_features = json_events_features[event]['relevant_features']                    
                        if(isinstance(list_current_relevant_features,list)):
                            list_lists_specific_relevant_features.append(list_current_relevant_features)
                        else:
                            raise KeyError
                    except KeyError:
                        list_lists_specific_relevant_features.append([])
                    
                    #getting list discarded features
                    try:
                        list_current_dicarded_variables = json_events_features[event]['discarded_features']
                        if(isinstance(list_current_dicarded_variables,list)):                          
                            list_lists_specific_discarded_features.append(list_current_dicarded_variables)
                        else:
                            raise KeyError
                    except KeyError:
                        list_lists_specific_discarded_features.append([])
    else:
        raise ValueError('The file with the events provided does not exist')
    return vector_with_valid_events,list_lists_specific_relevant_features,list_lists_specific_discarded_features


def generate_learnings_dict():
    '''
    It allows to get the dictionary with the available learning methods

    Parameters:   
    :param str fullpath_to_events_json_file: fullpath to the json file with the events, the relevant features and the discarded features

    :return: dictionary_learnings, dictionary with avilable learnings
    :rtype: dict<int:str>
    '''

    dictionary_learnings = {}
    dictionary_learnings[0] =  'All'
    dictionary_learnings[1] =  'Supervised'    
    dictionary_learnings[2] =  'Unsupervised'    
    
    return dictionary_learnings


def get_dictionary_with_learnings_related_to_models():
    '''
    It allows to get the dictionary with the available learning methods with the corresponding list of available models

    Parameters:   
    :no input parameters

    :return: dictionary_relationships_learnings_models, dictionary where the keys are the avilable learnings and the values for each one of the keys
    are the supervised and the unsupervised models
    :rtype: dict<str:[str]>
    '''

    learnings_dictionary = generate_learnings_dict()        
    
    supervised_models_dictionary = generate_dictionary_available_supervised_learning_models()
    list_supervised_models = []
    for key in supervised_models_dictionary:
        accumulated_supervised_models = supervised_models_dictionary[key]
        if(accumulated_supervised_models != 'All'):
             list_supervised_models.append(accumulated_supervised_models)
    
    unsupervised_models_dictionary = generate_dictionary_available_unsupervised_learning_models()
    list_unsupervised_models = []
    for key in unsupervised_models_dictionary:
        accumulated_unsupervised_models = unsupervised_models_dictionary[key]
        if(accumulated_unsupervised_models != 'All'):
             list_unsupervised_models.append(accumulated_unsupervised_models)
    
    
    dictionary_relationships_learnings_models = {}
    dictionary_relationships_learnings_models[learnings_dictionary[0]] = [list_supervised_models,list_unsupervised_models]
    dictionary_relationships_learnings_models[learnings_dictionary[1]] = list_supervised_models
    dictionary_relationships_learnings_models[learnings_dictionary[2]] = list_unsupervised_models
    
    return dictionary_relationships_learnings_models


def generate_dictionary_available_supervised_learning_models():
    '''
    It allows to get the dictionary with the available supervised learning models

    Parameters:   
    :no input parameters

    :return: dictionary_supervised_models, dictionary where the keys sequential numbers and the values for each one of the keys
    is the name of the supervised model
    :rtype: dict<str:[str]>
    '''

    dictionary_supervised_models = {}
    dictionary_supervised_models[0] = 'All'    
    dictionary_supervised_models[1] = 'Tree'
    dictionary_supervised_models[2] = 'Ada'    
    dictionary_supervised_models[3] = 'Boosting'                                   
    dictionary_supervised_models[4] = 'RandomForest'
    dictionary_supervised_models[5] = 'MLPerceptron'    
    
    return dictionary_supervised_models


def generate_dictionary_available_unsupervised_learning_models():
    '''
    It allows to get the dictionary with the available unsupervised learning models

    Parameters:   
    :no input parameters

    :return: dictionary_unsupervised_models, dictionary where the keys sequential numbers and the values for each one of the keys
    is the name of the unsupervised model
    :rtype: dict<str:[str]>
    '''

    dictionary_unsupervised_models = {}
    dictionary_unsupervised_models[0] = 'All'
    dictionary_unsupervised_models[1] = 'Kmeans'
    return dictionary_unsupervised_models

def get_common_discarded_features_by_user_from_file(fullpath_to_file_with_discarded_features_by_user,input_field_delimiter):
    '''
    It allows to get the list of common discarded features for all the events that are read from json file

    Parameters:   
    :param str fullpath_to_file_with_discarded_features_by_user: string with the fullpath to the file with the discarded features provided by the user
    :param str input_field_delimiter: field delimiter of the input file

    :return: list_discarded_variables, list with the variables discarded by the user
    :rtype: list<str>
    '''

    list_discarded_features = []
    if(os.path.lexists(fullpath_to_file_with_discarded_features_by_user)):
        with open(fullpath_to_file_with_discarded_features_by_user,'r') as fichero:
            array_listas_variables_descartadas = fichero.read().splitlines()
            read_line = array_listas_variables_descartadas[0]        
            if(read_line.strip() != ''):
                variables = read_line.split(input_field_delimiter)                
                for variable_act in variables:
                    list_discarded_features.append(variable_act.replace('\n','').replace('\r',''))            
    return list_discarded_features


def check_correspondency_input_lists_events_learnings_and_models(list_of_events,list_of_learnings,list_of_models):
    '''
    This function allows to check:
        that the length is the same for the list of events, learnings and models
        that the learning for each event corresponds to the associated models
        if any of the parameters is not correct an exception is thrown and execution is finished

    Exceptions:
    raise ValueError if any of the models do not correspond to the learnings
    raise ValueError if any of the models do not exist
    raise IndexError if the lengths events do not match with the number of models or the number of learnings
    Parameters:    
    :param list<str> list_of_events: list with the events provided previously
    :param list<str> list_of_learnings: list with the learning associated to each event
    :param list<tr> list_of_models: list with the models associated to each learning

    return: None
    '''

    lenght_events = len(list_of_events)
    lenght_learnings = len(list_of_learnings)
    length_models = len(list_of_models)
    if((lenght_events == lenght_learnings)and(lenght_learnings == length_models)): #three lists must have the same length
        diccionario_relaciones_ap_mod =  get_dictionary_with_learnings_related_to_models()        
        for indice in range(len(list_of_learnings)):
            aprendizaje = list_of_learnings[indice]            
            modelos_pedidos = list_of_models[indice]
            diccionario_relacion_modelos_permitidos = diccionario_relaciones_ap_mod[aprendizaje]
            
            if(aprendizaje == 'All'): #both learning methods
                modelos_pedidos_sup = modelos_pedidos[0]                
                diccionario_permitidos_sup =  diccionario_relacion_modelos_permitidos[0]            

                #checking supervised models
                modelos_sup_correctos = True
                i=0
                while(modelos_sup_correctos and i<len(modelos_pedidos_sup)):
                    modelo_comprobar = modelos_pedidos_sup[i]
                    if modelo_comprobar not in diccionario_permitidos_sup:
                        modelos_sup_correctos = False
                        raise ValueError('Any of the supervised models specified for the event ' + list_of_events[indice] + ' and the learning ' +  aprendizaje  + ' is not allowed')
                    i+=1
                
            else:#only supervised learning
                diccionario_permitidos = diccionario_relaciones_ap_mod[aprendizaje]
                
                modelos_correctos = True
                i=0
                while(modelos_correctos and i<len(modelos_pedidos)):
                    modelo_comprobar = modelos_pedidos[i]
                    try:                    
                        if modelo_comprobar not in diccionario_permitidos:
                            modelos_correctos = False
                            raise ValueError('Some model in the list ' + str(modelos_pedidos) +  ' that the user has specified for the event ' + list_of_events[indice] + ' and the learning ' + aprendizaje  + ' is not allowed. \n')
                    except Exception as e:                        
                        raise e
                    i+=1
                
    else:        
        raise IndexError('>> The the number of models does not match with the number of events ')


def check_lists_of_features_specified_by_user(list_of_events,list_of_lists_relevant_features_specified_by_user,list_of_lists_of_discarded_variables_by_user):
    '''
    It allows to check if the length of the list of events, relevant user features and discarded user features

    Exceptions:
    raise ValueError: in the number of elements in the three list does not match

    Parameters:
    :param list_of_events: list with the events to process
    :param list_of_lists_relevant_features_specified_by_user: list of lists with relevant variables for each event
    :param list_of_lists_of_discarded_variables_by_user: list of list with discarded variables for each event
    '''
    
    if(len(list_of_events) != len(list_of_lists_relevant_features_specified_by_user) != len(list_of_lists_of_discarded_variables_by_user)):
        raise ValueError('The number of events and the number of list of features relevants for the user must match in number')


def checking_percentil_value(percentil_value):
    '''
    It allows to check if the length of the list of events, relevant user features and discarded user features

    Exceptions:
    raise ValueError: in the number of elements in the three list does not match

    parameters:
    :param float percentil_value: value of the percentaje of variables that will be discarded for the selection of relevant features in the mutual information function

    :return percentil_value, checked percentil value
    :rtype: float
    '''

    percentil_value = float(percentil_value)
    if(percentil_value < 0 or percentil_value > 100):
        raise ValueError('The minimum value for the percentil of the selected features by the mutual information function must be in the range [0,1]')            
    return percentil_value


def generate_information_about_current_execution(list_of_events,list_of_learnings,list_of_models,target,percentil,list_of_lists_relevant_variables_by_user,list_of_lists_discarded_variables_by_user):
    learnings_dictionary = generate_learnings_dict()
    gathered_information = ''
    
    for index_of_event in range(len(list_of_events)):                             
        gathered_information+="************Summary of the parameters for the event " + list_of_events[index_of_event] + " and target " + target + " *****************\n"
        
        gathered_information+="*The percentaje of the relevant features selected after applying MIF will be " + str(100 - percentil) + "\n"    
        
        if(list_of_lists_relevant_variables_by_user[index_of_event]):
            gathered_information+="*The user has specified the next features to be considered in the process: \n"
            for variable in list_of_lists_relevant_variables_by_user[index_of_event]:
                gathered_information+="*\t->" + variable + "\n"
            gathered_information+="\n"
        else:
            gathered_information+="*The user has not specified features to be considered in the process \n"
            
        if(list_of_lists_discarded_variables_by_user):
            gathered_information+="*The user has specified the next features to be discarded in the process: \n"
            for variable in list_of_lists_discarded_variables_by_user:
                gathered_information+="*\t->" + variable + "\n"
            gathered_information+="\n"
        else:
            gathered_information+="*The user has not specified features to be discarded in the process \n"
                
        gathered_information+="*The next learning models will be applied to the event " + list_of_events[index_of_event] + " for the target "+ target + ": \n"
        learning = list_of_learnings[index_of_event]
        models = list_of_models[index_of_event]
        if(learning == 'All'):            
            #looping supervised
            gathered_information+="*\t-> " + str(learnings_dictionary[1]) + "\n"
            modelos_sup = models[0]
            for current_model in modelos_sup:
                gathered_information+="*\t\t-> " + current_model + "\n"
                
        else:
            gathered_information+="*\t-> " + learning + "\n"
            for current_model in models:
                gathered_information+="*\t\t-> " + current_model + "\n"            
        gathered_information+= "\n"
            
        gathered_information+="****************************************************************************************************\n"
    return gathered_information
