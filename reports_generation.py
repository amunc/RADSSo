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


import base64
import numpy as np
import os
import codecs
import operator
import xhtml2pdf.pisa as pisa
from jinja2 import Environment, FileSystemLoader

import auxiliary_functions as auxf
import logging

class PisaNullHandler(logging.Handler):
    def emit(self, record):
        pass

def encode_image(path_to_image):
    with open(path_to_image, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())                        
    return "data:image/png;base64,"+ encoded_string                    

def register_log(array_rutas_ficheros,mensaje,opcion,enco):
    modo = 'a'
    f=''
    for ruta_fichero in array_rutas_ficheros:
        if(opcion == 0):
            modo = 'w'
        if(enco !='none'):
            f = codecs.open(ruta_fichero,modo,encoding=enco)
        else:
            f = open(ruta_fichero,modo)
        f.write(mensaje)
        f.close()

def register_target_values_distribution(diccionario_targets_valores,mensaje, array_rutas_ficheros_log,report_dict,enco,opcion=''):
    '''
    This function allows to register in the report_dict the distribution of the targets for the current execution
    
    Parameters:    
    :param pandas_dataframe df_datos: pandas dataframe with all the available data for the current event
    :param target: Current objective feature
    :param str mensaje: Message to register in the log
    :param list array_rutas_ficheros_log: List with the path to the log registers
    :param dict report_dict: Dictionary with the current processed information about the event
    
    :return: updated report_dict
    :rtype: dict<python_hashable_type:python_type>
    '''

    register_log(array_rutas_ficheros_log,mensaje,'',enco)
    for key in sorted(diccionario_targets_valores):
        register_log(array_rutas_ficheros_log, "\ttarget: "+ str(key) +" Number of elements: "+ str(diccionario_targets_valores[key]) + "\n",'',enco)
        if opcion == '':
            report_dict['General_info']['Target'][str(key)] = str(diccionario_targets_valores[key])
        elif opcion == 'Train':
            report_dict = update_train_division(report_dict,str(key), str(diccionario_targets_valores[key]))
        elif opcion == 'Test':
            report_dict = update_test_division(report_dict,str(key), str(diccionario_targets_valores[key]))
    return report_dict
        

def create_basic_report_data_dict(umbral,target,main_metric,lista_variables_descartadas,ruta_logo):
    report_data = {"title": "Overview With Execution Information",
                   "logo":ruta_logo,                   
                   "Umbral": str(umbral),
                   "Main_metric": str(main_metric),
                   "Objective_target": target,
                   "Variables":{'Deleted_by_user':lista_variables_descartadas},
                   "general_information_execution":''
                    }
    return report_data

def create_report_data_dict(evento,umbral,target,lista_variables_descartadas,ruta_logo):
    report_data = {'Objective_target': target,
                   'Event':evento,
                   'Logo':ruta_logo,
                   'General_info':{'Target':{},
                                  'Variables':{'Deleted_by_user':lista_variables_descartadas,'Empty_constant':[],"Mif_relevant":[]},
                                  'Training_division':{},
                                  'Test_division':{},                                  
                                  },
                    'Umbral': str(umbral),                    
                    'Warning_info': ''
                        }
    return report_data

def upddate_report_warning_info(report_dict,informacion):
    report_dict['Warning_info'] = informacion
    return report_dict

def update_report_percentil(report_dict,valor):
    report_dict['Umbral'] = valor
    return report_dict

def update_report_empty_constant_features(report_dict,lista_vacias_constantes):
    report_dict['General_info']['Variables']['Empty_constant'] = lista_vacias_constantes
    return report_dict

def update_report_relevant_user_features(report_dict,lista_importantes):
    report_dict['General_info']['Variables']['User_requested'] = lista_importantes
    return report_dict

def update_report_user_discarded_features(report_dict,lista_descartadas):
    report_dict['General_info']['Variables']['User_discarded'] = lista_descartadas
    return report_dict
    
def update_report_training_models_features(report_dict,diccionario_variables_scores):
    report_dict['General_info']['Variables']["Mif_relevant"] = diccionario_variables_scores
    return report_dict

def update_report_full_list_features_used_in_process(report_dict,lista_variables_in_process):
    report_dict['General_info']['Variables']['Used_in_process'] = lista_variables_in_process
    return report_dict

def update_train_division(report_dict,key,valor):
    report_dict['General_info']['Training_division'][key] = valor
    return report_dict
    
def update_test_division(report_dict,key,valor):
    report_dict['General_info']['Test_division'][key] = valor
    return report_dict

def add_model_to_report(report_dict,modelo):
    report_dict[modelo]={'Parameters':{},
                         'Time_parameters':{},
                         'Accuracy':0,
                         'Decision_index':0,
                         }                       
                         
    return report_dict

def update_model_parameters(report_dict,modelo,parametro,valor):
    report_dict[modelo]['Parameters'][parametro] = valor
    return report_dict

def update_report_model_time(report_dict,modelo,parametro_temporal,valor):
    report_dict[modelo]['Time_parameters'][parametro_temporal] = valor
    return report_dict

def actualizar_accuracy_modelo(report_dict,modelo,valor): #'Accuracy',
    report_dict[modelo]['Accuracy'] = valor
    return report_dict

def update_model_feature(report_dict,modelo,parametro,valor): #'Learning_curve_*_path','Confussion_matrix_train_path','Confussion_matrix_test_path'
    report_dict[modelo][parametro] = "'"+ valor+"'"
    return report_dict

def update_report_current_model_decision_index(report_dict,modelo,valor): #'Learning_curve_*_path','Confussion_matrix_train_path','Confussion_matrix_test_path'
    report_dict[modelo]['Decision_index'] = valor
    return report_dict

def update_report_metrics(report_dict,model_name,array_metrics_vectors,target_recodified_values,list_names_metrics):
    precisions_vector, recalls_vector,specificity_vector,f1_score_vector,mcc_vector,confusion_tables_by_class = array_metrics_vectors
    
    '''Computation of macro avg'''
    precision_macro_avg = 0.0
    number_valid_precision_items = 0.0
    recall_macro_avg = 0.0
    number_valid_recall_items = 0.0
    specificity_macro_avg = 0.0
    number_valid_specificity_items = 0.0
    f1_score_macro_avg = 0.0
    number_valid_f1_score_items = 0.0
    mcc_macro_avg = 0.0
    number_valid_mcc_items = 0.0
    
    report_dict[model_name]['metrics'] = {}
    for ind in range(len(target_recodified_values)):
        temp_dict = {'target_'+str(target_recodified_values[ind]):{list_names_metrics[0]:str(precisions_vector[ind]),
                                                                                          list_names_metrics[1]:str(recalls_vector[ind]),
                                                                                          list_names_metrics[2]:str(specificity_vector[ind]),
                                                                                          list_names_metrics[3]:str(f1_score_vector[ind]),
                                                                                          list_names_metrics[4]:str(mcc_vector[ind])}}
        report_dict[model_name]['metrics'].update(temp_dict)
        try:
            precision_macro_avg+=float(precisions_vector[ind])
            number_valid_precision_items+=1
        except:
            pass
        try:
            recall_macro_avg+=float(recalls_vector[ind])
            number_valid_recall_items+=1
        except:
            pass
        try:
            specificity_macro_avg+=float(specificity_vector[ind])
            number_valid_specificity_items+=1
        except:
            pass
        try:
            f1_score_macro_avg+=float(f1_score_vector[ind])
            number_valid_f1_score_items+=1
        except:
            pass
        try:
            mcc_macro_avg+=float(mcc_vector[ind])
            number_valid_mcc_items+=1
        except:
            pass
    
    #Calculate macro avg of each metric
    try:
        precision_macro_avg = round(precision_macro_avg/number_valid_precision_items,4)        
    except:
        precision_macro_avg = "--"
        
    try:
        recall_macro_avg = round(recall_macro_avg/number_valid_recall_items,4)
    except:
        recall_macro_avg= "--"
    
    try:
        specificity_macro_avg = round(specificity_macro_avg/number_valid_specificity_items,4)
    except:
        specificity_macro_avg ="--"
        
    try:
        f1_score_macro_avg = round(f1_score_macro_avg/number_valid_f1_score_items,4)
    except:
        f1_score_macro_avg = "--"
        
    try:
        mcc_macro_avg=round(mcc_macro_avg/number_valid_mcc_items,4)
    except:
        mcc_macro_avg = "--"
        
    #report_dict[model_name]['metrics_means'] ={list_names_metrics[0]:str(precision_macro_avg),
    report_dict[model_name]['metrics_macro_avg'] ={list_names_metrics[0]:str(precision_macro_avg),
                                                   list_names_metrics[1]:str(recall_macro_avg),
                                                   list_names_metrics[2]:str(specificity_macro_avg),
                                                   list_names_metrics[3]:str(f1_score_macro_avg),
                                                   list_names_metrics[4]:str(mcc_macro_avg)}
    
    
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
    
    
    for ind in range(len(target_recodified_values)):
        #recall
        num_recall+=float(confusion_tables_by_class[ind][0,0]) #TP each class
        den_recall+=float(confusion_tables_by_class[ind][0,0]) + float(confusion_tables_by_class[ind][0,1]) #TP + FN each class
        
        #specificity
        num_specificity+=float(confusion_tables_by_class[ind][1,1]) #TN each class
        den_specificity+=float(confusion_tables_by_class[ind][1,1]) + float(confusion_tables_by_class[ind][1,0]) #TN + FP each class
        
        #precision
        num_precision+=float(confusion_tables_by_class[ind][0,0]) #TP each class
        den_precision+=float(confusion_tables_by_class[ind][0,0]) + float(confusion_tables_by_class[ind][1,0]) #TP + FP each class
        
        total_TP += float(confusion_tables_by_class[ind][0,0])
        total_FN += float(confusion_tables_by_class[ind][0,1])
        total_FP += float(confusion_tables_by_class[ind][1,0])
        total_TN += float(confusion_tables_by_class[ind][1,1])
        
    
    recall_micro_avg = round(num_recall/den_recall,4)
    specificity_micro_avg = round(num_specificity/den_specificity,4)
    precision_micro_avg = round(num_precision/den_precision,4)
    
    f1_score_micro_avg = 2*( (precision_micro_avg*recall_micro_avg)/ (precision_micro_avg + recall_micro_avg) )
    
    mcc_micro_avg = ( (total_TP*total_TN) - (total_FP*total_FN) ) / np.sqrt( (total_TP+total_FP) * (total_TP+total_FN) * (total_TN+total_FP) * (total_TN+total_FN) )
    
    
    report_dict[model_name]['metrics_micro_avg'] ={list_names_metrics[0]:str(precision_micro_avg),
                                                   list_names_metrics[1]:str(recall_micro_avg),
                                                   list_names_metrics[2]:str(specificity_micro_avg),
                                                   list_names_metrics[3]:str(f1_score_micro_avg),
                                                   list_names_metrics[4]:str(mcc_micro_avg)}
        
    return report_dict
    
    
    


def get_string_with_ranking_of_models(lista_modelo_ranking,modelo_actual):
    informacion = "<h3>&nbsp;Models ranking</h3><p>"
    for par_modelo_indice in lista_modelo_ranking:
        modelo = par_modelo_indice[0]
        indice_dec = par_modelo_indice[1]
        if(modelo_actual == modelo):
            informacion+= "&nbsp;&nbsp;<strong>"+modelo+":&nbsp;&nbsp;"+ str(float(round(indice_dec,4))) +"</strong></br>"
        else:
            informacion+= "&nbsp;&nbsp;" + modelo + ":&nbsp;&nbsp;" + str(float(round(indice_dec,4))) + "</br>"
    informacion += "</p>"
        
    return informacion


def create_report_current_execution(report_dict,lista_eventos,lista_variables_usuario,lista_listas_variables_descartadas,lista_aprendizajes,lista_modelos, diccionario_aprendizajes, ruta_relativa_datos_auxiliares, ruta_directorio_resultados,enco):
    env = Environment(loader=FileSystemLoader('.'))
    ruta_plantilla_temporal = os.path.join(ruta_relativa_datos_auxiliares,'temp_html.html')
    template = env.get_template(ruta_relativa_datos_auxiliares + '/' + 'general_execution_template.html')
       
    template_vars = {"title": report_dict['title'],
                     "logo":encode_image(report_dict['logo'].replace('\'','')),                     
                     "general_information_execution":''
                     }
    
    #Parametros generales (target,umbral,variables_descartadas)
    target = report_dict['Objective_target']
    umbral = report_dict['Umbral']
    main_metric = report_dict['Main_metric']
    lista_variables_descartadas = report_dict['Variables']['Deleted_by_user']
    
    tabulacion = "&nbsp;&nbsp;&nbsp;&nbsp;"
    informacion= "<h3>Common Parameters </h3></p>"
    informacion+= tabulacion+tabulacion + "<i>Objective Target: </i>" + target + "</br></br>"
    informacion+=tabulacion+tabulacion + "<i>Threshold for Mutual Information Function: </i>" + umbral + "</br></br>"
    informacion+=tabulacion+tabulacion + "<i>Main metric: </i>" + main_metric + "</br></br>"    
    informacion+=tabulacion+tabulacion + "<i>Common Discarded Variables:</i></br>"
    for variable_descartada in lista_variables_descartadas:
        informacion+=tabulacion+tabulacion+tabulacion + variable_descartada + "</br>"
    if(lista_variables_descartadas == []):
        informacion+=tabulacion+"No variables were selected to be discarded</br>"
    informacion+="</p>"
    
    
    #Parametros de cada evento
    informacion+= "<h3>Events to be processed: </h3><p>"
    for indice in range(len(lista_eventos)):
        informacion+=tabulacion+"<strong>"+ lista_eventos[indice] + "</strong></br>"        
        informacion+=tabulacion+tabulacion+"<i>Important features for the user:</i> </br>"
        if(lista_variables_usuario[indice]):            
            for variable in lista_variables_usuario[indice]:
                informacion+=tabulacion+tabulacion+tabulacion+variable + "</br>"
        else:
            informacion+=tabulacion+tabulacion+tabulacion+"No important features were specified</br>"
        informacion+="</br>"
        
        informacion+=tabulacion+tabulacion+"<i>Discarded variables by the user:</i> </br>"
        if(lista_listas_variables_descartadas[indice]):            
            for variable in lista_listas_variables_descartadas[indice]:
                informacion+=tabulacion+tabulacion+tabulacion+variable + "</br>"
        else:
            informacion+=tabulacion+tabulacion+tabulacion+"No variables were discarded</br>"
        informacion+="</br>"
        
        informacion+=tabulacion+tabulacion+"<i>Learnings to be applied: </i></br>"
        aprendizaje = lista_aprendizajes[indice]
        modelos = lista_modelos[indice]
        if(aprendizaje == 'All'):            
            #recorremos los supervisados
            informacion+=tabulacion+tabulacion+tabulacion+"<u>"+str(diccionario_aprendizajes[1]) + "</u>:</br>"
            modelos_sup = modelos[0]
            for modelo_act in modelos_sup:
                informacion+=tabulacion+tabulacion+tabulacion+tabulacion + modelo_act + "</br>"
            informacion+="</br>"
            
        else:
            informacion+=tabulacion+tabulacion+tabulacion+"<u>"+aprendizaje + "</u>:</br>"
            for modelo_act in modelos:
                informacion+=tabulacion+tabulacion+tabulacion+tabulacion + modelo_act + "</br>"        
            
        informacion+="</p>"
        
        template_vars["general_information_execution"] = informacion        
                    
    with codecs.open(ruta_plantilla_temporal,'w',encoding=enco) as output_file:
        output_file.write(template.render(template_vars))
                
    with codecs.open(ruta_plantilla_temporal, 'r') as html_leido:
        pdf_resultante=os.path.join(ruta_directorio_resultados,"General_execution_report_"+ target +".pdf")
        with open(pdf_resultante, "w") as gen_report:
            pisa.CreatePDF(html_leido.read(),gen_report)
            logging.getLogger("xhtml2pdf").addHandler(PisaNullHandler())
            
    if(os.path.exists(ruta_plantilla_temporal)):
        os.remove(ruta_plantilla_temporal)
    
                
def create_report_current_model(report_dict,lista_modelos,ruta_relativa_datos_auxiliares,ruta_directorio_informes,enco):

    env = Environment(loader=FileSystemLoader('.'))
    ruta_plantilla_temporal = os.path.join(ruta_relativa_datos_auxiliares,'temp_html.html')    
    
    if(lista_modelos == []): #if process not completed
        template = env.get_template(ruta_relativa_datos_auxiliares + '/' + 'incomplete_event_report_template.html') #usamos la plantilla de informes incompletos
        
        template_vars = {"title": "Incomplete Execution Report",
                         "logo": encode_image(report_dict['Logo'].replace('\'','')),
                         "target": report_dict['Objective_target'],
                         "event": report_dict['Event'],
                         "info": "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" + report_dict['Warning_info']
                         }
        
        #html
        with codecs.open(ruta_plantilla_temporal,'w',encoding=enco) as output_file:
            output_file.write(template.render(template_vars))        
        
        #pdf
        with codecs.open(ruta_plantilla_temporal, 'r') as html_leido:
            pdf_resultante=os.path.join(ruta_directorio_informes,"report_"+report_dict['Event']+"_incomplete.pdf")
            with open(pdf_resultante, "wb") as incomplete_rep:
                pisa.CreatePDF(html_leido.read(),incomplete_rep)
                logging.getLogger("xhtml2pdf").addHandler(PisaNullHandler())
        
    else:
        lista_pares_modelo_indice = auxf.order_models_by_score_and_time(report_dict,lista_modelos)        
        template = env.get_template(ruta_relativa_datos_auxiliares + '/' +'report_template.html') #usamos la plantilla estandar de los informes
        for modelo in lista_modelos:
            if(modelo in report_dict):
            
                observations_targets="<p><strong>Target distribution of observations</strong></br>"
                for ob_target in auxf.natsorted(report_dict['General_info']['Target'].keys()):
                    observations_targets+="&nbsp;&nbsp;&nbsp;&nbsp;"+ "With target " + str(ob_target) + " :"+ str(report_dict['General_info']['Target'][ob_target]) + "</br>"
                observations_targets+="</p>"
            
                variables_summary="<p><strong>Summary of variables</strong></br>"
            
            
                discarded_for_event = report_dict['General_info']['Variables']['User_discarded']
                
                variables_summary+="<br><i><u>Deleted by the user at the begining:</i></u></br>"
                for deleted_var in report_dict['General_info']['Variables']['Deleted_by_user']:
                    variable_dis=''
                    if deleted_var in discarded_for_event:
                        variable_dis = "<strong>" + deleted_var + "</strong>"
                    else:
                        variable_dis = deleted_var
                    variables_summary+="&nbsp;&nbsp;&nbsp;&nbsp;"+ variable_dis + "</br>"
                variables_summary+="&nbsp;&nbsp;&nbsp;&nbsp;<i>*variables in bold were specified by the user to be discarded specifically for this event<i></br>"
                variables_summary+="</br>"
                                                
                variables_summary+="<br><i><u>Deleted in execution time(Empty or Constant):</i></u></br>"
                for emp_con_var in report_dict['General_info']['Variables']['Empty_constant']:
                    variables_summary+="&nbsp;&nbsp;&nbsp;&nbsp;"+ emp_con_var + "</br>"
                variables_summary+="</br>"
                
                variables_summary+="<br><i><u>Requested for the event by the user:</i></u></br>"
                for req_var in report_dict['General_info']['Variables']['User_requested']:
                    variables_summary+="&nbsp;&nbsp;&nbsp;&nbsp;"+ req_var + "</br>"
                variables_summary+="</br>"
                                       
                variables_summary+="<br><i><u>Used during the process:</i></u></br>"
                
                diccionario_relevantes_mif = report_dict['General_info']['Variables']['Mif_relevant']
                sorted_relevant_vars = sorted(diccionario_relevantes_mif.items(), key=operator.itemgetter(1), reverse=True)
                for relevant_var in sorted_relevant_vars:
                    rel_variable= relevant_var[0]
                    rel_variable = "<strong>" + rel_variable +'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'+ str(diccionario_relevantes_mif[rel_variable]) +"</strong>"
                    variables_summary+="&nbsp;&nbsp;&nbsp;&nbsp;"+ rel_variable + "</br>"
                
                for relevant_var in report_dict['General_info']['Variables']['Used_in_process']:
                    if (relevant_var not in diccionario_relevantes_mif)   :
                        variables_summary+="&nbsp;&nbsp;&nbsp;&nbsp;"+ relevant_var + "</br>"
                variables_summary+="&nbsp;&nbsp;&nbsp;&nbsp;<i>*variables in bold were used to train the models<i></br>"
                variables_summary+="</p>"
            
            
                #Information about the model                    
                accuracy = "</br></br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<strong>Accuracy: "+ str(float(round(report_dict[modelo]['Accuracy'],5)))+"</strong>"
            
                ranking = get_string_with_ranking_of_models(lista_pares_modelo_indice,modelo)
                
                model_info = "<p><strong>Parameters used to configure the model</strong></br>"
                for param in report_dict[modelo]['Parameters']:
                    model_info+= "&nbsp;&nbsp;&nbsp;&nbsp;<i>"+ param + "</i>: " + str(report_dict[modelo]['Parameters'][param]) + "</br>"
                model_info+="</p>"
                
                time_info = "<p><strong>Time elapsed</strong></br>"
                tiempo_seleccion_parametros = report_dict[modelo]['Time_parameters']['time_sel_finish'] - report_dict[modelo]['Time_parameters']['time_sel_init']
                tiempo_entrenamiento = report_dict[modelo]['Time_parameters']['time_train_finish'] - report_dict[modelo]['Time_parameters']['time_train_init']
                time_info+="&nbsp;&nbsp;&nbsp;&nbsp;"+ "Parameters selection time: "+ str(tiempo_seleccion_parametros) + "</br>"
                time_info+="&nbsp;&nbsp;&nbsp;&nbsp;"+ "Training time: "+ str(tiempo_entrenamiento) + "</br>"
                time_info+="</p>"
                
                
                total_train = 0.0
                vector_of_targets = []
                vector_of_values_by_target = []
                vector_of_percentages_by_target = []
                train_distribution_info ="<p></br><strong>Training Data Distribution</strong></br>"
                for train_target in auxf.natsorted(report_dict['General_info']['Training_division'].keys()):
                    train_distribution_info+="&nbsp;&nbsp;&nbsp;&nbsp;"+ "With target " + str(train_target) + " :"+ str(report_dict['General_info']['Training_division'][train_target]) + "</br>"
                    vector_of_targets.append(train_target)
                    vector_of_values_by_target.append(float(report_dict['General_info']['Training_division'][train_target]))
                    total_train+=float(report_dict['General_info']['Training_division'][train_target])
                train_distribution_info+="</p>"
                #getting null train accuracy
                null_train_accuracy = 0.0
                for indice_t in range(len(vector_of_values_by_target)):
                    vector_of_percentages_by_target.append(round(vector_of_values_by_target[indice_t]/total_train,4))
                
                null_train_accuracy = max(vector_of_percentages_by_target)
                
                                            
                total_test = 0.0
                vector_of_targets = []
                vector_of_values_by_target = []
                vector_of_percentages_by_target = []
                test_distribution_info ="<p><strong>Test Data Distribution</strong></br>"
                for test_target in auxf.natsorted(report_dict['General_info']['Test_division'].keys()):
                    test_distribution_info+="&nbsp;&nbsp;&nbsp;&nbsp;"+ "With target " + str(test_target) + " :"+ str(report_dict['General_info']['Test_division'][test_target]) + "</br>"
                    vector_of_targets.append(test_target)
                    vector_of_values_by_target.append(float(report_dict['General_info']['Test_division'][test_target]))
                    total_test+=float(report_dict['General_info']['Test_division'][test_target])
                test_distribution_info+="</p>"
                null_test_accuracy = 0.0
                for indice_t in range(len(vector_of_values_by_target)):
                    vector_of_percentages_by_target.append(round(vector_of_values_by_target[indice_t]/total_test,4))
                
                null_test_accuracy = max(vector_of_percentages_by_target)
                
                           
                event = report_dict['Event']
                template_vars = {"title": "Execution Report",
                             "logo":encode_image(report_dict['Logo'].replace('\'','')),                             
                             "model": modelo,
                             "target": report_dict['Objective_target'],
                             "event": event,#.decode('utf-8'),
                             "accuracy": str(accuracy)+"<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<strong>Null train acc: "+ str(null_train_accuracy)+"</strong>"+"<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<strong>Null test acc: "+ str(null_test_accuracy)+ "</strong></p>",
                             "models_ranking": ranking,
                             "observations_targets": observations_targets,
                             "variables_summary": variables_summary,
                             "model_info" : model_info,
                             "time_info": time_info,
                             "train_distribution_info": train_distribution_info,
                             "test_distribution_info": test_distribution_info
                         }
                template_vars["metrics_info"] =""
                for metric in report_dict[modelo]['metrics_micro_avg']:
                    template_vars["metrics_info"] +="<p>"+"<strong>"+metric+"</strong>: " + report_dict[modelo]['metrics_micro_avg'][metric] +"</br>"
                template_vars["metrics_info"] +="</p>"
                #cargamos las imagenes en las variables del reporte
                if 'parameters_plot' in report_dict[modelo]:
                    template_vars['image_parameters_accuracy'] = encode_image(report_dict[modelo]['parameters_plot'].replace('\'',''))
                
                if 'Confussion_matrix_train_path' in report_dict[modelo]:
                    template_vars["conf_train_img"] = encode_image(report_dict[modelo]['Confussion_matrix_train_path'].replace('\'',''))
                    
                if 'Confussion_matrix_test_path' in report_dict[modelo]:
                    template_vars["conf_test_img"] = encode_image(report_dict[modelo]['Confussion_matrix_test_path'].replace('\'',''))
            
                if('lc_0_0' in report_dict[modelo]):                
                    template_vars['lc_0_0'] = encode_image(report_dict[modelo]['lc_0_0'].replace('\'',''))
                
                
                metrics_by_label = "<table width='100%' border='1' cellspacing='0' cellpadding='5'>"
                keys = ''
                for elemento in auxf.natsorted(report_dict[modelo]['metrics'].keys()):
                    if(keys == ''):
                        keys = report_dict[modelo]['metrics'][elemento].keys()
                        metrics_by_label+="<tr><td align='center' class='black'>"+ 'Target' +"</td>"
                        for cabecera in keys:                        
                            metrics_by_label+="<td align='center' class='black'>" + cabecera +"</td>"
                        metrics_by_label += "</tr>"
                    metrics_by_label+= "<tr><td>" + elemento.replace('target_','') + "</td>"
                    for key in keys:
                        metrics_by_label += "<td>"+str(report_dict[modelo]['metrics'][elemento][key])+"</td>"
                    metrics_by_label+= "</tr>"
                metrics_by_label+="</table>"
                template_vars['metrics_by_label'] = metrics_by_label
                
                #generamos el html
                with codecs.open(ruta_plantilla_temporal,'w',encoding=enco) as output_file:
                    output_file.write(template.render(template_vars))
                                        
                #generamos el pdf
                with codecs.open(ruta_plantilla_temporal, mode='r',encoding=enco) as read_html:
                    pdf_resultante=os.path.join(ruta_directorio_informes,modelo + "_report_for_"+ event +".pdf")
                    with open(pdf_resultante, mode='w') as pdf_gen:                                             
                        pisa.CreatePDF(read_html.read(),pdf_gen)                        
                        logging.getLogger("xhtml2pdf").addHandler(PisaNullHandler())
    
    if(os.path.exists(ruta_plantilla_temporal)):
        os.remove(ruta_plantilla_temporal)
        
    
def create_report_current_dictionary_models(dictionary_of_models,ruta_relativa_datos_auxiliares,ruta_directorio_resultados,list_of_parameters_models_events_dict,logo_path,enco):
    env = Environment(loader=FileSystemLoader('.'))
    ruta_plantilla_temporal = os.path.join(ruta_relativa_datos_auxiliares,'temp_html.html')
    template = env.get_template(ruta_relativa_datos_auxiliares + '/' + 'dictionary_models_template.html')
    
    tabulacion = "&nbsp;&nbsp;&nbsp;&nbsp;"
    
    template_vars = {"title": "Report of the information of the Dictionary of models",
                     "logo": encode_image(logo_path.replace('\'',''))
                     }
           
    #['learning','model_path','dict_reassignment','features','original_features']
    list_of_parameters_models_events_dict
    list_elements = [list_of_parameters_models_events_dict[0],list_of_parameters_models_events_dict[3],list_of_parameters_models_events_dict[4],list_of_parameters_models_events_dict[1]]
    informacion= ""
    for event in dictionary_of_models:
        informacion+= "<strong><u>"+ event +"</u></strong></br></br>"
        for target in dictionary_of_models[event]:
            informacion+= tabulacion + tabulacion + "<strong><i>Target:</i></strong>" + "&nbsp;&nbsp;" + target + "</br>"
            for key in list_elements:
                informacion+=tabulacion + tabulacion + "<strong><i>" + key + ": </i></strong>"
                if(type(list()) == type(dictionary_of_models[event][target][key])):
                    informacion+="<br>"
                    contador = 0
                    ordered_list_features = sorted(dictionary_of_models[event][target][key])
                    while(contador < len(ordered_list_features)):
                        element = ordered_list_features[contador]
                        '''if(contador != len(ordered_list_features)-1):
                            informacion+=element + ","
                            if((contador+1)>=3) and ((contador+1)%3 == 0):
                                informacion+="</br>" + tabulacion + tabulacion + tabulacion +tabulacion
                        else:
                            informacion+=element'''
                        informacion+=tabulacion + tabulacion + tabulacion +tabulacion + element + "</br>"
                        contador+=1                    
                else:                                            
                    informacion+= dictionary_of_models[event][target][key] + "</br>"
                    if(key == list_of_parameters_models_events_dict[0]):
                        informacion+= tabulacion + tabulacion + "<strong><i>best model: </i></strong>&nbsp;&nbsp;" + dictionary_of_models[event][target][list_of_parameters_models_events_dict[1]].split('_')[-1].split('.')[0] + "</br>" #get model name
                        if(dictionary_of_models[event][target][key] == 'Unsupervised'):
                            informacion+= tabulacion + tabulacion + "<strong><i>dic_reassingment: </i></strong>&nbsp;&nbsp;" + str(dictionary_of_models[event][target][list_of_parameters_models_events_dict[2]]) + "</br>"
            informacion+="</br>"
        
    
    if(informacion == ""):
        informacion = "No models were created yet"
    template_vars["info"] = informacion
    #generamos el html
    with codecs.open(ruta_plantilla_temporal,'w',encoding='utf-8') as output_file:
        renderizado = template.render(template_vars)                                
        output_file.write(renderizado)
                    
                #generamos el pdf
    with codecs.open(ruta_plantilla_temporal, mode='r',encoding=enco) as read_html:
        pdf_resultante=os.path.join(ruta_directorio_resultados,"Current_status_dictionary_events_and_models.pdf")
        with open(pdf_resultante, mode='w') as pdf_gen:                                             
            pisa.CreatePDF(read_html.read().encode(enco, 'ignore').decode(enco),pdf_gen)
    
    if(os.path.exists(ruta_plantilla_temporal)):
        os.remove(ruta_plantilla_temporal)        

    ###################Funciones realtivas a datos del reporte en html #####################


def create_mif_report(directorio_salida,nombre_informe,report,percentil,enco):
    try:
        nombre_informe='MIF_'+ nombre_informe + '_report_.csv'
        ruta_relativa_informe= os.path.join(directorio_salida,nombre_informe)
        with codecs.open(ruta_relativa_informe,'w',encoding=enco) as g:
            cabecera = u'variables'+u','+u'peso'+u'\n'
            g.write(cabecera)
            for elemento in report:        
                variable_act = elemento[0]
            
                variable_act
                
                score = elemento[1]
                g.write(variable_act  + u',' + unicode(str(score),enco) +u'\n')
        mensaje = "Mif report created succesfully"
    except Exception as e:
        print e
        pass

    return mensaje

def save_data_to_file(datos,ruta_directorio,nombre_fichero,delimitador,extension):
            
    if(extension == 'txt'):
        nombre_fichero = nombre_fichero + "." + extension
        ruta_destino = os.path.join(ruta_directorio,nombre_fichero)
        np.savetxt(ruta_destino, datos, delimiter=delimitador)
    elif(extension == 'csv'):
        nombre_fichero = nombre_fichero + "." + extension
        ruta_destino = os.path.join(ruta_directorio,nombre_fichero)
        datos.to_csv(ruta_destino,index=False,sep=delimitador)   
    elif(extension == 'numpy'):
        nombre_fichero = nombre_fichero + "." + 'txt'
        ruta_destino = os.path.join(ruta_directorio,nombre_fichero)
        f = open(ruta_destino,'w')
        f.write("\t Predicted label \n")        
        filas = datos.tolist()
        for fila in filas:
            f.write('\t' + str(fila) + '\n')
        f.close()
        #np.save(ruta_destino, datos)

def generate_model_report(porc_acierto_test,porc_acierto_validacion,ruta_directorio_informes_accuracy,nombre_informe):
    datos_accuracy=np.array([porc_acierto_test,porc_acierto_validacion])
    save_data_to_file(datos_accuracy,ruta_directorio_informes_accuracy,nombre_informe,'','txt')