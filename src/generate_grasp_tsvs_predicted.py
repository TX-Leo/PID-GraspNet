import re
import json
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer

from src.generate_grasp_tsvs_real import load_grasp_txt_real

def generate_grasp_tsvs_predicted_encoded(graspnet_root,train_or_test,camera,format,scene_sum,sceneId_start,sceneId_end,annId_start,annId_end):
    '''    
    @Function: get all predicted uncoded grasp info data from json files and save them in tsv files.
            
    @Input:
        - graspnet_root: ROOT PATH FOR GRASPNET.
        - train_or_test: string, a flag of "train" or "test".
        - camera: string of type of camera, "realsense" or "kinect".
        - format: string of grasp format, '6d' or 'rect'.
        - scene_sum: int of the number of scenes in the dataset.
        - sceneId_start: int of the starting scene index.(sceneId max range is [0:180])
        - sceneId_end: int of the ending scene index.
        - annId_start: int of the starting annotation index.(annId max range is [0:256])
        - annId_end: int of the ending annotation index.

    @Output: None
        - grasp_tsvs_predicted 
            - grasp_tsvs_predicted_kinect_6d_100
                - 0000
                - 0001
                    - 0000
                    - 0001
                        - grasp_05_00.tsv
                        - grasp_05_01.tsv

                        ....
                        - grasp_05_09.tsv
                    ....
                    - 0256

                ....
                - 0099
    '''
    
    def generate_object_string(object_dict):
        ''' generate the textual description of all objects from an object_dict '''
        object_names = []
        for obj_id, obj_name in object_dict.items():
            article = 'a'
            if obj_name[0] in ['a', 'e', 'i', 'o', 'u']:
                article = 'an'
            object_names.append(f"{article} {obj_name}")
        if len(object_names) == 1:
            return object_names[0]
        elif len(object_names) == 2:
            return f"{object_names[0]} and {object_names[1]}"
        else:
            last_object = object_names[-1]
            remaining_objects = ', '.join(object_names[:-1])
            return f"{remaining_objects} and {last_object}"

    def process_response(response,format):
        ''' process the response to get the predicted grasp info '''
        if format == '6d':
            s_predicted = re.findall(r"s_(\d+)", response)[0] if re.findall(r"s_(\d+)", response) else '__'
            w_predicted = re.findall(r"w_(\d+)", response)[0] if re.findall(r"w_(\d+)", response) else '__'
            h_predicted = re.findall(r"h_(\d+)", response)[0] if re.findall(r"h_(\d+)", response) else '__'
            d_predicted = re.findall(r"d_(\d+)", response)[0] if re.findall(r"d_(\d+)", response) else '__'
            t1_predicted = re.findall(r"t1_(\d+)", response)[0] if re.findall(r"t1_(\d+)", response) else '__'
            t2_predicted = re.findall(r"t2_(\d+)", response)[0] if re.findall(r"t2_(\d+)", response) else '__'
            t3_predicted = re.findall(r"t3_(\d+)", response)[0] if re.findall(r"t3_(\d+)", response) else '__'
            r1_predicted = re.findall(r"r1_(\d+)", response)[0] if re.findall(r"r1_(\d+)", response) else '__'
            r2_predicted = re.findall(r"r2_(\d+)", response)[0] if re.findall(r"r2_(\d+)", response) else '__'
            r3_predicted = re.findall(r"r3_(\d+)", response)[0] if re.findall(r"r3_(\d+)", response) else '__'
            predicted_grasp = [s_predicted,w_predicted,h_predicted,d_predicted,t1_predicted,t2_predicted,t3_predicted,r1_predicted,r2_predicted,r3_predicted]
        
        elif format == 'rect':
            s_predicted = re.findall(r"s_(\d+)", response)[0] if re.findall(r"s_(\d+)", response) else '__'
            h_predicted = re.findall(r"h_(\d+)", response)[0] if re.findall(r"h_(\d+)", response) else '__'
            cp1_predicted = re.findall(r"cp1_(\d+)", response)[0] if re.findall(r"cp1_(\d+)", response) else '__'
            cp2_predicted = re.findall(r"cp2_(\d+)", response)[0] if re.findall(r"cp2_(\d+)", response) else '__'
            op1_predicted = re.findall(r"op1_(\d+)", response)[0] if re.findall(r"op1_(\d+)", response) else '__'
            op2_predicted = re.findall(r"op2_(\d+)", response)[0] if re.findall(r"op2_(\d+)", response) else '__'
            predicted_grasp = [s_predicted,h_predicted,cp1_predicted,cp2_predicted,op1_predicted,op2_predicted]
            
        return predicted_grasp
    
    # ====== get the text_templates ======      
    text_templates_file_path = graspnet_root+'/else/text_templates_'+format+'_10_for_eval.json'
    with open(text_templates_file_path,'r') as json_file:
        text_templates = json.load(json_file)
    
    # ====== for every scene ======
    for sceneId in list(range(sceneId_start, sceneId_end)):

        # ====== get all_obj_names_and_ids_dict and all_obj_names for this scene ======
        with open(graspnet_root + '/grasp_jsons/' + camera + '/' + format + '/{:04d}/'.format(sceneId)+'/all_obj_names_and_ids_dict.json','r') as json_file:
            all_obj_names_and_ids_dict = json.load(json_file)
        all_obj_names = generate_object_string(all_obj_names_and_ids_dict)
        
        # ====== for every ann ======
        for annId in list(range(annId_start, annId_end)):   
            
            # ====== grasp_tsvs_predicted_for_one_ann_in_one_scene_path ======
            grasp_tsvs_predicted_for_one_ann_in_one_scene_path = graspnet_root + '/grasp_tsvs_predicted/grasp_tsvs_predicted_'+camera+'_'+format+'_'+str(scene_sum)+ '/{:04d}/'.format(sceneId)+ '/{:04d}'.format(annId)  
            if not os.path.exists(grasp_tsvs_predicted_for_one_ann_in_one_scene_path):
                os.makedirs(grasp_tsvs_predicted_for_one_ann_in_one_scene_path)
            
            # ====== get img_path (for this ann in this scene)======
            img_path = graspnet_root+'/scenes/scene_' +'{:04d}'.format(sceneId) + '/' + camera + '/rgb'+ '/{:04}'.format(annId)+'.png'
            
            # ====== loop through all objects ====== 
            for object_id in all_obj_names_and_ids_dict.keys():
                object_name = all_obj_names_and_ids_dict[object_id]  
                
                # ====== get text_input for one object(10) ======
                for grasp_number in range(10):
                    # ====== get FLAG ======
                    FLAG = '0'
                    FLAG += '00' if train_or_test == 'train' else ('01' if train_or_test == 'test' else None)
                    FLAG += '00' if camera == 'kinect' else ('01' if camera == 'realsense' else None)
                    FLAG += '00' if format == '6d' else ('01' if format == 'rect' else None)
                    FLAG += '{:04d}'.format(sceneId)
                    FLAG += '{:04d}'.format(annId)
                    FLAG += '{:02d}'.format(int(object_id))
                    FLAG += '{:02d}'.format(grasp_number)

                    # ====== get one text ======
                    # random_number = random.randint(1, 30)
                    text = text_templates[str(grasp_number+1)]
                    text = text.replace("{all_obj_names}", str(all_obj_names))
                    text = text.replace("{object_name}", 'The '+str(object_name))
                    # print(f'====== original text is:======\n{text}')
                    
                    # ====== get predicted data ======
                    _,_,_,_,_,response = generate_predictions(image_input=img_path, text_input=text,prompt_text_input=None)
                    # response = "The grasp score is s_0. The grasp width is w_99. The grasp height is h_0. The grasp depth is d_100. The grasp translation is t1_11 t2_22 t3_33. The grasp rotation is r1_111 r2_222 r3_333."
                    predicted_encoded_data = process_response(response,format) # 6d:[0,99,0,100,11,22,33,111,222,333]
                    predicted_encoded_data.insert(0, FLAG) # add FLAG to the top
                    
                    # ====== write the predicted data in tsv file ======
                    grasp_tsvs_predicted_encoded_file_path = grasp_tsvs_predicted_for_one_ann_in_one_scene_path + '/grasp_{:02d}_{:02d}.tsv'.format(int(object_id),int(grasp_number))
                    with open(grasp_tsvs_predicted_encoded_file_path,'w',newline='') as tsv_file:
                        writer = csv.writer(tsv_file, delimiter='\t')
                        writer.writerow(predicted_encoded_data) 
                    
                    # print(f'====[scene_{sceneId},ann_{annId},object_id_{object_id},grasp_number_{grasp_number}] grasp tsvs predicted encoded are generated successfully!!! ====')
                # print(f'====[scene_{sceneId},ann_{annId},object_id_{object_id} grasp tsvs predicted encoded are generated successfully!!! ====')
            print(f'====[scene_{sceneId},ann_{annId}] grasp tsvs predicted encoded are generated successfully!!! ====')


def load_grasp_txt_predicted(grasp_tsvs_predicted_file_path,uncoded_or_encoded,format):  
    '''    
    @Function: load all grasp predicted_uncoded or predicted_encoded data from txt file
            
    @Input:
        - grasp_tsvs_predicted_file_path: the path of the floder: "the grasp_tsvs_predicted"
        - uncoded_or_encoded: string, load grasp predicted_uncoded data if set to "uncoded", load grasp predicted_encoded data if set to "encoded"

    @Output: 
        - all data: a list of all grasp data(score,width,height,depth,translation_1,translation_2,translation_3,rotation_1,rotation_2,rotation_3)
    '''

    def open_grasp_txt_predicted(caption,txt_file_path,uncoded_or_encoded):
        ''' get data from the txt file'''
        with open(txt_file_path, "r") as file:
            lines = file.readlines()
            data = np.array([float(line.strip()) for line in lines])
            file.close()
        print(f'====== {caption} {uncoded_or_encoded} txt file is opened successfully!!! ======')
        return data 
    
    if format == '6d':
        all_score = open_grasp_txt_predicted(caption='[Score]',txt_file_path = grasp_tsvs_predicted_file_path+'/all_score_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        all_width = open_grasp_txt_predicted(caption='[Width]',txt_file_path = grasp_tsvs_predicted_file_path+'/all_width_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        all_height = open_grasp_txt_predicted(caption='[Height]',txt_file_path = grasp_tsvs_predicted_file_path+'/all_height_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        all_depth = open_grasp_txt_predicted(caption='[Depth]',txt_file_path = grasp_tsvs_predicted_file_path+'/all_depth_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        all_translation_1 = open_grasp_txt_predicted(caption='[Translation-1]',txt_file_path = grasp_tsvs_predicted_file_path+'/all_translation_1_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        all_translation_2 = open_grasp_txt_predicted(caption='[Translation-2]',txt_file_path = grasp_tsvs_predicted_file_path+'/all_translation_2_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        all_translation_3 = open_grasp_txt_predicted(caption='[Translation-3]',txt_file_path = grasp_tsvs_predicted_file_path+'/all_translation_3_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        all_rotation_1 = open_grasp_txt_predicted(caption='[Rotation-1]',txt_file_path = grasp_tsvs_predicted_file_path+'/all_rotation_1_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        all_rotation_2 = open_grasp_txt_predicted(caption='[Rotation-2]',txt_file_path = grasp_tsvs_predicted_file_path+'/all_rotation_2_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        all_rotation_3 = open_grasp_txt_predicted(caption='[Rotation-3]',txt_file_path = grasp_tsvs_predicted_file_path+'/all_rotation_3_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        print(f'====== There are {len(all_width)} data totally!!! ======') 

        all_data = [all_score,all_width,all_height,all_depth,all_translation_1,all_translation_2,all_translation_3,all_rotation_1,all_rotation_2,all_rotation_3]

    elif format == 'rect':
        all_score = open_grasp_txt_predicted(caption='[Score]',txt_file_path = grasp_tsvs_predicted_file_path+'/all_score_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        all_height = open_grasp_txt_predicted(caption='[Height]',txt_file_path = grasp_tsvs_predicted_file_path+'/all_height_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        all_center_point_1 = open_grasp_txt_predicted(caption='[Center-Point-1]',txt_file_path = grasp_tsvs_predicted_file_path+'/all_center_point_1_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        all_center_point_2 = open_grasp_txt_predicted(caption='[Center-Point-2]',txt_file_path = grasp_tsvs_predicted_file_path+'/all_center_point_2_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        all_open_point_1 = open_grasp_txt_predicted(caption='[Open-Point-1]',txt_file_path = grasp_tsvs_predicted_file_path+'/all_open_point_1_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        all_open_point_2 = open_grasp_txt_predicted(caption='[Open-Point-2]',txt_file_path = grasp_tsvs_predicted_file_path+'/all_open_point_2_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        print(f'====== There are {len(all_width)} data totally!!! ======') 

        all_data = [all_score,all_height,all_center_point_1,all_center_point_2,all_open_point_1,all_open_point_2]
    
    return all_data

def generate_grasp_txt_predicted_encoded(graspnet_root,camera,format,scene_sum):
    '''    
    @Function: save all grasp predicted_encoded data from tsv files in one txt file
            
    @Input:
        - graspnet_root: ROOT PATH FOR GRASPNET
        - camera: string of type of camera, "realsense" or "kinect".
        - format: string of grasp format, '6d' or 'rect'.
        - scene_sum: int of the number of scenes in the dataset.

    @Output: None
        - all_score_encoded.txt
        - all_width_encoded.txt
        - all_height_encoded.txt
        - all_depth_encoded.txt
        - all_translation_1_encoded.txt
        - all_translation_2_encoded.txt
        - all_translation_3_encoded.txt
        - all_rotation_1_encoded.txt
        - all_rotation_2_encoded.txt
        - all_rotation_3_encoded.txt
    '''

    grasp_tsvs_predicted_file_path=graspnet_root + '/grasp_tsvs_predicted/grasp_tsvs_predicted_'+camera+'_'+format+'_'+str(scene_sum)
    
    def save_grasp_txt_predicted(data,caption,txt_file_path):
        ''' save data in txt file '''
        with open(txt_file_path, "w") as file:
            for item in data:          
                file.write(str(item)+"\n")
            file.close()
        print(f'====== {caption} txt file is created successfully!!! ======') 
        
    # ====== all grasp data ======
    if format == '6d':
        all_score_encoded = []
        all_width_encoded = []
        all_height_encoded = []
        all_depth_encoded = []
        all_translation_1_encoded = []
        all_translation_2_encoded = []
        all_translation_3_encoded = []
        all_rotation_1_encoded = []
        all_rotation_2_encoded = []
        all_rotation_3_encoded = []
    
    elif format == 'rect':
        all_score_encoded = []
        all_height_encoded = []
        all_center_point_1_encoded = []
        all_center_point_2_encoded = []
        all_open_point_1_encoded = []
        all_open_point_2_encoded = []

    data_num = 0 
    data_sum = 0
    data_num_and_sum = {}
    # ====== get grasp predicted data from tsv files ======
    for root,dirs,files in os.walk(grasp_tsvs_predicted_file_path): # grasp_tsvs_predicted_file_path/
        dirs.sort()
        for dir in dirs:
            dir_path = os.path.join(root,dir) # grasp_tsvs_predicted_file_path/0000/
            print(f'====== {dir_path} is looping through!!! ======') 
            for sub_root,sub_dirs,sub_files in os.walk(dir_path):
                sub_dirs.sort()
                for sub_dir in sub_dirs:
                    sub_dir_path = os.path.join(sub_root,sub_dir) # grasp_tsvs_predicted_file_path/0000/0000
                    # print(f'====== {sub_dir_path} is looping through!!! ======') 
                    for sub_sub_root,sub_sub_dirs,sub_sub_files in os.walk(sub_dir_path):           
                        sub_sub_files.sort()
                        for file in sub_sub_files:
                            tsv_file_path = os.path.join(sub_sub_root,file) # grasp_tsvs_predicted_file_path/0000/0000/grasp_05_00.tsv
                            if not tsv_file_path.endswith(".tsv"):
                                continue
                            # print(f'====== {tsv_file_path} is looping through!!! ======') 
                            with open(tsv_file_path,'r') as tsv_file:
                                for row_num, row in enumerate(tsv_file):
                                    data_num += 1
                                    data_sum += 1
                                    tsv_grasp_info_list = row.strip().split('\t')    
                                    # FLAG = tsv_grasp_info_list[0]
                                    
                                    if format == '6d':
                                        all_score_encoded.append(tsv_grasp_info_list[1])
                                        all_width_encoded.append(tsv_grasp_info_list[2])
                                        all_height_encoded.append(tsv_grasp_info_list[3])
                                        all_depth_encoded.append(tsv_grasp_info_list[4])
                                        all_translation_1_encoded.append(tsv_grasp_info_list[5])
                                        all_translation_2_encoded.append(tsv_grasp_info_list[6])
                                        all_translation_3_encoded.append(tsv_grasp_info_list[7])
                                        all_rotation_1_encoded.append(tsv_grasp_info_list[8])
                                        all_rotation_2_encoded.append(tsv_grasp_info_list[9])
                                        all_rotation_3_encoded.append(tsv_grasp_info_list[10])
                                        
                                    elif format == 'rect':
                                        all_score_encoded.append(tsv_grasp_info_list[1])
                                        all_height_encoded.append(tsv_grasp_info_list[2])
                                        all_center_point_1_encoded.append(tsv_grasp_info_list[3])
                                        all_center_point_2_encoded.append(tsv_grasp_info_list[4])
                                        all_open_point_1_encoded.append(tsv_grasp_info_list[5])
                                        all_open_point_2_encoded.append(tsv_grasp_info_list[6])
                                        
                                        
                    # ====== save data_num and data_sum for this ann in this scene ======
                    data_num_and_sum['data_num_'+dir+'_'+sub_dir] = data_num
                    data_num_and_sum['data_sum_'+dir+'_'+sub_dir] = data_sum
                    data_num = 0
                    
    # ====== save all data_num and data_sum ======
    with open(grasp_tsvs_predicted_file_path+'/data_num_and_sum.json','w') as data_num_and_sum_json:
        json.dump(data_num_and_sum, data_num_and_sum_json,indent=4)
    
    print(f'====== There are {data_sum} data totally!!! ======') 
    
    # ====== save grasp predicted data in txt files ======
    if format == '6d':
        save_grasp_txt_predicted(data=all_score_encoded,caption='all_score_encoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_score_encoded.txt')
        save_grasp_txt_predicted(data=all_width_encoded,caption='all_width_encoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_width_encoded.txt')
        save_grasp_txt_predicted(data=all_height_encoded,caption='all_height_encoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_height_encoded.txt')
        save_grasp_txt_predicted(data=all_depth_encoded,caption='all_depth_encoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_depth_encoded.txt')
        save_grasp_txt_predicted(data=all_translation_1_encoded,caption='all_translation_1_encoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_translation_1_encoded.txt')
        save_grasp_txt_predicted(data=all_translation_2_encoded,caption='all_translation_2_encoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_translation_2_encoded.txt')
        save_grasp_txt_predicted(data=all_translation_3_encoded,caption='all_translation_3_encoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_translation_3_encoded.txt')
        save_grasp_txt_predicted(data=all_rotation_1_encoded,caption='all_rotation_1_encoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_rotation_1_encoded.txt')
        save_grasp_txt_predicted(data=all_rotation_2_encoded,caption='all_rotation_2_encoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_rotation_2_encoded.txt')
        save_grasp_txt_predicted(data=all_rotation_3_encoded,caption='all_rotation_3_encoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_rotation_3_encoded.txt')
        
    elif format == 'rect':
        save_grasp_txt_predicted(data=all_score_encoded,caption='all_score_encoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_score_encoded.txt')
        save_grasp_txt_predicted(data=all_height_encoded,caption='all_height_encoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_height_encoded.txt')
        save_grasp_txt_predicted(data=all_center_point_1_encoded,caption='all_center_point_1_encoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_center_point_1_encoded.txt')
        save_grasp_txt_predicted(data=all_center_point_2_encoded,caption='all_center_point_2_encoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_center_point_2_encoded.txt')
        save_grasp_txt_predicted(data=all_open_point_1_encoded,caption='all_open_point_1_encoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_open_point_1_encoded.txt')
        save_grasp_txt_predicted(data=all_open_point_2_encoded,caption='all_open_point_2_encoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_open_point_2_encoded.txt')

def plot_grasp_txt_predicted_encoded(graspnet_root,camera,format,scene_sum):
    '''    
    @Function: plot image of the predicted encoded data distribution from txt files and save
            
    @Input:
        - graspnet_root: ROOT PATH FOR GRASPNET
        - camera: string of type of camera, "realsense" or "kinect".
        - format: string of grasp format, '6d' or 'rect'.
        - scene_sum: int of the number of scenes in the dataset.

    @Output: None
        - all_score_encoded_distribution.png
        - all_width_encoded_distribution.png
        - all_height_encoded_distribution.png
        - all_depth_encoded_distribution.png
        - all_translation_1_encoded_distribution.png
        - all_translation_2_encoded_distribution.png
        - all_translation_3_encoded_distribution.png
        - all_rotation_1_encoded_distribution.png
        - all_rotation_2_encoded_distribution.png
        - all_rotation_3_encoded_distribution.png
    '''

    grasp_tsvs_predicted_file_path=graspnet_root + '/grasp_tsvs_predicted/grasp_tsvs_predicted_'+camera+'_'+format+'_'+str(scene_sum)
       
    def plot(data,caption,save_dir): 
        ''' plot image in speicified save directory'''
        plt.figure(figsize=(10, 5))
        plt.hist(data, bins=256, color='red', alpha=0.7, label='Encoded Data')
        plt.xlabel('Encoded Value')
        plt.ylabel('Frequency')
        plt.title(caption+' Distribution of Encoded Data')
        plt.legend()
        plt.savefig(save_dir+'/'+caption+'_distribution.png')
        print(f'====== {caption} distribution is ploted successfully!!! ======') 
    
    # ====== get all grasp predicted encoded data ======
    if format == '6d':
        all_data = load_grasp_txt_predicted(grasp_tsvs_predicted_file_path,uncoded_or_encoded='encoded',format=format)  
        all_score_encoded,all_width_encoded,all_height_encoded,all_depth_encoded,all_translation_1_encoded,all_translation_2_encoded,all_translation_3_encoded,all_rotation_1_encoded,all_rotation_2_encoded,all_rotation_3_encoded = all_data
    
    elif format == 'rect':
        all_data = load_grasp_txt_predicted(grasp_tsvs_predicted_file_path,uncoded_or_encoded='encoded',format=format)  
        all_score_encoded,all_height_encoded,all_center_point_1_encoded,all_center_point_2_encoded,all_open_point_1_encoded,all_open_point_2_encoded = all_data
    
    # ====== plot and save ======
    if format == '6d':
        plot(data=all_score_encoded,caption='all_score_encoded',save_dir = grasp_tsvs_predicted_file_path)
        plot(data=all_width_encoded,caption='all_width_encoded',save_dir = grasp_tsvs_predicted_file_path)
        plot(data=all_height_encoded,caption='all_height_encoded',save_dir = grasp_tsvs_predicted_file_path)
        plot(data=all_depth_encoded,caption='all_depth_encoded',save_dir = grasp_tsvs_predicted_file_path)
        plot(data=all_translation_1_encoded,caption='all_translation_1_encoded',save_dir = grasp_tsvs_predicted_file_path)
        plot(data=all_translation_2_encoded,caption='all_translation_2_encoded',save_dir = grasp_tsvs_predicted_file_path)
        plot(data=all_translation_3_encoded,caption='all_translation_3_encoded',save_dir = grasp_tsvs_predicted_file_path)
        plot(data=all_rotation_1_encoded,caption='all_rotation_1_encoded',save_dir = grasp_tsvs_predicted_file_path)
        plot(data=all_rotation_2_encoded,caption='all_rotation_2_encoded',save_dir = grasp_tsvs_predicted_file_path)
        plot(data=all_rotation_3_encoded,caption='all_rotation_3_encoded',save_dir = grasp_tsvs_predicted_file_path)
        # plt.show()
    
    elif format == 'rect':
        plot(data=all_score_encoded,caption='all_score_encoded',save_dir = grasp_tsvs_predicted_file_path)
        plot(data=all_height_encoded,caption='all_height_encoded',save_dir = grasp_tsvs_predicted_file_path)
        plot(data=all_center_point_1_encoded,caption='all_center_point_1_encoded',save_dir = grasp_tsvs_predicted_file_path)
        plot(data=all_center_point_2_encoded,caption='all_center_point_2_encoded',save_dir = grasp_tsvs_predicted_file_path)
        plot(data=all_open_point_1_encoded,caption='all_open_point_1_encoded',save_dir = grasp_tsvs_predicted_file_path)
        plot(data=all_open_point_2_encoded,caption='all_open_point_2_encoded',save_dir = grasp_tsvs_predicted_file_path)
        # plt.show()

def generate_grasp_txt_predicted_uncoded(graspnet_root,camera,format,scene_sum):
    '''    
    @Function: save all grasp predicted uncoded data in one txt file
            
    @Input:
        - graspnet_root: ROOT PATH FOR GRASPNET
        - camera: string of type of camera, "realsense" or "kinect".
        - format: string of grasp format, '6d' or 'rect'.
        - scene_sum: int of the number of scenes in the dataset.

    @Output: None
        - all_score_uncoded.txt
        - all_wdith_uncoded.txt
        - all_height_uncoded.txt
        - all_depth_uncoded.txt
        - all_translation_1_uncoded.txt
        - all_translation_2_uncoded.txt
        - all_translation_3_uncoded.txt
        - all_rotation_1_uncoded.txt
        - all_rotation_2_uncoded.txt
        - all_rotation_3_uncoded.txt
    '''

    grasp_tsvs_predicted_file_path=graspnet_root + '/grasp_tsvs_predicted/grasp_tsvs_predicted_'+camera+'_'+format+'_'+str(scene_sum)
    
    def save_grasp_txt_predicted(data,caption,txt_file_path):
        ''' save uncoded data in txt file '''
        with open(txt_file_path, "w") as file:
            for item in data:          
                file.write(str(item)+"\n")
            file.close()
        print(f'====== {caption} txt file is created successfully!!! ======') 
    
    # ====== get all grasp predicted encoded data ======
    if format == '6d':
        all_data = load_grasp_txt_predicted(grasp_tsvs_predicted_file_path,uncoded_or_encoded='encoded',format=format)  
        all_score_encoded,all_width_encoded,all_height_encoded,all_depth_encoded,all_translation_1_encoded,all_translation_2_encoded,all_translation_3_encoded,all_rotation_1_encoded,all_rotation_2_encoded,all_rotation_3_encoded = all_data
    
    elif format == 'rect':
        all_data = load_grasp_txt_predicted(grasp_tsvs_predicted_file_path,uncoded_or_encoded='encoded',format=format)  
        all_score_encoded,all_height_encoded,all_center_point_1_encoded,all_center_point_2_encoded,all_open_point_1_encoded,all_open_point_2_encoded = all_data
    
    # ====== get all grasp real uncoded data ======
    grasp_tsvs_real_file_path=graspnet_root + '/grasp_tsvs_real/grasp_tsvs_real_'+camera+'_'+format+'_'+str(scene_sum)
    if format == '6d':
        all_data_real = load_grasp_txt_real(grasp_tsvs_real_file_path,uncoded_or_encoded='uncoded',format=format)  
        all_score_uncoded_real,all_width_uncoded_real,all_height_uncoded_real,all_depth_uncoded_real,all_translation_1_uncoded_real,all_translation_2_uncoded_real,all_translation_3_uncoded_real,all_rotation_1_uncoded_real,all_rotation_2_uncoded_real,all_rotation_3_uncoded_real = all_data_real
    
    elif format == 'rect':
        all_data_real = load_grasp_txt_real(grasp_tsvs_real_file_path,uncoded_or_encoded='uncoded',format=format)  
        all_score_uncoded_real,all_height_uncoded_real,all_center_point_1_uncoded_real,all_center_point_2_uncoded_real,all_open_point_1_uncoded_real,all_open_point_2_uncoded_real = all_data_real
        
    # ====== decode all grasp predicted data ======
    if format == '6d':
        all_score_uncoded = KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit(all_score_uncoded_real.reshape(-1, 1)).inverse_transform(np.array(all_score_encoded).reshape(-1, 1)).flatten().tolist()
        all_width_uncoded = KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit(all_width_uncoded_real.reshape(-1, 1)).inverse_transform(np.array(all_width_encoded).reshape(-1, 1)).flatten().tolist()
        all_height_uncoded = KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit(all_height_uncoded_real.reshape(-1, 1)).inverse_transform(np.array(all_height_encoded).reshape(-1, 1)).flatten().tolist()
        all_depth_uncoded = KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit(all_depth_uncoded_real.reshape(-1, 1)).inverse_transform(np.array(all_depth_encoded).reshape(-1, 1)).flatten().tolist()
        all_translation_1_uncoded = KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit(all_translation_1_uncoded_real.reshape(-1, 1)).inverse_transform(np.array(all_translation_1_encoded).reshape(-1, 1)).flatten().tolist()
        all_translation_2_uncoded = KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit(all_translation_2_uncoded_real.reshape(-1, 1)).inverse_transform(np.array(all_translation_2_encoded).reshape(-1, 1)).flatten().tolist()
        all_translation_3_uncoded = KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit(all_translation_3_uncoded_real.reshape(-1, 1)).inverse_transform(np.array(all_translation_3_encoded).reshape(-1, 1)).flatten().tolist()
        all_rotation_1_uncoded = KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit(all_rotation_1_uncoded_real.reshape(-1, 1)).inverse_transform(np.array(all_rotation_1_encoded).reshape(-1, 1)).flatten().tolist()
        all_rotation_2_uncoded = KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit(all_rotation_2_uncoded_real.reshape(-1, 1)).inverse_transform(np.array(all_rotation_2_encoded).reshape(-1, 1)).flatten().tolist()
        all_rotation_3_uncoded = KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit(all_rotation_3_uncoded_real.reshape(-1, 1)).inverse_transform(np.array(all_rotation_3_encoded).reshape(-1, 1)).flatten().tolist()

        # ====== special data ======
        data_num = len(all_score_uncoded)
        all_score_uncoded = [1.0]*data_num
        all_height_uncoded = [0.019999999552965164] * data_num
        all_depth_uncoded = [0.009999999776482582 if 25 < num <= 75 else 0.019999999552965164 if 75 < num <= 125 else 0.029999999329447746 if 125 < num <= 175 else 0.03999999910593033 if 175 < num <= 225 else None for num in all_depth_encoded]

    elif format == 'rect':
        all_score_uncoded = KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit(all_score_uncoded_real.reshape(-1, 1)).inverse_transform(np.array(all_score_encoded).reshape(-1, 1)).flatten().tolist()
        all_height_uncoded = KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit(all_height_uncoded_real.reshape(-1, 1)).inverse_transform(np.array(all_height_encoded).reshape(-1, 1)).flatten().tolist()
        all_center_point_1_uncoded = KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit(all_center_point_1_uncoded_real.reshape(-1, 1)).inverse_transform(np.array(all_center_point_1_encoded).reshape(-1, 1)).flatten().tolist()
        all_center_point_2_uncoded = KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit(all_center_point_2_uncoded_real.reshape(-1, 1)).inverse_transform(np.array(all_center_point_2_encoded).reshape(-1, 1)).flatten().tolist()
        all_open_point_1_uncoded = KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit(all_open_point_1_uncoded_real.reshape(-1, 1)).inverse_transform(np.array(all_open_point_1_encoded).reshape(-1, 1)).flatten().tolist()
        all_open_point_2_uncoded = KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit(all_open_point_2_uncoded_real.reshape(-1, 1)).inverse_transform(np.array(all_open_point_2_encoded).reshape(-1, 1)).flatten().tolist()
    
    # ====== save grasp predicted uncoded data in txt files ======
    if format == '6d':
        save_grasp_txt_predicted(data=all_score_uncoded,caption='all_score_uncoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_score_uncoded.txt')
        save_grasp_txt_predicted(data=all_width_uncoded,caption='all_width_uncoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_width_uncoded.txt')
        save_grasp_txt_predicted(data=all_height_uncoded,caption='all_height_uncoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_height_uncoded.txt')
        save_grasp_txt_predicted(data=all_depth_uncoded,caption='all_depth_uncoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_depth_uncoded.txt')
        save_grasp_txt_predicted(data=all_translation_1_uncoded,caption='all_translation_1_uncoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_translation_1_uncoded.txt')
        save_grasp_txt_predicted(data=all_translation_2_uncoded,caption='all_translation_2_uncoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_translation_2_uncoded.txt')
        save_grasp_txt_predicted(data=all_translation_3_uncoded,caption='all_translation_3_uncoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_translation_3_uncoded.txt')
        save_grasp_txt_predicted(data=all_rotation_1_uncoded,caption='all_rotation_1_uncoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_rotation_1_uncoded.txt')
        save_grasp_txt_predicted(data=all_rotation_2_uncoded,caption='all_rotation_2_uncoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_rotation_2_uncoded.txt')
        save_grasp_txt_predicted(data=all_rotation_3_uncoded,caption='all_rotation_3_uncoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_rotation_3_uncoded.txt')
        
    elif format == 'rect':
        save_grasp_txt_predicted(data=all_score_uncoded,caption='all_score_uncoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_score_uncoded.txt')
        save_grasp_txt_predicted(data=all_height_uncoded,caption='all_height_uncoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_height_uncoded.txt')
        save_grasp_txt_predicted(data=all_center_point_1_uncoded,caption='all_center_point_1_uncoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_center_point_1_uncoded.txt')
        save_grasp_txt_predicted(data=all_center_point_2_uncoded,caption='all_center_point_2_uncoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_center_point_2_uncoded.txt')
        save_grasp_txt_predicted(data=all_open_point_1_uncoded,caption='all_open_point_1_uncoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_open_point_1_uncoded.txt')
        save_grasp_txt_predicted(data=all_open_point_2_uncoded,caption='all_open_point_2_uncoded',txt_file_path = grasp_tsvs_predicted_file_path+'/all_open_point_2_uncoded.txt')
    
def plot_grasp_txt_predicted_uncoded(graspnet_root,camera,format,scene_sum):
    '''    
    @Function: plot image of the predicted_uncoded data distribution from txt files and save
            
    @Input:
        - graspnet_root: ROOT PATH FOR GRASPNET
        - camera: string of type of camera, "realsense" or "kinect".
        - format: string of grasp format, '6d' or 'rect'.
        - scene_sum: int of the number of scenes in the dataset.

    @Output: None
        - all_score_uncoded_distribution.png
        - all_width_uncoded_distribution.png
        - all_height_uncoded_distribution.png
        - all_depth_uncoded_distribution.png
        - all_translation_1_uncoded_distribution.png
        - all_translation_2_uncoded_distribution.png
        - all_translation_3_uncoded_distribution.png
        - all_rotation_1_uncoded_distribution.png
        - all_rotation_2_uncoded_distribution.png
        - all_rotation_3_uncoded_distribution.png
    '''
        
    grasp_tsvs_predicted_file_path=graspnet_root + '/grasp_tsvs_predicted/grasp_tsvs_predicted_'+camera+'_'+format+'_'+str(scene_sum)
    
    def plot(data,caption,save_dir): 
        ''' plot image in speicified save directory'''
        plt.figure(figsize=(10, 5))
        plt.hist(data, bins=100, color='blue', alpha=0.7, label='Uncoded Data')
        plt.xlabel('Uncoded Value')
        plt.ylabel('Frequency')
        plt.title(caption+' Distribution of Uncoded Data')
        plt.legend()
        plt.savefig(save_dir+'/'+caption+'_distribution.png')
        print(f'====== {caption} distribution is ploted successfully!!! ======') 
        
    # ====== get all grasp predicted data ======
    if format == '6d':
        all_data = load_grasp_txt_predicted(grasp_tsvs_predicted_file_path,uncoded_or_encoded='uncoded',format=format)  
        all_score_uncoded,all_width_uncoded,all_height_uncoded,all_depth_uncoded,all_translation_1_uncoded,all_translation_2_uncoded,all_translation_3_uncoded,all_rotation_1_uncoded,all_rotation_2_uncoded,all_rotation_3_uncoded = all_data
    
    elif format == 'rect':
        all_data = load_grasp_txt_predicted(grasp_tsvs_predicted_file_path,uncoded_or_encoded='uncoded',format=format)  
        all_score_uncoded,all_height_uncoded,all_center_point_1_uncoded,all_center_point_2_uncoded,all_open_point_1_uncoded,all_open_point_2_uncoded = all_data
        
    # ====== plot and save ======
    if format == '6d':
        plot(data=all_score_uncoded,caption='all_score_uncoded',save_dir = grasp_tsvs_predicted_file_path)
        plot(data=all_width_uncoded,caption='all_width_uncoded',save_dir = grasp_tsvs_predicted_file_path)
        plot(data=all_height_uncoded,caption='all_height_uncoded',save_dir = grasp_tsvs_predicted_file_path)
        plot(data=all_depth_uncoded,caption='all_depth_uncoded',save_dir = grasp_tsvs_predicted_file_path)
        plot(data=all_translation_1_uncoded,caption='all_translation_1_uncoded',save_dir = grasp_tsvs_predicted_file_path)
        plot(data=all_translation_2_uncoded,caption='all_translation_2_uncoded',save_dir = grasp_tsvs_predicted_file_path)
        plot(data=all_translation_3_uncoded,caption='all_translation_3_uncoded',save_dir = grasp_tsvs_predicted_file_path)
        plot(data=all_rotation_1_uncoded,caption='all_rotation_1_uncoded',save_dir = grasp_tsvs_predicted_file_path)
        plot(data=all_rotation_2_uncoded,caption='all_rotation_2_uncoded',save_dir = grasp_tsvs_predicted_file_path)
        plot(data=all_rotation_3_uncoded,caption='all_rotation_3_uncoded',save_dir = grasp_tsvs_predicted_file_path)
        # plt.show()
    
    elif format == 'rect':
        plot(data=all_score_uncoded,caption='all_score_uncoded',save_dir = grasp_tsvs_predicted_file_path)
        plot(data=all_height_uncoded,caption='all_height_uncoded',save_dir = grasp_tsvs_predicted_file_path)
        plot(data=all_center_point_1_uncoded,caption='all_center_point_1_uncoded',save_dir = grasp_tsvs_predicted_file_path)
        plot(data=all_center_point_2_uncoded,caption='all_center_point_2_uncoded',save_dir = grasp_tsvs_predicted_file_path)
        plot(data=all_open_point_1_uncoded,caption='all_open_point_1_uncoded',save_dir = grasp_tsvs_predicted_file_path)
        plot(data=all_open_point_2_uncoded,caption='all_open_point_2_uncoded',save_dir = grasp_tsvs_predicted_file_path)
        # plt.show()

def generate_grasp_tsvs_predicted(graspnet_root,train_or_test,camera,format,scene_sum,sceneId_start,sceneId_end,annId_start,annId_end,if_generate_grasp_tsvs_predicted_encoded=False,if_generate_grasp_txt_predicted_encoded=False,if_plot_grasp_txt_predicted_encoded=False,if_generate_grasp_txt_predicted_uncoded=False,if_plot_grasp_txt_predicted_uncoded=False):
    '''    
    @Function: 
        - 1.get all predicted encoded grasp info data from json files and save them in tsv files.
        - 2.save all grasp predicted_encoded data from tsv files in one txt file.
        - 3.plot image of the predicted_encoded data distribution from txt files and save
        - 4.save all grasp predicted uncoded data in one txt file
        - 5.plot image of the predicted uncoded data distribution from txt files and save
        
            
    @Input:
        - graspnet_root: ROOT PATH FOR GRASPNET.
        - train_or_test: string, a flag of "train" or "test".
        - camera: string of type of camera, "realsense" or "kinect".
        - format: string of grasp format, '6d' or 'rect'.
        - scene_sum: int of the number of scenes in the dataset.
        - sceneId_start: int of the starting scene index.(sceneId max range is [0:180])
        - sceneId_end: int of the ending scene index.
        - annId_start: int of the starting annotation index.(annId max range is [0:256])
        - annId_end: int of the ending annotation index.

    @Output: None
        - grasp_tsvs_real 
            - grasp_tsvs_real_kinect_6d_100
                - 0000
                - 0001
                    - 0000
                    - 0001
                        - grasp_05_00.tsv
                        - grasp_05_01.tsv

                        ....
                        - grasp_05_09.tsv
                    ....
                    - 0256

                ....
                - 0099

                - all_score_encoded.txt
                - all_width_encoded.txt
                - all_height_encoded.txt
                - all_depth_encoded.txt
                - all_translation_1_encoded.txt
                - all_translation_2_encoded.txt
                - all_translation_3_encoded.txt
                - all_rotation_1_encoded.txt
                - all_rotation_2_encoded.txt
                - all_rotation_3_encoded.txt

                - all_score_uncoded.txt
                - all_wdith_uncoded.txt
                - all_height_uncoded.txt
                - all_depth_uncoded.txt
                - all_translation_1_uncoded.txt
                - all_translation_2_uncoded.txt
                - all_translation_3_uncoded.txt
                - all_rotation_1_uncoded.txt
                - all_rotation_2_uncoded.txt
                - all_rotation_3_uncoded.txt
                
                - data_num_and_sum.json

                - all_score_encoded_distribution.png
                - all_width_encoded_distribution.png
                - all_height_encoded_distribution.png
                - all_depth_encoded_distribution.png
                - all_translation_1_encoded_distribution.png
                - all_translation_2_encoded_distribution.png
                - all_translation_3_encoded_distribution.png
                - all_rotation_1_encoded_distribution.png
                - all_rotation_2_encoded_distribution.png
                - all_rotation_3_encoded_distribution.png
                
                - all_score_uncoded_distribution.png
                - all_width_uncoded_distribution.png
                - all_height_uncoded_distribution.png
                - all_depth_uncoded_distribution.png
                - all_translation_1_uncoded_distribution.png
                - all_translation_2_uncoded_distribution.png
                - all_translation_3_uncoded_distribution.png
                - all_rotation_1_uncoded_distribution.png
                - all_rotation_2_uncoded_distribution.png
                - all_rotation_3_uncoded_distribution.png
    '''
    if if_generate_grasp_tsvs_predicted_encoded:
        generate_grasp_tsvs_predicted_encoded(graspnet_root,train_or_test,camera,format,scene_sum,sceneId_start,sceneId_end,annId_start,annId_end)
    
    if if_generate_grasp_txt_predicted_encoded:
        generate_grasp_txt_predicted_encoded(graspnet_root,camera,format,scene_sum)
    
    if if_plot_grasp_txt_predicted_encoded:
        plot_grasp_txt_predicted_encoded(graspnet_root,camera,format,scene_sum)
    
    if if_generate_grasp_txt_predicted_uncoded:
        generate_grasp_txt_predicted_uncoded(graspnet_root,camera,format,scene_sum)
    
    if if_plot_grasp_txt_predicted_uncoded:
        plot_grasp_txt_predicted_uncoded(graspnet_root,camera,format,scene_sum)

def main():
    graspnet_root = '/mnt/msranlpintern/dataset/graspnet-v2' #'D:/dataset/graspnet' # 'D:/dataset/graspnet' # /Volumes/WD/dataset/graspnet
    train_or_test = 'train'
    camera = 'kinect'
    format = '6d'
    scene_sum = 100
    sceneId_start = 0
    sceneId_end = 1
    annId_start = 0
    annId_end = 1
    if_generate_grasp_tsvs_predicted_encoded=False
    if_generate_grasp_txt_predicted_encoded=False
    if_plot_grasp_txt_predicted_encoded=False
    if_generate_grasp_txt_predicted_uncoded=False
    if_plot_grasp_txt_predicted_uncoded=False
    
    generate_grasp_tsvs_predicted(graspnet_root=graspnet_root,train_or_test=train_or_test,camera=camera,format=format,scene_sum=scene_sum,sceneId_start=sceneId_start,sceneId_end=sceneId_end,annId_start=annId_start,annId_end=annId_end,if_generate_grasp_tsvs_predicted_encoded=if_generate_grasp_tsvs_predicted_encoded,if_generate_grasp_txt_predicted_encoded=if_generate_grasp_txt_predicted_encoded,if_plot_grasp_txt_predicted_encoded=if_plot_grasp_txt_predicted_encoded,if_generate_grasp_txt_predicted_uncoded=if_generate_grasp_txt_predicted_uncoded,if_plot_grasp_txt_predicted_uncoded=if_plot_grasp_txt_predicted_uncoded)


if __name__ == '__main__':
    main()
