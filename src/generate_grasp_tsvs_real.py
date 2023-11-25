__author__ = 'Zhi Wang'
__date__ = '2023.11.10'

import os
import json
import csv
import re
import transforms3d as tfs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

def generate_grasp_tsvs_real_uncoded(graspnet_root,train_or_test,camera,format,scene_sum,sceneId_start,sceneId_end,annId_start,annId_end):
    '''    
    @Function: get all real uncoded grasp info data from json files and save them in tsv files.
            
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
    '''
    
    # ====== for every scene ======
    for sceneId in list(range(sceneId_start, sceneId_end)):
        # ====== for every ann ======
        for annId in list(range(annId_start, annId_end)):
            grasp_tsvs_real_for_one_ann_in_one_scene_path = graspnet_root + '/grasp_tsvs_real/grasp_tsvs_real_'+camera+'_'+format+'_'+str(scene_sum)+'/{:04d}'.format(sceneId) +'/{:04d}'.format(annId) 
            if not os.path.exists(grasp_tsvs_real_for_one_ann_in_one_scene_path):
                os.makedirs(grasp_tsvs_real_for_one_ann_in_one_scene_path)   
            grasp_jsons_for_one_ann_in_one_scene_path = graspnet_root + '/grasp_jsons/' + camera + '/' + format + '/{:04d}/'.format(sceneId)+ '/{:04d}'.format(annId)  
            
            # ====== loop through all grasp json files for this ann in this scene ====== 
            for root,dirs,files in os.walk(grasp_jsons_for_one_ann_in_one_scene_path):
                files.sort()
                json_files_num = 0
                for json_file in files:
                    json_files_num += 1
                    json_file_path = os.path.join(root,json_file)
                    
                    # ====== get FLAG ======
                    FLAG = '0'
                    FLAG += '00' if train_or_test == 'train' else ('01' if train_or_test == 'test' else None)
                    FLAG += '00' if camera == 'kinect' else ('01' if camera == 'realsense' else None)
                    FLAG += '00' if format == '6d' else ('01' if format == 'rect' else None)
                    FLAG += '{:04d}'.format(sceneId)
                    FLAG += '{:04d}'.format(annId)
                    FLAG += re.search(r"grasp_(\d+)_(\d+)\.json", json_file).group(1) 
                    FLAG += re.search(r"grasp_(\d+)_(\d+)\.json", json_file).group(2)
                    
                    # ====== get grasp info from json files ======
                    grasp_json_file_path = json_file_path
                    with open(grasp_json_file_path,'r') as grasp_json:
                        data = json.load(grasp_json)
                        rbg_img_path = data["rbg_img_path"]
                        depth_img_path = data["depth_img_path"]
                        all_obj_names_and_ids_dict = data["all_obj_names_and_ids_dict"]
                        object_id = data["object_id"]
                        object_name = data["object_name"]
                        score = data["score"]
                        
                        if format == '6d':
                            width = data["width"]
                            height = data["height"]
                            depth = data["depth"]
                            translation = data["translation"]
                            rotation_matrix = data["rotation_matrix"]
                            rotation = list(tfs.euler.mat2euler(rotation_matrix, 'sxyz')) # rotation: 3*3 --> 3*1
                            translation_1 = translation[0]
                            translation_2 = translation[1]
                            translation_3 = translation[2]
                            rotation_1 = rotation[0]
                            rotation_2 = rotation[1]
                            rotation_3 = rotation[2]
                            # print(f'====== FLAG:====== \n{FLAG}')
                            # print(f'====== score:====== \n{score}') 
                            # print(f'====== width:====== \n{width}') 
                            # print(f'====== height:====== \n{height}') 
                            # print(f'====== depth:====== \n{depth}') 
                            # print(f'====== translation:====== \n{translation}') 
                            # print(f'====== rotation:====== \n{rotation}') 
                            
                        elif format == 'rect':
                            height = data["height"]
                            center_point = data["center_point"]
                            open_point = data["open_point"]
                            center_point_1 = center_point[0]
                            center_point_2 = center_point[1]
                            open_point_1 = open_point[0]
                            open_point_2 = open_point[1]
                            # print(f'====== FLAG:====== \n{FLAG}')
                            # print(f'====== score:====== \n{score}') 
                            # print(f'====== height:====== \n{height}') 
                            # print(f'====== center_point:====== \n{center_point}') 
                            # print(f'====== open_point:====== \n{open_point}') 
                    
                    # ====== save real grasp info to tsvs file ======
                    tsv_file_path = grasp_tsvs_real_for_one_ann_in_one_scene_path + '/' + json_file[:-5]+'.tsv'
                    with open(tsv_file_path,'w',newline='') as tsv_file:
                        if format == '6d':
                            data = [FLAG,score,width,height,depth,translation_1,translation_2,translation_3,rotation_1,rotation_2,rotation_3]
                        elif format == 'rect':
                            data = [FLAG,score,height,center_point_1,center_point_2,open_point_1,open_point_2]
                        writer = csv.writer(tsv_file, delimiter='\t')
                        writer.writerow(data)  
                    # print(f'====== [scene_{sceneId},ann_{annId},num_{json_files_num}] grasp tsvs are generated successfully!!! ======')
            print(f'====== [scene_{sceneId},ann_{annId}] grasp tsvs real are generated successfully!!! ======')

def load_grasp_txt_real(grasp_tsvs_real_file_path,uncoded_or_encoded,format):  
    '''    
    @Function: load all grasp real_uncoded or real_encoded data from txt file
            
    @Input:
        - grasp_tsvs_real_file_path: the path of the floder: "the grasp_tsvs_real"
        - uncoded_or_encoded: string, load grasp real_uncoded data if set to "uncoded", load grasp real_encoded data if set to "encoded"
        - format: string of grasp format, '6d' or 'rect'.

    @Output: 
        - all data: a list of all grasp data
            format == '6d': score,width,height,depth,translation_1,translation_2,translation_3,rotation_1,rotation_2,rotation_3
            format == 'rect: score,height,center_point_1,center_point_2,open_point_1,open_point_2
    '''

    def open_grasp_txt_real(caption,txt_file_path,uncoded_or_encoded):
        ''' get data from the txt file'''
        with open(txt_file_path, "r") as file:
            lines = file.readlines()
            data = np.array([float(line.strip()) for line in lines])
            file.close()
        print(f'====== {caption} {uncoded_or_encoded} txt file is opened successfully!!! ======')
        return data 
    
    if format == '6d':
        all_score = open_grasp_txt_real(caption='[Score]',txt_file_path = grasp_tsvs_real_file_path+'/all_score_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        all_width = open_grasp_txt_real(caption='[Width]',txt_file_path = grasp_tsvs_real_file_path+'/all_width_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        all_height = open_grasp_txt_real(caption='[Height]',txt_file_path = grasp_tsvs_real_file_path+'/all_height_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        all_depth = open_grasp_txt_real(caption='[Depth]',txt_file_path = grasp_tsvs_real_file_path+'/all_depth_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        all_translation_1 = open_grasp_txt_real(caption='[Translation-1]',txt_file_path = grasp_tsvs_real_file_path+'/all_translation_1_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        all_translation_2 = open_grasp_txt_real(caption='[Translation-2]',txt_file_path = grasp_tsvs_real_file_path+'/all_translation_2_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        all_translation_3 = open_grasp_txt_real(caption='[Translation-3]',txt_file_path = grasp_tsvs_real_file_path+'/all_translation_3_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        all_rotation_1 = open_grasp_txt_real(caption='[Rotation-1]',txt_file_path = grasp_tsvs_real_file_path+'/all_rotation_1_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        all_rotation_2 = open_grasp_txt_real(caption='[Rotation-2]',txt_file_path = grasp_tsvs_real_file_path+'/all_rotation_2_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        all_rotation_3 = open_grasp_txt_real(caption='[Rotation-3]',txt_file_path = grasp_tsvs_real_file_path+'/all_rotation_3_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        print(f'====== There are {len(all_width)} data totally!!! ======') 

        all_data = [all_score,all_width,all_height,all_depth,all_translation_1,all_translation_2,all_translation_3,all_rotation_1,all_rotation_2,all_rotation_3]

    elif format == 'rect':
        all_score = open_grasp_txt_real(caption='[Score]',txt_file_path = grasp_tsvs_real_file_path+'/all_score_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        all_height = open_grasp_txt_real(caption='[Height]',txt_file_path = grasp_tsvs_real_file_path+'/all_height_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        all_center_point_1 = open_grasp_txt_real(caption='[Center-Point-1]',txt_file_path = grasp_tsvs_real_file_path+'/all_center_point_1_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        all_center_point_2 = open_grasp_txt_real(caption='[Center-Point-2]',txt_file_path = grasp_tsvs_real_file_path+'/all_center_point_2_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        all_open_point_1 = open_grasp_txt_real(caption='[Open-Point-1]',txt_file_path = grasp_tsvs_real_file_path+'/all_open_point_1_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        all_open_point_2 = open_grasp_txt_real(caption='[Open-Point-2]',txt_file_path = grasp_tsvs_real_file_path+'/all_open_point_2_'+uncoded_or_encoded+'.txt',uncoded_or_encoded=uncoded_or_encoded)
        print(f'====== There are {len(all_height)} data totally!!! ======') 

        all_data = [all_score,all_height,all_center_point_1,all_center_point_2,all_open_point_1,all_open_point_2]
    
    return all_data

def generate_grasp_txt_real_uncoded(graspnet_root,camera,format,scene_sum):
    '''    
    @Function: save all grasp real_uncoded data from tsv files in one txt file
            
    @Input:
        - graspnet_root: ROOT PATH FOR GRASPNET
        - camera: string of type of camera, "realsense" or "kinect".
        - format: string of grasp format, '6d' or 'rect'.
        - scene_sum: int of the number of scenes in the dataset.

    @Output: None
        - all_score_uncoded.txt
        - all_width_uncoded.txt
        - all_height_uncoded.txt
        - all_depth_uncoded.txt
        - all_translation_1_uncoded.txt
        - all_translation_2_uncoded.txt
        - all_translation_3_uncoded.txt
        - all_rotation_1_uncoded.txt
        - all_rotation_2_uncoded.txt
        - all_rotation_3_uncoded.txt
    '''

    grasp_tsvs_real_file_path=graspnet_root + '/grasp_tsvs_real/grasp_tsvs_real_'+camera+'_'+format+'_'+str(scene_sum)
    
    def save_grasp_txt_real(data,caption,txt_file_path):
        ''' save data in txt file '''
        with open(txt_file_path, "w") as file:
            for item in data:          
                file.write(str(item)+"\n")
            file.close()
        print(f'====== {caption} txt file is created successfully!!! ======') 
        
    # ====== all grasp data ======
    if format == '6d':
        all_score_uncoded = []
        all_width_uncoded = []
        all_height_uncoded = []
        all_depth_uncoded = []
        all_translation_1_uncoded = []
        all_translation_2_uncoded = []
        all_translation_3_uncoded = []
        all_rotation_1_uncoded = []
        all_rotation_2_uncoded = []
        all_rotation_3_uncoded = []
    
    elif format == 'rect':
        all_score_uncoded = []
        all_height_uncoded = []
        all_center_point_1_uncoded = []
        all_center_point_2_uncoded = []
        all_open_point_1_uncoded = []
        all_open_point_2_uncoded = []

    data_num = 0 
    data_sum = 0
    data_num_and_sum = {}
    # ====== get grasp real data from tsv files ======
    for root,dirs,files in os.walk(grasp_tsvs_real_file_path): # grasp_tsvs_real_file_path/
        dirs.sort()
        for dir in dirs:
            dir_path = os.path.join(root,dir) # grasp_tsvs_real_file_path/0000/
            print(f'====== {dir_path} is looping through!!! ======') 
            for sub_root,sub_dirs,sub_files in os.walk(dir_path):
                sub_dirs.sort()
                for sub_dir in sub_dirs:
                    sub_dir_path = os.path.join(sub_root,sub_dir) # grasp_tsvs_real_file_path/0000/0000
                    # print(f'====== {sub_dir_path} is looping through!!! ======') 
                    for sub_sub_root,sub_sub_dirs,sub_sub_files in os.walk(sub_dir_path):           
                        sub_sub_files.sort()
                        for file in sub_sub_files:
                            tsv_file_path = os.path.join(sub_sub_root,file) # grasp_tsvs_real_file_path/0000/0000/grasp_05_00.tsv
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
                                        all_score_uncoded.append(tsv_grasp_info_list[1])
                                        all_width_uncoded.append(tsv_grasp_info_list[2])
                                        all_height_uncoded.append(tsv_grasp_info_list[3])
                                        all_depth_uncoded.append(tsv_grasp_info_list[4])
                                        all_translation_1_uncoded.append(tsv_grasp_info_list[5])
                                        all_translation_2_uncoded.append(tsv_grasp_info_list[6])
                                        all_translation_3_uncoded.append(tsv_grasp_info_list[7])
                                        all_rotation_1_uncoded.append(tsv_grasp_info_list[8])
                                        all_rotation_2_uncoded.append(tsv_grasp_info_list[9])
                                        all_rotation_3_uncoded.append(tsv_grasp_info_list[10])
                                        
                                    elif format == 'rect':
                                        all_score_uncoded.append(tsv_grasp_info_list[1])
                                        all_height_uncoded.append(tsv_grasp_info_list[2])
                                        all_center_point_1_uncoded.append(tsv_grasp_info_list[3])
                                        all_center_point_2_uncoded.append(tsv_grasp_info_list[4])
                                        all_open_point_1_uncoded.append(tsv_grasp_info_list[5])
                                        all_open_point_2_uncoded.append(tsv_grasp_info_list[6])
                                        
                    # ====== save data_num and data_sum for this ann in this scene ======
                    data_num_and_sum['data_num_'+dir+'_'+sub_dir] = data_num
                    data_num_and_sum['data_sum_'+dir+'_'+sub_dir] = data_sum
                    data_num = 0
                    
    # ====== save all data_num and data_sum ======
    with open(grasp_tsvs_real_file_path+'/data_num_and_sum.json','w') as data_num_and_sum_json:
        json.dump(data_num_and_sum, data_num_and_sum_json,indent=4)
    
    print(f'====== There are {data_sum} data totally!!! ======') 
    
    # ====== save grasp real data in txt files ======
    if format == '6d':
        save_grasp_txt_real(data=all_score_uncoded,caption='all_score_uncoded',txt_file_path = grasp_tsvs_real_file_path+'/all_score_uncoded.txt')
        save_grasp_txt_real(data=all_width_uncoded,caption='all_width_uncoded',txt_file_path = grasp_tsvs_real_file_path+'/all_width_uncoded.txt')
        save_grasp_txt_real(data=all_height_uncoded,caption='all_height_uncoded',txt_file_path = grasp_tsvs_real_file_path+'/all_height_uncoded.txt')
        save_grasp_txt_real(data=all_depth_uncoded,caption='all_depth_uncoded',txt_file_path = grasp_tsvs_real_file_path+'/all_depth_uncoded.txt')
        save_grasp_txt_real(data=all_translation_1_uncoded,caption='all_translation_1_uncoded',txt_file_path = grasp_tsvs_real_file_path+'/all_translation_1_uncoded.txt')
        save_grasp_txt_real(data=all_translation_2_uncoded,caption='all_translation_2_uncoded',txt_file_path = grasp_tsvs_real_file_path+'/all_translation_2_uncoded.txt')
        save_grasp_txt_real(data=all_translation_3_uncoded,caption='all_translation_3_uncoded',txt_file_path = grasp_tsvs_real_file_path+'/all_translation_3_uncoded.txt')
        save_grasp_txt_real(data=all_rotation_1_uncoded,caption='all_rotation_1_uncoded',txt_file_path = grasp_tsvs_real_file_path+'/all_rotation_1_uncoded.txt')
        save_grasp_txt_real(data=all_rotation_2_uncoded,caption='all_rotation_2_uncoded',txt_file_path = grasp_tsvs_real_file_path+'/all_rotation_2_uncoded.txt')
        save_grasp_txt_real(data=all_rotation_3_uncoded,caption='all_rotation_3_uncoded',txt_file_path = grasp_tsvs_real_file_path+'/all_rotation_3_uncoded.txt')
        
    elif format == 'rect':
        save_grasp_txt_real(data=all_score_uncoded,caption='all_score_uncoded',txt_file_path = grasp_tsvs_real_file_path+'/all_score_uncoded.txt')
        save_grasp_txt_real(data=all_height_uncoded,caption='all_height_uncoded',txt_file_path = grasp_tsvs_real_file_path+'/all_height_uncoded.txt')
        save_grasp_txt_real(data=all_center_point_1_uncoded,caption='all_center_point_1_uncoded',txt_file_path = grasp_tsvs_real_file_path+'/all_center_point_1_uncoded.txt')
        save_grasp_txt_real(data=all_center_point_2_uncoded,caption='all_center_point_2_uncoded',txt_file_path = grasp_tsvs_real_file_path+'/all_center_point_2_uncoded.txt')
        save_grasp_txt_real(data=all_open_point_1_uncoded,caption='all_open_point_1_uncoded',txt_file_path = grasp_tsvs_real_file_path+'/all_open_point_1_uncoded.txt')
        save_grasp_txt_real(data=all_open_point_2_uncoded,caption='all_open_point_2_uncoded',txt_file_path = grasp_tsvs_real_file_path+'/all_open_point_2_uncoded.txt')
    
def plot_grasp_txt_real_uncoded(graspnet_root,camera,format,scene_sum):
    '''    
    @Function: plot image of the real_uncoded data distribution from txt files and save
            
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
        
    grasp_tsvs_real_file_path=graspnet_root + '/grasp_tsvs_real/grasp_tsvs_real_'+camera+'_'+format+'_'+str(scene_sum)
    
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
        
    # ====== get all grasp real data ======
    if format == '6d':
        all_data = load_grasp_txt_real(grasp_tsvs_real_file_path,uncoded_or_encoded='uncoded',format=format)  
        all_score_uncoded,all_width_uncoded,all_height_uncoded,all_depth_uncoded,all_translation_1_uncoded,all_translation_2_uncoded,all_translation_3_uncoded,all_rotation_1_uncoded,all_rotation_2_uncoded,all_rotation_3_uncoded = all_data
    
    elif format == 'rect':
        all_data = load_grasp_txt_real(grasp_tsvs_real_file_path,uncoded_or_encoded='uncoded',format=format)  
        all_score_uncoded,all_height_uncoded,all_center_point_1_uncoded,all_center_point_2_uncoded,all_open_point_1_uncoded,all_open_point_2_uncoded = all_data
        
    # ====== plot and save ======
    if format == '6d':
        plot(data=all_score_uncoded,caption='all_score_uncoded',save_dir = grasp_tsvs_real_file_path)
        plot(data=all_width_uncoded,caption='all_width_uncoded',save_dir = grasp_tsvs_real_file_path)
        plot(data=all_height_uncoded,caption='all_height_uncoded',save_dir = grasp_tsvs_real_file_path)
        plot(data=all_depth_uncoded,caption='all_depth_uncoded',save_dir = grasp_tsvs_real_file_path)
        plot(data=all_translation_1_uncoded,caption='all_translation_1_uncoded',save_dir = grasp_tsvs_real_file_path)
        plot(data=all_translation_2_uncoded,caption='all_translation_2_uncoded',save_dir = grasp_tsvs_real_file_path)
        plot(data=all_translation_3_uncoded,caption='all_translation_3_uncoded',save_dir = grasp_tsvs_real_file_path)
        plot(data=all_rotation_1_uncoded,caption='all_rotation_1_uncoded',save_dir = grasp_tsvs_real_file_path)
        plot(data=all_rotation_2_uncoded,caption='all_rotation_2_uncoded',save_dir = grasp_tsvs_real_file_path)
        plot(data=all_rotation_3_uncoded,caption='all_rotation_3_uncoded',save_dir = grasp_tsvs_real_file_path)
        # plt.show()
    
    elif format == 'rect':
        plot(data=all_score_uncoded,caption='all_score_uncoded',save_dir = grasp_tsvs_real_file_path)
        plot(data=all_height_uncoded,caption='all_height_uncoded',save_dir = grasp_tsvs_real_file_path)
        plot(data=all_center_point_1_uncoded,caption='all_center_point_1_uncoded',save_dir = grasp_tsvs_real_file_path)
        plot(data=all_center_point_2_uncoded,caption='all_center_point_2_uncoded',save_dir = grasp_tsvs_real_file_path)
        plot(data=all_open_point_1_uncoded,caption='all_open_point_1_uncoded',save_dir = grasp_tsvs_real_file_path)
        plot(data=all_open_point_2_uncoded,caption='all_open_point_2_uncoded',save_dir = grasp_tsvs_real_file_path)
        # plt.show()
        
def generate_grasp_txt_real_encoded(graspnet_root,camera,format,scene_sum):
    '''    
    @Function: save all grasp real encoded data in one txt file
            
    @Input:
        - graspnet_root: ROOT PATH FOR GRASPNET
        - camera: string of type of camera, "realsense" or "kinect".
        - format: string of grasp format, '6d' or 'rect'.
        - scene_sum: int of the number of scenes in the dataset.

    @Output: None
        - all_score_encoded.txt
        - all_wdith_encoded.txt
        - all_height_encoded.txt
        - all_depth_encoded.txt
        - all_translation_1_encoded.txt
        - all_translation_2_encoded.txt
        - all_translation_3_encoded.txt
        - all_rotation_1_encoded.txt
        - all_rotation_2_encoded.txt
        - all_rotation_3_encoded.txt
    '''

    grasp_tsvs_real_file_path=graspnet_root + '/grasp_tsvs_real/grasp_tsvs_real_'+camera+'_'+format+'_'+str(scene_sum)
    
    def save_grasp_txt_real(data,caption,txt_file_path):
        ''' save encoded data in txt file '''
        with open(txt_file_path, "w") as file:
            for item in data:          
                file.write(str(item)+"\n")
            file.close()
        print(f'====== {caption} txt file is created successfully!!! ======') 
    
    # ====== get all grasp real data ======
    if format == '6d':
        all_data = load_grasp_txt_real(grasp_tsvs_real_file_path,uncoded_or_encoded='uncoded',format=format)  
        all_score_uncoded,all_width_uncoded,all_height_uncoded,all_depth_uncoded,all_translation_1_uncoded,all_translation_2_uncoded,all_translation_3_uncoded,all_rotation_1_uncoded,all_rotation_2_uncoded,all_rotation_3_uncoded = all_data
    
    elif format == 'rect':
        all_data = load_grasp_txt_real(grasp_tsvs_real_file_path,uncoded_or_encoded='uncoded',format=format)  
        all_score_uncoded,all_height_uncoded,all_center_point_1_uncoded,all_center_point_2_uncoded,all_open_point_1_uncoded,all_open_point_2_uncoded = all_data
    
    # ====== encode all grasp real data ======
    if format == '6d':
        all_score_encoded = [int(x) for x in KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit_transform(all_score_uncoded.reshape(-1, 1))]
        all_width_encoded = [int(x) for x in KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit_transform(all_width_uncoded.reshape(-1, 1))]
        all_height_encoded = [int(x) for x in KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit_transform(all_height_uncoded.reshape(-1, 1))]
        all_depth_encoded = [int(x) for x in KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit_transform(all_depth_uncoded.reshape(-1, 1))]
        all_translation_1_encoded = [int(x) for x in KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit_transform(all_translation_1_uncoded.reshape(-1, 1))]
        all_translation_2_encoded = [int(x) for x in KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit_transform(all_translation_2_uncoded.reshape(-1, 1))]
        all_translation_3_encoded = [int(x) for x in KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit_transform(all_translation_3_uncoded.reshape(-1, 1))]
        all_rotation_1_encoded = [int(x) for x in KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit_transform(all_rotation_1_uncoded.reshape(-1, 1))]
        all_rotation_2_encoded = [int(x) for x in KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit_transform(all_rotation_2_uncoded.reshape(-1, 1))]
        all_rotation_3_encoded = [int(x) for x in KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit_transform(all_rotation_3_uncoded.reshape(-1, 1))]
    
        # ====== special data ======
        all_depth_encoded = [50 if 0.005 <= num <= 0.015 else 100 if 0.015 < num <= 0.025 else 150 if 0.025 < num <= 0.035 else 200 for num in all_depth_uncoded]

    elif format == 'rect':
        all_score_encoded = [int(x) for x in KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit_transform(all_score_uncoded.reshape(-1, 1))]
        all_height_encoded = [int(x) for x in KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit_transform(all_height_uncoded.reshape(-1, 1))]
        all_center_point_1_encoded = [int(x) for x in KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit_transform(all_center_point_1_uncoded.reshape(-1, 1))]
        all_center_point_2_encoded = [int(x) for x in KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit_transform(all_center_point_2_uncoded.reshape(-1, 1))]
        all_open_point_1_encoded = [int(x) for x in KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit_transform(all_open_point_1_uncoded.reshape(-1, 1))]
        all_open_point_2_encoded = [int(x) for x in KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit_transform(all_open_point_2_uncoded.reshape(-1, 1))]
        
    # ====== save grasp real encoded data in txt files ======
    if format == '6d':
        save_grasp_txt_real(data=all_score_encoded,caption='all_score_encoded',txt_file_path = grasp_tsvs_real_file_path+'/all_score_encoded.txt')
        save_grasp_txt_real(data=all_width_encoded,caption='all_width_encoded',txt_file_path = grasp_tsvs_real_file_path+'/all_width_encoded.txt')
        save_grasp_txt_real(data=all_height_encoded,caption='all_height_encoded',txt_file_path = grasp_tsvs_real_file_path+'/all_height_encoded.txt')
        save_grasp_txt_real(data=all_depth_encoded,caption='all_depth_encoded',txt_file_path = grasp_tsvs_real_file_path+'/all_depth_encoded.txt')
        save_grasp_txt_real(data=all_translation_1_encoded,caption='all_translation_1_encoded',txt_file_path = grasp_tsvs_real_file_path+'/all_translation_1_encoded.txt')
        save_grasp_txt_real(data=all_translation_2_encoded,caption='all_translation_2_encoded',txt_file_path = grasp_tsvs_real_file_path+'/all_translation_2_encoded.txt')
        save_grasp_txt_real(data=all_translation_3_encoded,caption='all_translation_3_encoded',txt_file_path = grasp_tsvs_real_file_path+'/all_translation_3_encoded.txt')
        save_grasp_txt_real(data=all_rotation_1_encoded,caption='all_rotation_1_encoded',txt_file_path = grasp_tsvs_real_file_path+'/all_rotation_1_encoded.txt')
        save_grasp_txt_real(data=all_rotation_2_encoded,caption='all_rotation_2_encoded',txt_file_path = grasp_tsvs_real_file_path+'/all_rotation_2_encoded.txt')
        save_grasp_txt_real(data=all_rotation_3_encoded,caption='all_rotation_3_encoded',txt_file_path = grasp_tsvs_real_file_path+'/all_rotation_3_encoded.txt')
    
    elif format == 'rect':
        save_grasp_txt_real(data=all_score_encoded,caption='all_score_encoded',txt_file_path = grasp_tsvs_real_file_path+'/all_score_encoded.txt')
        save_grasp_txt_real(data=all_height_encoded,caption='all_height_encoded',txt_file_path = grasp_tsvs_real_file_path+'/all_height_encoded.txt')
        save_grasp_txt_real(data=all_center_point_1_encoded,caption='all_center_point_1_encoded',txt_file_path = grasp_tsvs_real_file_path+'/all_center_point_1_encoded.txt')
        save_grasp_txt_real(data=all_center_point_2_encoded,caption='all_center_point_2_encoded',txt_file_path = grasp_tsvs_real_file_path+'/all_center_point_2_encoded.txt')
        save_grasp_txt_real(data=all_open_point_1_encoded,caption='all_open_point_1_encoded',txt_file_path = grasp_tsvs_real_file_path+'/all_open_point_1_encoded.txt')
        save_grasp_txt_real(data=all_open_point_2_encoded,caption='all_open_point_2_encoded',txt_file_path = grasp_tsvs_real_file_path+'/all_open_point_2_encoded.txt')
    
    
def plot_grasp_txt_real_encoded(graspnet_root,camera,format,scene_sum):
    '''    
    @Function: plot image of the real encoded data distribution from txt files and save
            
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

    grasp_tsvs_real_file_path=graspnet_root + '/grasp_tsvs_real/grasp_tsvs_real_'+camera+'_'+format+'_'+str(scene_sum)
       
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
    
    # ====== get all grasp real encoded data ======
    if format == '6d':
        all_data = load_grasp_txt_real(grasp_tsvs_real_file_path,uncoded_or_encoded='encoded',format=format)  
        all_score_encoded,all_width_encoded,all_height_encoded,all_depth_encoded,all_translation_1_encoded,all_translation_2_encoded,all_translation_3_encoded,all_rotation_1_encoded,all_rotation_2_encoded,all_rotation_3_encoded = all_data
    
    elif format == 'rect':
        all_data = load_grasp_txt_real(grasp_tsvs_real_file_path,uncoded_or_encoded='encoded',format=format)  
        all_score_encoded,all_height_encoded,all_center_point_1_encoded,all_center_point_2_encoded,all_open_point_1_encoded,all_open_point_2_encoded = all_data
    
    # ====== plot and save ======
    if format == '6d':
        plot(data=all_score_encoded,caption='all_score_encoded',save_dir = grasp_tsvs_real_file_path)
        plot(data=all_width_encoded,caption='all_width_encoded',save_dir = grasp_tsvs_real_file_path)
        plot(data=all_height_encoded,caption='all_height_encoded',save_dir = grasp_tsvs_real_file_path)
        plot(data=all_depth_encoded,caption='all_depth_encoded',save_dir = grasp_tsvs_real_file_path)
        plot(data=all_translation_1_encoded,caption='all_translation_1_encoded',save_dir = grasp_tsvs_real_file_path)
        plot(data=all_translation_2_encoded,caption='all_translation_2_encoded',save_dir = grasp_tsvs_real_file_path)
        plot(data=all_translation_3_encoded,caption='all_translation_3_encoded',save_dir = grasp_tsvs_real_file_path)
        plot(data=all_rotation_1_encoded,caption='all_rotation_1_encoded',save_dir = grasp_tsvs_real_file_path)
        plot(data=all_rotation_2_encoded,caption='all_rotation_2_encoded',save_dir = grasp_tsvs_real_file_path)
        plot(data=all_rotation_3_encoded,caption='all_rotation_3_encoded',save_dir = grasp_tsvs_real_file_path)
        # plt.show()
    
    elif format == 'rect':
        plot(data=all_score_encoded,caption='all_score_encoded',save_dir = grasp_tsvs_real_file_path)
        plot(data=all_height_encoded,caption='all_height_encoded',save_dir = grasp_tsvs_real_file_path)
        plot(data=all_center_point_1_encoded,caption='all_center_point_1_encoded',save_dir = grasp_tsvs_real_file_path)
        plot(data=all_center_point_2_encoded,caption='all_center_point_2_encoded',save_dir = grasp_tsvs_real_file_path)
        plot(data=all_open_point_1_encoded,caption='all_open_point_1_encoded',save_dir = grasp_tsvs_real_file_path)
        plot(data=all_open_point_2_encoded,caption='all_open_point_2_encoded',save_dir = grasp_tsvs_real_file_path)
        # plt.show()

def generate_grasp_tsvs_real(graspnet_root,train_or_test,camera,format,scene_sum,sceneId_start,sceneId_end,annId_start,annId_end,if_generate_grasp_tsvs_real_uncoded=False,if_generate_grasp_txt_real_uncoded=False,if_plot_grasp_txt_real_uncoded=False,if_generate_grasp_txt_real_encoded=False,if_plot_grasp_txt_real_encoded=False):
    '''    
    @Function: 
        - 1.get all real uncoded grasp info data from json files and save them in tsv files.
        - 2.save all grasp real_uncoded data from tsv files in one txt file.
        - 3.plot image of the real_uncoded data distribution from txt files and save
        - 4.save all grasp real encoded data in one txt file
        - 5.plot image of the real encoded data distribution from txt files and save
        
            
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

                - all_score_uncoded.txt
                - all_width_uncoded.txt
                - all_height_uncoded.txt
                - all_depth_uncoded.txt
                - all_translation_1_uncoded.txt
                - all_translation_2_uncoded.txt
                - all_translation_3_uncoded.txt
                - all_rotation_1_uncoded.txt
                - all_rotation_2_uncoded.txt
                - all_rotation_3_uncoded.txt

                - all_score_encoded.txt
                - all_wdith_encoded.txt
                - all_height_encoded.txt
                - all_depth_encoded.txt
                - all_translation_1_encoded.txt
                - all_translation_2_encoded.txt
                - all_translation_3_encoded.txt
                - all_rotation_1_encoded.txt
                - all_rotation_2_encoded.txt
                - all_rotation_3_encoded.txt
                
                - data_num_and_sum.json

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
    if if_generate_grasp_tsvs_real_uncoded:
        generate_grasp_tsvs_real_uncoded(graspnet_root,train_or_test,camera,format,scene_sum,sceneId_start,sceneId_end,annId_start,annId_end)
    
    if if_generate_grasp_txt_real_uncoded:
        generate_grasp_txt_real_uncoded(graspnet_root,camera,format,scene_sum)
    
    if if_plot_grasp_txt_real_uncoded:
        plot_grasp_txt_real_uncoded(graspnet_root,camera,format,scene_sum)
    
    if if_generate_grasp_txt_real_encoded:
        generate_grasp_txt_real_encoded(graspnet_root,camera,format,scene_sum)
    
    if if_plot_grasp_txt_real_encoded:
        plot_grasp_txt_real_encoded(graspnet_root,camera,format,scene_sum)
        
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
    if_generate_grasp_tsvs_real_uncoded=False
    if_generate_grasp_txt_real_uncoded=False
    if_plot_grasp_txt_real_uncoded=False
    if_generate_grasp_txt_real_encoded=False
    if_plot_grasp_txt_real_encoded=False
    
    generate_grasp_tsvs_real(graspnet_root=graspnet_root,train_or_test=train_or_test,camera=camera,format=format,scene_sum=scene_sum,sceneId_start=sceneId_start,sceneId_end=sceneId_end,annId_start=annId_start,annId_end=annId_end,if_generate_grasp_tsvs_real_uncoded=if_generate_grasp_tsvs_real_uncoded,if_generate_grasp_txt_real_uncoded=if_generate_grasp_txt_real_uncoded,if_plot_grasp_txt_real_uncoded=if_plot_grasp_txt_real_uncoded,if_generate_grasp_txt_real_encoded=if_generate_grasp_txt_real_encoded,if_plot_grasp_txt_real_encoded=if_plot_grasp_txt_real_encoded)
    

if __name__ == "__main__":
    main()