__author__ = 'Zhi Wang'
__date__ = '2023.11.10'

import json
import os
import base64
from io import BytesIO
from PIL import Image
import csv
import re
import random
import numpy as np

from src.generate_grasp_tsvs_real import load_grasp_txt_real
  
def generate_grasp_tsvs_train(graspnet_root,train_or_test,camera,format,scene_sum,sceneId_start,sceneId_end,annId_start,annId_end):
    '''
    @Function: generate image-text-3d-graspinfo pairs in tsv files for kosmos-e training
            
    @Input:
        - graspnet_root: ROOT PATH FOR GRASPNET
        - train_or_test: string, a flag of "train" or "test".
        - camera: string of type of camera, "realsense" or "kinect".
        - format: string of grasp format, '6d' or 'rect'.
        - scene_sum: int of the number of scenes in the dataset.
        - sceneId_start: int of the starting scene index.(sceneId max range is [0:180])
        - sceneId_end: int of the ending scene index.
        - annId_start: int of the starting annotation index.(annId max range is [0:256])
        - annId_end: int of the ending annotation index.

    @Output: None
        - grasp_tsvs_train
            - grasp_tsvs_train_kinect_6d_100
                - 0000
                - 0001
                    - 0000.tsv
                    - 0001.tsv
                        - FLAG(19 bins)
                        - text
                            - scene description: all_obj_names
                            - the name of grasped object
                            - grasp info(s,w,h,d,t1,t2,t3,r1,r2,r3)
                        - image(base64)
                        - image_width
                        - image_height
                        - the point cloud file path of the scene
                    ....
                    - 0255.tsv
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
    
    def get_img_size(filename):
        ''' get the size of an image '''
        with Image.open(filename) as img:
            width, height = img.size
            return width, height

    def img_to_base64(filename):
        ''' image_png to image_base64 '''
        with open(filename, 'rb') as file:
            image_data = file.read()
            base64_data = base64.b64encode(image_data).decode('utf-8')
            return base64_data
        
    def base64_to_img(base64_string, save_path):
        ''' image_base64 to image_png '''
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        image_data = base64.b64decode(base64_string)
        image_buffer = BytesIO(image_data)
        image = Image.open(image_buffer)
        image.save(save_path)
    
    # ====== get all grasp real encoded data ======
    grasp_tsvs_real_file_path = graspnet_root + '/grasp_tsvs_real/grasp_tsvs_real_'+camera+'_'+format+'_'+str(scene_sum)
    
    if format == '6d':
        all_data = load_grasp_txt_real(grasp_tsvs_real_file_path,uncoded_or_encoded='encoded',format=format)  
        all_score_encoded,all_width_encoded,all_height_encoded,all_depth_encoded,all_translation_1_encoded,all_translation_2_encoded,all_translation_3_encoded,all_rotation_1_encoded,all_rotation_2_encoded,all_rotation_3_encoded = all_data
    
    elif format == 'rect':
        all_data = load_grasp_txt_real(grasp_tsvs_real_file_path,uncoded_or_encoded='encoded',format=format)  
        all_score_encoded,all_height_encoded,all_center_point_1_encoded,all_center_point_2_encoded,all_open_point_1_encoded,all_open_point_2_encoded = all_data
    
    # ====== get data_sum from data_num_and_sum.json ======
    with open(grasp_tsvs_real_file_path + '/data_num_and_sum.json', 'r') as data_num_and_sum_json:
        data_num_and_sum = json.load(data_num_and_sum_json)
    
    # ====== for every scene ====== 
    for sceneId in list(range(sceneId_start, sceneId_end)):
        # ====== get the textual description of all object for one scene ====== 
        with open(graspnet_root + '/grasp_jsons/' + camera + '/' + format + '/{:04d}/'.format(sceneId)+'/all_obj_names_and_ids_dict.json','r') as json_file:
            all_obj_names_and_ids_dict = json.load(json_file)
        all_obj_names = generate_object_string(all_obj_names_and_ids_dict)
        
        # ====== the path of tsvs_train file====== 
        grasp_tsvs_train_for_one_scene_path = graspnet_root + '/grasp_tsvs_train/grasp_tsvs_train_'+camera+'_'+format+'_'+str(scene_sum)+'/{:04d}'.format(sceneId) 
        if not os.path.exists(grasp_tsvs_train_for_one_scene_path):
            os.makedirs(grasp_tsvs_train_for_one_scene_path)
        
        # ====== the path of point cloud file====== 
        grasp_scene_point_cloud_for_one_scene_path = graspnet_root + '/grasp_scene_point_clouds/' + camera + '/npy_points/{:04d}.npy'.format(sceneId) 
        
        # ====== for every ann ======
        for annId in list(range(annId_start, annId_end)):   
            all_grasp_tsvs_train_data_for_one_ann_in_one_scene = []
            
            # ====== get image ======
            img_path = graspnet_root+'/scenes/scene_' +'{:04d}'.format(sceneId) + '/' + camera + '/rgb'+ '/{:04}'.format(annId)+'.png'
            img_base64 = img_to_base64(img_path)
            # img_base64 = "====for test===="
            img_width, img_height = 1280,720 # img_width, img_height = get_img_size(img_path)    

            # ====== initialize data_num_for_this_ann ======
            data_num_for_this_ann = 0 if annId == 0 and sceneId == 0 else data_num_and_sum['data_sum_{:04d}_{:04d}'.format(sceneId-1, 255)] if annId == 0 else data_num_and_sum['data_sum_{:04d}_{:04d}'.format(sceneId, annId-1)]
            
            # ====== loop through all tsv files ======
            grasp_tsvs_real_for_one_ann_in_one_scene_path = grasp_tsvs_real_file_path + '/{:04d}'.format(sceneId)+ '/{:04d}'.format(annId)    
            
            # ====== for evert tsv file ======
            for root,dirs,files in os.walk(grasp_tsvs_real_for_one_ann_in_one_scene_path):
                files.sort()  
                for tsv_real_file in files:
                    if not tsv_real_file.endswith(".tsv"):
                        continue
                    # ====== get the name of grasped object ======
                    object_id = str(int(re.search(r"grasp_(\d+)_(\d+)\.tsv", tsv_real_file).group(1))) # grasp_05_00->05->5
                    object_name =all_obj_names_and_ids_dict[object_id]
                    
                    # ====== get FLAG ======
                    FLAG = '0'
                    FLAG += '00' if train_or_test == 'train' else ('01' if train_or_test == 'test' else None)
                    FLAG += '00' if camera == 'kinect' else ('01' if camera == 'realsense' else None)
                    FLAG += '00' if format == '6d' else ('01' if format == 'rect' else None)
                    FLAG += '{:04d}'.format(sceneId)
                    FLAG += '{:04d}'.format(annId)
                    FLAG += re.search(r"grasp_(\d+)_(\d+)\.tsv", tsv_real_file).group(1) # object_id
                    FLAG += re.search(r"grasp_(\d+)_(\d+)\.tsv", tsv_real_file).group(2) # grasp_num
                    
                    # ====== get graspinfo(encoded) ======
                    if format == '6d':
                        score = int(all_score_encoded[data_num_for_this_ann])
                        width = int(all_width_encoded[data_num_for_this_ann])
                        height = int(all_height_encoded[data_num_for_this_ann])
                        depth = int(all_depth_encoded[data_num_for_this_ann])
                        translation_1 = int(all_translation_1_encoded[data_num_for_this_ann])
                        translation_2 = int(all_translation_2_encoded[data_num_for_this_ann])
                        translation_3 = int(all_translation_3_encoded[data_num_for_this_ann])
                        rotation_1 = int(all_rotation_1_encoded[data_num_for_this_ann])
                        rotation_2 = int(all_rotation_2_encoded[data_num_for_this_ann])
                        rotation_3 = int(all_rotation_3_encoded[data_num_for_this_ann])
                        # ====== special data ======
                        score = 0
                        width = 0
                    
                    elif format == 'rect':
                        score = int(all_score_encoded[data_num_for_this_ann])
                        height = int(all_height_encoded[data_num_for_this_ann])
                        center_point_1 = int(all_center_point_1_encoded[data_num_for_this_ann])
                        center_point_2 = int(all_center_point_2_encoded[data_num_for_this_ann])
                        open_point_1 = int(all_open_point_1_encoded[data_num_for_this_ann])
                        open_point_2 = int(all_open_point_2_encoded[data_num_for_this_ann])
                        # ====== special data ======
                        score = 0
                    
                    # ====== update data_num_for_this_ann ======
                    data_num_for_this_ann += 1
                    
                    # ====== get text(all_obj_names;object_name;graspinfo) ======
                    random_number = random.randint(1, 50)
                    text_templates_json_file_path = graspnet_root+'/else/text_templates_'+format+'_50.json'
                    with open(text_templates_json_file_path,'r') as text_templates:
                        text_templates = json.load(text_templates)
                        text = text_templates[str(random_number)]
                        # print(f'====== original text is: ======\n{text}')
                    
                    if format == '6d':
                        text = text.replace("{all_obj_names}", str(all_obj_names))
                        text = text.replace("{object_name}", 'the '+str(object_name))
                        text = text.replace("{score}", 's_'+str(score))
                        text = text.replace("{width}", 'w_'+str(width))
                        text = text.replace("{height}", 'h_'+str(height))
                        text = text.replace("{depth}", 'd_'+str(depth))
                        text = text.replace("{translation}", 't1_'+str(translation_1)+' t2_'+str(translation_2)+' t3_'+str(translation_3))
                        text = text.replace("{rotation}", 'r1_'+str(rotation_1)+' r2_'+str(rotation_2)+' r3_'+str(rotation_3))

                    elif format == 'rect':
                        text = text.replace("{all_obj_names}", str(all_obj_names))
                        text = text.replace("{object_name}", 'the '+str(object_name))
                        text = text.replace("{score}", 's_'+str(score))
                        text = text.replace("{height}", 'h_'+str(height))
                        text = text.replace("{center_point}", 'cp1_'+str(center_point_1)+' cp2_'+str(center_point_2))
                        text = text.replace("{open_point}", 'op1_'+str(open_point_1)+' op2_'+str(open_point_2))
                    
                    # print(f'====== FLAG:====== \n{FLAG}')
                    # print(f'====== text:====== \n{text}') 
                    # print(f'====== img_base64:====== \n{img_base64}') 
                    # print(f'====== img_width:====== \n{img_width}') 
                    # print(f'====== img_height:====== \n{img_height}') 
                    # print(f'====== grasp_scene_point_cloud_for_one_scene_path:====== \n{grasp_scene_point_cloud_for_one_scene_path}') 
                    all_grasp_tsvs_train_data_for_one_ann_in_one_scene.append([FLAG,text,img_base64,str(img_width),str(img_height),grasp_scene_point_cloud_for_one_scene_path])
                    
            # ====== save FLAG+text+img_base64+image_height+image_width+grasp_scene_point_cloud_for_one_scene_path in tsv file ======
            grasp_tsvs_train_for_one_ann_in_one_scene_path = grasp_tsvs_train_for_one_scene_path +'/{:04d}.tsv'.format(annId)
            with open(grasp_tsvs_train_for_one_ann_in_one_scene_path,'w',newline='') as tsv_file:
                writer = csv.writer(tsv_file, delimiter='\t')
                for data in all_grasp_tsvs_train_data_for_one_ann_in_one_scene:
                    writer.writerow(data)  
    
            print(f'====== [scene_{sceneId},ann_{annId},num_{data_num_for_this_ann}] grasp tsvs train are generated successfully!!! ======')
                        

def main():
    graspnet_root = '/mnt/msranlpintern/dataset/graspnet-v2' #'D:/dataset/graspnet' 
    train_or_test = 'train'
    camera = 'kinect'
    format = '6d'
    sceneId_start = 0
    sceneId_end = 100
    annId_start = 0
    annId_end = 256
    scene_sum = 100

    generate_grasp_tsvs_train(graspnet_root = graspnet_root ,train_or_test = train_or_test, camera = camera , format = format ,scene_sum=scene_sum, sceneId_start = sceneId_start,sceneId_end = sceneId_start + 1,annId_start = annId_start,annId_end = 256)
    generate_grasp_tsvs_train(graspnet_root = graspnet_root ,train_or_test = train_or_test, camera = camera , format = format ,scene_sum=scene_sum, sceneId_start = sceneId_start + 1,sceneId_end = sceneId_end,annId_start = 0,annId_end = annId_end)
    
if __name__ == "__main__":
    main()