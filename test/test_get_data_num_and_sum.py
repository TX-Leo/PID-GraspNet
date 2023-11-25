import os
import json
import csv
import re
import transforms3d as tfs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

def generate_grasp_txt_real(graspnet_root,camera,format,scene_sum):
    '''    
    @Function: save all grasp real data from tsv files in one txt file
            
    @Input:
        - graspnet_root: ROOT PATH FOR GRASPNET
        - camera: string of type of camera, "realsense" or "kinect".
        - format: string of grasp format, '6d' or 'rect'.
        - scene_sum: int of the number of scenes in the dataset.

    @Output: None
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
    all_score = []
    all_width = []
    all_height = []
    all_depth = []
    all_translation_1 = []
    all_translation_2 = []
    all_translation_3 = []
    all_rotation_1 = []
    all_rotation_2 = []
    all_rotation_3 = []

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
                            if tsv_file_path.endswith(".json"):
                                with open(tsv_file_path, 'r') as data_num_and_sum_json:
                                    temp = json.load(data_num_and_sum_json)
                                    data_num_and_sum['data_num_'+dir+'_'+sub_dir] = temp["data_num_for_this_ann"]
                                    data_num_and_sum['data_sum_'+dir+'_'+sub_dir] = temp["data_num"]
                                break
                        break
                    
    # ====== save all data_num and data_sum ======
    with open(grasp_tsvs_real_file_path+'/data_num_and_sum.json','w') as data_num_and_sum_json:
        json.dump(data_num_and_sum, data_num_and_sum_json,indent=4)
    
    # print(f'====== There are {data_sum} data totally!!! ======') 
    
    # # ====== save grasp real data in txt files ======
    # save_grasp_txt_real(data=all_score,caption='[Score]',txt_file_path = grasp_tsvs_real_file_path+'/all_score.txt')
    # save_grasp_txt_real(data=all_width,caption='[Width]',txt_file_path = grasp_tsvs_real_file_path+'/all_width.txt')
    # save_grasp_txt_real(data=all_height,caption='[Height]',txt_file_path = grasp_tsvs_real_file_path+'/all_height.txt')
    # save_grasp_txt_real(data=all_depth,caption='[Depth]',txt_file_path = grasp_tsvs_real_file_path+'/all_depth.txt')
    # save_grasp_txt_real(data=all_translation_1,caption='[Translation-1]',txt_file_path = grasp_tsvs_real_file_path+'/all_translation_1.txt')
    # save_grasp_txt_real(data=all_translation_2,caption='[Translation-2]',txt_file_path = grasp_tsvs_real_file_path+'/all_translation_2.txt')
    # save_grasp_txt_real(data=all_translation_3,caption='[Translation-3]',txt_file_path = grasp_tsvs_real_file_path+'/all_translation_3.txt')
    # save_grasp_txt_real(data=all_rotation_1,caption='[Rotation-1]',txt_file_path = grasp_tsvs_real_file_path+'/all_rotation_1.txt')
    # save_grasp_txt_real(data=all_rotation_2,caption='[Rotation-2]',txt_file_path = grasp_tsvs_real_file_path+'/all_rotation_2.txt')
    # save_grasp_txt_real(data=all_rotation_3,caption='[Rotation-3]',txt_file_path = grasp_tsvs_real_file_path+'/all_rotation_3.txt')
    
    
def main():
    graspnet_root = r'C:\Users\v-zhiwang2\Downloads\graspnet-v2' #'D:/dataset/graspnet' # 'D:/dataset/graspnet' # /Volumes/WD/dataset/graspnet
    train_or_test = 'train'
    camera = 'kinect'
    format = '6d'
    scene_sum = 100
    sceneId_start = 0
    sceneId_end = 1
    annId_start = 0
    annId_end = 1
    
    # # generate_grasp_tsvs_real
    # generate_grasp_tsvs_real(graspnet_root,train_or_test,camera,format,scene_sum,sceneId_start,sceneId_end,annId_start,annId_end)
    
    # generate_grasp_txt_real
    generate_grasp_txt_real(graspnet_root,camera,format,scene_sum)
    
    # # plot_grasp_txt_real
    # plot_grasp_txt_real(graspnet_root,camera,format,scene_sum)
    
    # # generate_grasp_txt_real_encoded
    # generate_grasp_txt_real_encoded(graspnet_root,camera,format,scene_sum)
    
    # # plot_grasp_txt_real_encoded
    # plot_grasp_txt_real_encoded(graspnet_root,camera,format,scene_sum)

if __name__ == "__main__":
    main()