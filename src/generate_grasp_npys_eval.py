__author__ = 'Zhi Wang'
__date__ = '2023.11.10'

import os
import json
import re
import numpy as np
import transforms3d as tfs

from src.generate_grasp_tsvs_predicted import load_grasp_txt_predicted
from graspnetAPI import GraspNet,GraspGroup,RectGraspGroup

def generate_grasp_npys_eval(graspnet_root,train_or_test,camera,format,scene_sum,sceneId_start,sceneId_end,annId_start,annId_end):
    '''
    @Function: generate GraspGroup in npy for eval
            
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
    '''

    # ====== get all grasp predicted uncoded data ======
    grasp_tsvs_predicted_file_path = graspnet_root + '/grasp_tsvs_predicted/grasp_tsvs_predicted_'+camera+'_'+format+'_'+str(scene_sum)
    if format == '6d':
        all_data = load_grasp_txt_predicted(grasp_tsvs_predicted_file_path,uncoded_or_encoded='uncoded',format=format)  
        all_score_uncoded,all_width_uncoded,all_height_uncoded,all_depth_uncoded,all_translation_1_uncoded,all_translation_2_uncoded,all_translation_3_uncoded,all_rotation_1_uncoded,all_rotation_2_uncoded,all_rotation_3_uncoded = all_data
    
    elif format == 'rect':
        all_data = load_grasp_txt_predicted(grasp_tsvs_predicted_file_path,uncoded_or_encoded='uncoded',format=format)  
        all_score_uncoded,all_height_uncoded,all_center_point_1_uncoded,all_center_point_2_uncoded,all_open_point_1_uncoded,all_open_point_2_uncoded = all_data
        
    # ====== get data_sum from data_num_and_sum.json ======
    with open(grasp_tsvs_predicted_file_path + '/data_num_and_sum.json', 'r') as data_num_and_sum_json:
        data_num_and_sum = json.load(data_num_and_sum_json)

    # ====== for every scene ====== 
    for sceneId in list(range(sceneId_start, sceneId_end)):
        
        # ====== the path of npys_eval file====== 
        grasp_npys_eval_for_one_scene_path = graspnet_root + '/grasp_npys_eval/grasp_npys_eval_'+camera+'_'+format+'_'+str(scene_sum)+'/scene_{:04d}/'.format(sceneId) + camera + '/'
        if not os.path.exists(grasp_npys_eval_for_one_scene_path):
            os.makedirs(grasp_npys_eval_for_one_scene_path)

        # ====== for every ann ======
        for annId in list(range(annId_start, annId_end)):   
            
            # ====== initialize the grasp_group_for_one_ann_in_one_scene_list ======
            grasp_array_list_for_one_ann_in_one_scene = []

            # ====== initialize data_num_for_this_ann ======
            data_num_for_this_ann = 0 if annId == 0 and sceneId == 0 else data_num_and_sum['data_sum_{:04d}_{:04d}'.format(sceneId-1, 255)] if annId == 0 else data_num_and_sum['data_sum_{:04d}_{:04d}'.format(sceneId, annId-1)]

            # ====== loop through all tsv files ======
            grasp_tsvs_predicted_for_one_ann_in_one_scene_path = grasp_tsvs_predicted_file_path + '/{:04d}'.format(sceneId)+ '/{:04d}'.format(annId)    
            
            # ====== for evert tsv file ======
            for root,dirs,files in os.walk(grasp_tsvs_predicted_for_one_ann_in_one_scene_path):
                files.sort()  
                for tsv_predicted_file in files:
                    if not tsv_predicted_file.endswith(".tsv"):
                        continue
                    
                    # ====== get the name of grasped object ======
                    object_id = str(int(re.search(r"grasp_(\d+)_(\d+)\.tsv", tsv_predicted_file).group(1))) # grasp_05_00->05->5
                    
                    # ====== get graspinfo(uncoded) ======
                    if format == '6d':
                        score = all_score_uncoded[data_num_for_this_ann]
                        width = all_width_uncoded[data_num_for_this_ann]
                        height = all_height_uncoded[data_num_for_this_ann]
                        depth = all_depth_uncoded[data_num_for_this_ann]
                        translation_1 = all_translation_1_uncoded[data_num_for_this_ann]
                        translation_2 = all_translation_2_uncoded[data_num_for_this_ann]
                        translation_3 = all_translation_3_uncoded[data_num_for_this_ann]
                        rotation_1 = all_rotation_1_uncoded[data_num_for_this_ann]
                        rotation_2 = all_rotation_2_uncoded[data_num_for_this_ann]
                        rotation_3 = all_rotation_3_uncoded[data_num_for_this_ann]
                        
                        # ====== get translation and rotation:3*1->3*3 ======
                        translation = np.array([translation_1,translation_2,translation_3])
                        rotation_matrix = tfs.euler.euler2mat(rotation_1, rotation_2, rotation_3, 'sxyz')

                        # ====== get grasp_array ======
                        grasp_array = np.concatenate([np.array((score, width, height, depth)),rotation_matrix.reshape(-1), translation, np.array((int(object_id))).reshape(-1)]).astype(np.float64)#.tolist()
                    
                    elif format == 'rect':
                        score = all_score_uncoded[data_num_for_this_ann]
                        height = all_height_uncoded[data_num_for_this_ann]
                        center_point_1 = all_center_point_1_uncoded[data_num_for_this_ann]
                        center_point_2 = all_center_point_2_uncoded[data_num_for_this_ann]
                        open_point_1 = all_open_point_1_uncoded[data_num_for_this_ann]
                        open_point_2 = all_open_point_2_uncoded[data_num_for_this_ann]
                        
                        # ====== get center_point and open_point ======
                        center_point = np.array([center_point_1,center_point_2])
                        open_point = np.array([open_point_1,open_point_2])
                        
                        # ====== get grasp_array ======
                        grasp_array = np.concatenate([np.array((score,height)),center_point.reshape(-1), open_point.reshape(-1), np.array((int(object_id))).reshape(-1)]).astype(np.float64)#.tolist()
                        
                    # ====== add this graspinfo to grasp array list ======
                    grasp_array_list_for_one_ann_in_one_scene.append(grasp_array)
                    
                    # ====== update data_num_for_this_ann ======
                    data_num_for_this_ann += 1

            # ====== get GraspGroup(6d) ======
            if format == '6d':
                gg=GraspGroup(np.array(grasp_array_list_for_one_ann_in_one_scene))
            
            elif format == 'rect':
                # ====== RectGraspGroup to GraspGroup ======
                gg_rect=RectGraspGroup(np.array(grasp_array_list_for_one_ann_in_one_scene))
                g = GraspNet(graspnet_root, camera = camera, split = 'all')
                depth = g.loadDepth(sceneId = sceneId, camera = camera, annId = annId)
                gg = gg_rect.to_grasp_group(camera, depth)
            
            # ====== save GraspGroup in npy file ======
            grasp_npys_eval_for_one_ann_in_one_scene_path = grasp_npys_eval_for_one_scene_path +'/{:04d}.npy'.format(annId)
            gg.save_npy(grasp_npys_eval_for_one_ann_in_one_scene_path)
            
            print(f'====== [scene_{sceneId},ann_{annId},num_{data_num_for_this_ann}] grasp npys eval are generated successfully!!! ======')
            
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
    
    generate_grasp_npys_eval(graspnet_root=graspnet_root,train_or_test=train_or_test,camera=camera,format=format,scene_sum=scene_sum,sceneId_start=sceneId_start,sceneId_end=sceneId_end,annId_start=annId_start,annId_end=annId_end)

if __name__ == "__main__":
    main()