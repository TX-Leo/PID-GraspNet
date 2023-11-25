__author__ = 'Zhi Wang'
__date__ = '2023.11.10'

import os
import json

def check_graspnet_1billion(root):
    '''
    @Function: check_graspnet_1billion
        - scene
            - scene_0000
            - scene_0001
            - object_id_list.txt
            - rs_wrt_kn.npy
            - kinect
                - rgb
                    - 0000.png to 0255.png
                - depth
                    - 0000.png to 0255.png
                - label
                    - 0000.png to 0255.png
                - annotations
                    - 0000.xml to 0255.xml
                - meta
                    - 0000.mat to 0255.mat
                - rect
                    - 0000.npy to 0255.npy
                - camK.npy
                - camera_poses.npy
                - cam0_wrt_table.npy
            - realsense
            - ....
            - scene_0189
        - models
            - 000
            - 001
            - nontextured.ply
            - nontextured_simplified.ply
            - textured.jpg
            - textured.obj
            - textured.obj.mtl
            - textured.sdf
            - ....
            - 087
            - readme.txt
            - sim_mesh.mlx
            - sim_mesh.py
            - updates.txt
        - dex_models
            - 000.okl
            - 001.pkl
            - ....
            - 087.pkl
        - grasp_label
            - 000_labels.npz
            - 001_labels.npz
            - ....
            - 087_labels.npz
        - collision_label
            - scene_0000
            - scene_0001
                - collision_labels.npz
            - ....
            - scene_0189
        
    @Input: root dir of graspnet_1billion

    @Output: None
    '''
    
    from graspnetAPI import GraspNet
    g = GraspNet(root, 'kinect', 'all')
    if g.checkDataCompleteness():
        print('====== Check for kinect passed ======')
    g = GraspNet(root, 'realsense', 'all')
    if g.checkDataCompleteness():
        print('====== Check for realsense passed ======')
        
    print(f'====== finish checking graspnet_1billion ======')
        
def check_folder(folder_path):
    '''
    @Function: check if a folder exists
        
    @Input: path of a folder
    
    @Output: None
    '''
    
    if not os.path.isdir(folder_path):
        print(f"====== {folder_path} doesn't exist ======")

def check_file(file_path):
    '''
    @Function: check if a file exists
        
    @Input: path of a file
    
    @Output: None
    '''
    
    if not os.path.isfile(file_path):
        print(f"====== {file_path} doesn't exist ======")

def check_grasp_scene_point_clouds(root,sceneId_start,sceneId_end):
    '''
    @Function: check_grasp_scene_point_clouds
        - grasp_scene_point_clouds
            - realsense
            - kinect
                - pcd
                - npy_points
                - npy_points_and_colors
                    - 0000.npy
                    - 0001.npy
                    - ....
                    - 0099.npy
        
    @Input: root dir of grasp_scene_point_clouds

    @Output: None
    '''
    
    # ====== for every scene ======
    for sceneId in list(range(sceneId_start, sceneId_end)):
        # ====== check scene ======
        scene_folder = root+'/{:04d}.npy'.format(sceneId)
        check_file(scene_folder)
        print(f'====== [scene_{sceneId}] finish checking grasp_scene_point_clouds ======')
    print(f'====== finish checking grasp_scene_point_clouds ======')

def check_grasp_jsons(root,sceneId_start,sceneId_end,annId_start,annId_end):
    '''
    @Function: check_grasp_jsons
        - grasp_jsons
            - realsense
            - kinect
                - rect
                    - 0000
                    - 0001
                        - 0000
                        - 0001
                            - grasp_05_00.json(grasp_id_num.json)
                            - grasp_05_01.json
                                - rbg_img_path
                                - depth_img_path
                                - all_obj_names_and_ids_dict
                                - object_id
                                - object_name
                                - score
                                - height
                                - center_point(2)
                                - open_point(2)
                            - ....
                            - grasp_05_09.json
                            - grasp_11_00.json
                            - grasp_11_01.json
                            - ....
                        - ....
                        - 0255
                    - ....
                    - 0099
                - 6d
                    - 0000
                    - 0001
                        - 0000
                        - 0001
                            - grasp_05_00.json(grasp_id_num.json)
                            - grasp_05_01.json
                                - rbg_img_path
                                - depth_img_path
                                - all_obj_names_and_ids_dict
                                - object_id
                                - object_name
                                - score
                                - width
                                - height
                                - depth
                                - translation(3*1)
                                - rotation_matrix(3*3)
                            - ....
                            - grasp_05_09.json
                            - grasp_11_00.json
                            - grasp_11_01.json
                            - ....
                        - ....
                        - 0255
                    - ....
                    - 0099
        
    @Input: root dir of grasp_jsons

    @Output: None
    '''
    
    # ====== for every scene ======
    for sceneId in list(range(sceneId_start, sceneId_end)):
        # ====== check scene ======
        scene_folder = root+'/{:04d}'.format(sceneId)
        check_folder(scene_folder)
        
        # ====== check all_obj_names_and_ids_dict.json ======
        all_obj_names_and_ids_dict_file = f"{scene_folder}/all_obj_names_and_ids_dict.json"
        check_file(all_obj_names_and_ids_dict_file)
        
        # ====== for every ann ======
        for annId in list(range(annId_start, annId_end)):
            # ====== check ann ======
            ann_folder = scene_folder + '/{:04d}'.format(annId)
            check_folder(ann_folder)
            # print(f'====== [scene_{sceneId},ann_{annId}] finish checking grasp_jsons ======')
        print(f'====== [scene_{sceneId}] finish checking grasp_jsons ======')
    print(f'====== finish checking grasp_jsons ======')


def check_grasp_tsvs_real(root,sceneId_start,sceneId_end,annId_start,annId_end):
    '''
    @Function: check_grasp_tsvs_real
        - grasp_tsvs_real 
            - grasp_tsvs_real_kinect_6d_100
                - 0000
                - 0001
                    - 0000
                    - 0001
                        - grasp_05_00.tsv(grasp_id_num.tsv)
                        - grasp_05_01.tsv
                            - score
                            - width
                            - height
                            - depth
                            - translation_1
                            - translation_2
                            - translation_3
                            - rotation_1
                            - rotation_2
                            - rotation_3
                        - ....
                        - grasp_05_09.tsv
                        - grasp_11_00.tsv
                        - grasp_11_01.tsv
                        - ....
                    - ....
                    - 0256
                - ....
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
        
    @Input: root dir of grasp_tsvs_real

    @Output: None
    '''
    
    # ====== for every scene ======
    for sceneId in list(range(sceneId_start, sceneId_end)):
        # ====== check scene ======
        scene_folder = root+'/{:04d}'.format(sceneId)
        check_folder(scene_folder)

        # ====== check all_data_uncoded.txt and all_data_encoded.txt ======
        check_file(root+'/all_score_uncoded.txt')
        check_file(root+'/all_width_uncoded.txt')
        check_file(root+'/all_height_uncoded.txt')
        check_file(root+'/all_depth_uncoded.txt')
        check_file(root+'/all_translation_1_uncoded.txt')
        check_file(root+'/all_translation_2_uncoded.txt')
        check_file(root+'/all_translation_3_uncoded.txt')
        check_file(root+'/all_rotation_1_uncoded.txt')
        check_file(root+'/all_rotation_2_uncoded.txt')
        check_file(root+'/all_rotation_3_uncoded.txt')
        check_file(root+'/all_score_encoded.txt')
        check_file(root+'/all_width_encoded.txt')
        check_file(root+'/all_height_encoded.txt')
        check_file(root+'/all_depth_encoded.txt')
        check_file(root+'/all_translation_1_encoded.txt')
        check_file(root+'/all_translation_2_encoded.txt')
        check_file(root+'/all_translation_3_encoded.txt')
        check_file(root+'/all_rotation_1_encoded.txt')
        check_file(root+'/all_rotation_2_encoded.txt')
        check_file(root+'/all_rotation_3_encoded.txt')
        
        # ====== check all_data_uncoded_distribution.png and all_data_encoded_distribution.png ======
        check_file(root+'/all_score_uncoded_distribution.png')
        check_file(root+'/all_width_uncoded_distribution.png')
        check_file(root+'/all_height_uncoded_distribution.png')
        check_file(root+'/all_depth_uncoded_distribution.png')
        check_file(root+'/all_translation_1_uncoded_distribution.png')
        check_file(root+'/all_translation_2_uncoded_distribution.png')
        check_file(root+'/all_translation_3_uncoded_distribution.png')
        check_file(root+'/all_rotation_1_uncoded_distribution.png')
        check_file(root+'/all_rotation_2_uncoded_distribution.png')
        check_file(root+'/all_rotation_3_uncoded_distribution.png')
        check_file(root+'/all_score_encoded_distribution.png')
        check_file(root+'/all_width_encoded_distribution.png')
        check_file(root+'/all_height_encoded_distribution.png')
        check_file(root+'/all_depth_encoded_distribution.png')
        check_file(root+'/all_translation_1_encoded_distribution.png')
        check_file(root+'/all_translation_2_encoded_distribution.png')
        check_file(root+'/all_translation_3_encoded_distribution.png')
        check_file(root+'/all_rotation_1_encoded_distribution.png')
        check_file(root+'/all_rotation_2_encoded_distribution.png')
        check_file(root+'/all_rotation_3_encoded_distribution.png')
        
        # ====== check data_num_and_sum.json ======
        check_file(root+'/data_num_and_sum.json')
        
        # ====== for every ann ======
        for annId in list(range(annId_start, annId_end)):
            # ====== check ann ======
            ann_folder = scene_folder + '/{:04d}'.format(annId)
            check_folder(ann_folder)
            # print(f'====== [scene_{sceneId},ann_{annId}] finish checking grasp_tsvs_real ======')
        print(f'====== [scene_{sceneId}] finish checking grasp_tsvs_real ======')
    print(f'====== finish checking grasp_tsvs_real ======')

def check_grasp_tsvs_train(root,sceneId_start,sceneId_end,annId_start,annId_end):
    '''
    @Function: check_grasp_tsvs_train
        - grasp_tsvs_train
            - grasp_tsvs_train_kinect_6d_100
                - 0000
                - 0001
                    - 0000.tsv
                    - 0001.tsv
                        - FLAG(19 bins)
                            - 0(no meaning)
                            - train=00(test=01)
                            - kinect=00(realsense=01)
                            - rect=01(6d=00)
                            - scene=0000-0100
                            - ann=0000-0256
                            - id=00(xx)
                            - num=00-09
                        - text(select one prompt from the prompt repertory (30 prompts))
                            - scene description: all_obj_names
                            - the name of grasped object
                            - grasp info(s,w,h,d,t1,t2,t3,r1,r2,r3)
                            - example:  This is a picture of {all_obj_names}. And the {object_name} is to be grasped. The score of the grasp is {score}. The width of the grasp is {width}. The height of the grasp is {height}. The depth of the grasp is {depth}. The translation_matrix of the grasp is {translation}. The rotation_matrix is {rotation}.
                        - image(base64)
                        - image_width
                        - image_height
                        - the point cloud file path of the scene
                    - ....
                    - 0255.tsv
                - ....
                - 0099
        
    @Input: root dir of grasp_tsvs_train

    @Output: None
    '''
    
    # ====== for every scene ======
    for sceneId in list(range(sceneId_start, sceneId_end)):
        # ====== check scene ======
        scene_folder = root+'/{:04d}'.format(sceneId)
        check_folder(scene_folder)
        
        # ====== for every ann ======
        for annId in list(range(annId_start, annId_end)):
            # ====== check ann ======
            ann_folder = scene_folder + '/{:04d}.tsv'.format(annId)
            check_file(ann_folder)
            # print(f'====== [scene_{sceneId},ann_{annId}] finish checking grasp_tsvs_train ======')
        print(f'====== [scene_{sceneId}] finish checking grasp_tsvs_train ======')
    print(f'====== finish checking grasp_tsvs_train ======')

def check_grasp_dataloader_config(root):
    '''
    @Function: check_grasp_dataloader_config
        - grasp_dataloder_config
            - grasp_dataloder_config_kinect_6d_100
                - json(train:valid=9:1)
                    - train.json
                    - valid.json
                - sentencepiece.bpe.model
                - dict.txt
        
    @Input: root dir of grasp_dataloader_config

    @Output: None
    '''
    
    check_file(root+'/json/train.json')
    check_file(root+'/json/valid.json')
    check_file(root+'/sentencepiece.bpe.model')
    check_file(root+'/dict.txt')
    
    print(f'====== finish checking grasp_dataloader_config ======')
    
def check_else(root):
    '''
    @Function: check_else
        - else
            - all_obj_names_and_ids_final.json
            - text_templates_6d_50.json
    
    @Input: root dir of else

    @Output: None
    '''
    
    # ====== check all_obj_names_and_ids_final.json======
    all_obj_names_and_ids_final_file = root+'/all_obj_names_and_ids_final.json'
    check_file(all_obj_names_and_ids_final_file)
    
    # ====== check text_templates_6d_50.json======
    text_templates_6d_50_file = root+'/text_templates_6d_50.json'
    check_file(text_templates_6d_50_file)
    
    print(f'====== finish checking else ======')

def check_dataset(graspnet_root,camera,format,scene_sum,sceneId_start,sceneId_end,annId_start,annId_end,if_check_graspnet_1billion=False,if_check_grasp_scene_point_clouds=False,if_check_grasp_jsons=False,if_check_grasp_tsvs_real=False,if_check_grasp_tsvs_train=False,if_check_grasp_dataloader_config=False,if_check_else=False):
    '''
    @Function:
        - 1.check_graspnet_1billion
        - 2.check_grasp_scene_point_clouds
        - 3.check_grasp_jsons
        - 4.check_grasp_tsvs_real
        - 5.check_grasp_tsvs_train
        - 6.check_grasp_dataloader_config
        - 7.check_else
            
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
        - if_check_graspnet_1billion
        - if_check_grasp_scene_point_clouds
        - if_check_grasp_jsons
        - if_check_grasp_tsvs_real
        - if_check_grasp_tsvs_train
        - if_check_grasp_dataloader_config
        - if_check_else

    @Output: None
        - scene
        - models
        - dex_models
        - grasp_label
        - collision_label
        - grasp_scene_point_clouds
        - grasp_jsons
        - grasp_tsvs_real
        - grasp_tsvs_train
        - grasp_dataloader_config
        - else
    '''
    
    if if_check_graspnet_1billion:
        check_graspnet_1billion(graspnet_root)
    if if_check_grasp_scene_point_clouds:
        check_grasp_scene_point_clouds(graspnet_root+'/grasp_scene_point_clouds/'+camera+'/npy_points/',sceneId_start,sceneId_end)
    if if_check_grasp_jsons:
        check_grasp_jsons(graspnet_root+'/grasp_jsons/'+camera+'/'+format,sceneId_start,sceneId_end,annId_start,annId_end)
    if if_check_grasp_tsvs_real:
        check_grasp_tsvs_real(graspnet_root+'/grasp_tsvs_real/grasp_tsvs_real_'+camera+'_'+format+'_'+str(scene_sum),sceneId_start,sceneId_end,annId_start,annId_end)
    if if_check_grasp_tsvs_train:
        check_grasp_tsvs_train(graspnet_root+'/grasp_tsvs_train/grasp_tsvs_train_'+camera+'_'+format+'_'+str(scene_sum),sceneId_start,sceneId_end,annId_start,annId_end)
    if if_check_grasp_dataloader_config:
        check_grasp_dataloader_config(graspnet_root+'/grasp_dataloader_config/grasp_dataloader_config_'+camera+'_'+format+'_'+str(scene_sum))
    if if_check_else:
        check_else(graspnet_root+'/else')
    
def main():
    graspnet_root = '/mnt/msranlpintern/dataset/graspnet-v2'  #'D:/dataset/graspnet'
    camera = 'kinect'
    format = '6d'
    scene_sum = 100
    sceneId_start = 0
    sceneId_end = 100
    annId_start = 0
    annId_end = 256
    
    # ====== check dataset ======
    check_dataset(graspnet_root=graspnet_root,camera=camera,format=format,scene_sum=scene_sum,sceneId_start=sceneId_start,sceneId_end=sceneId_end,annId_start=annId_start,annId_end=annId_end,if_check_graspnet_1billion=if_check_graspnet_1billion,if_check_grasp_scene_point_clouds=if_check_grasp_scene_point_clouds,if_check_grasp_jsons=if_check_grasp_jsons,if_check_grasp_tsvs_real=if_check_grasp_tsvs_real,if_check_grasp_tsvs_train=if_check_grasp_tsvs_train,if_check_grasp_dataloader_config=if_check_grasp_dataloader_config,if_check_else=if_check_else)
    
if __name__ == "__main__":
    main()