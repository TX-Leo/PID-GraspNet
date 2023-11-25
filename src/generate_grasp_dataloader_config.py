__author__ = 'Zhi Wang'
__date__ = '2023.11.10'

import json
import os

def generate_grasp_dataloader_config(graspnet_root,camera,format,scene_sum,sceneId_start,sceneId_end,annId_start,annId_end,train_valid_proportion):
    '''    
    @Function: get grasp dataloader config for kosmos-e training
            
    @Input:
        - graspnet_root: ROOT PATH FOR GRASPNET.
        - camera: string of type of camera, "realsense" or "kinect".
        - format: string of grasp format, '6d' or 'rect'.
        - scene_sum: int of the number of scenes in the dataset.
        - sceneId_start: int of the starting scene index.(sceneId max range is [0:180])
        - sceneId_end: int of the ending scene index.
        - annId_start: int of the starting annotation index.(annId max range is [0:256])
        - annId_end: int of the ending annotation index.
        - train_valid_proportion: int(range of 0-100), the proportion of train data and valid data

    @Output: None
        - grasp_dataloder_config
            - grasp_dataloder_config_kinect_6d_100
                - json
                    - train.json
                    - valid.json
    '''
    
    # ====== get dataloader config directory ======
    grasp_dataloader_config_dir = graspnet_root + '/grasp_dataloader_config/grasp_dataloader_config_'+ camera+'_'+format+'_'+str(scene_sum)+'/json/'
    if not os.path.exists(grasp_dataloader_config_dir):
        os.makedirs(grasp_dataloader_config_dir)
    grasp_tsvs_train_dir = 'grasp_tsvs_train/grasp_tsvs_train_'+ camera+'_'+format+'_'+str(scene_sum)
    
    train_source = []
    valid_source = []
    
    # ====== for every scene ======
    for sceneId in list(range(sceneId_start, sceneId_end)):
        if sceneId < train_valid_proportion:
            # ====== for every ann of train ======
            for annId in list(range(annId_start, annId_end)):
                train_source.append(f"../../{grasp_tsvs_train_dir}"+"/{:04d}".format(sceneId)+"/{:04d}.tsv".format(annId))
        else:
            # ====== for every ann of valid ======
            for annId in list(range(annId_start, annId_end)):
                valid_source.append(f"../../{grasp_tsvs_train_dir}"+"/{:04d}".format(sceneId)+"/{:04d}.tsv".format(annId))

    # ====== train data(a list(only a dict)) =====
    train_data = [{
        "source": train_source,
        "source_lang": "graspnet",
        "weight": 1.0,
        "name": "graspnet"
    }]

    # ====== valid data(a list(only a dict)) =====
    valid_data = [{
        "source": valid_source,
        "source_lang": "graspnet",
        "weight": 1.0,
        "name": "graspnet"
    }]

    # ====== save train data in train.json ======
    train_json_file_path = grasp_dataloader_config_dir+"/train.json"
    with open(train_json_file_path, "w") as json_file:
       json.dump(train_data, json_file, indent=4)
    
    # ====== save valid data in train.json ======
    valid_json_file_path = grasp_dataloader_config_dir+"/valid.json"
    with open(valid_json_file_path, "w") as json_file:
       json.dump(valid_data, json_file, indent=4)
    
def main():
    graspnet_root = '/mnt/msranlpintern/dataset/graspnet-v2' #'D:/dataset/graspnet' # 'D:/dataset/graspnet' # /Volumes/WD/dataset/graspnet
    camera = 'kinect'
    format = '6d'
    sceneId_start = 0
    sceneId_end = 100
    annId_start = 0
    annId_end = 256
    scene_sum = 100
    train_valid_proportion = 90
    
    generate_grasp_dataloader_config(graspnet_root = graspnet_root, camera = camera , format = format , scene_sum=scene_sum, sceneId_start = sceneId_start,sceneId_end = sceneId_end,annId_start = annId_start,annId_end = annId_end,train_valid_proportion=train_valid_proportion)
    
if __name__ == "__main__":
    main()