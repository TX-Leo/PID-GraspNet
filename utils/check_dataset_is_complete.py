import os
import json

def check_graspnet_1billion(root):
    from graspnetAPI import GraspNet
    g = GraspNet(root, 'kinect', 'all')
    if g.checkDataCompleteness():
        print('Check for kinect passed')
    g = GraspNet(root, 'realsense', 'all')
    if g.checkDataCompleteness():
        print('Check for realsense passed')
        
def check_grasp_jsons(root,sceneId_start,sceneId_end,annId_start,annId_end):
    for sceneId in list(range(sceneId_start, sceneId_end)):
        # 检查场景文件夹
        scene_folder = root+'/{:04d}'.format(sceneId)
        if not os.path.isdir(scene_folder):
            print(f"{scene_folder} doesn't exit")
            continue
        # 检查all_obj_names_and_ids_dict.json文件
        all_obj_names_and_ids_dict_file = f"{scene_folder}/all_obj_names_and_ids_dict.json"
        if not os.path.isfile(all_obj_names_and_ids_dict_file):
            print(f"{all_obj_names_and_ids_dict_file} doesn't exit")
        with open(all_obj_names_and_ids_dict_file, "r") as f:
            all_obj_names_and_ids_dict = json.load(f)
        all_obj_num = len(all_obj_names_and_ids_dict_file)
        # 检查子文件夹和文件数量
        for annId in list(range(annId_start, annId_end)):
            # 检查视角文件
            subfolder = scene_folder + '/{:04d}'.format(annId)
            if not os.path.isdir(subfolder):
                print(f"{subfolder} doesn't exit")
                continue
            # 检查每个视角下的json文件数量
            json_files = [f for f in os.listdir(subfolder) if f.endswith(".json")]
            # if len(json_files) != all_obj_num*10:
            # if len(json_files) % 10 != 0:
                # print(f"the num of json files in {subfolder} is not enough!")
    print(f'==== finish checking grasp jsons ====')


def check_grasp_tsvs_real(root,sceneId_start,sceneId_end,annId_start,annId_end):
    for sceneId in list(range(sceneId_start, sceneId_end)):
        # 检查场景文件夹
        scene_folder = root+'/{:04d}'.format(sceneId)
        if not os.path.isdir(scene_folder):
            print(f"{scene_folder} doesn't exit")
            continue
        # 检查all_data.txt
        all_score_file_path = root+'/all_score.txt'
        all_width_file_path = root+'/all_width.txt'
        all_height_file_path = root+'/all_height.txt'
        all_translation_1_file_path = root+'/all_translation_1.txt'
        all_translation_2_file_path = root+'/all_translation_2.txt'
        all_translation_3_file_path = root+'/all_translation_3.txt'
        all_rotation_1_file_path = root+'/all_rotation_1.txt'
        all_rotation_2_file_path = root+'/all_rotation_2.txt'
        all_rotation_3_file_path = root+'/all_rotation_3.txt'
        if not os.path.isfile(all_score_file_path):
            print(f"{all_score_file_path} doesn't exit")
        if not os.path.isfile(all_width_file_path):
            print(f"{all_width_file_path} doesn't exit")
        if not os.path.isfile(all_height_file_path):
            print(f"{all_height_file_path} doesn't exit")
        if not os.path.isfile(all_translation_1_file_path):
            print(f"{all_translation_1_file_path} doesn't exit")
        if not os.path.isfile(all_translation_2_file_path):
            print(f"{all_translation_2_file_path} doesn't exit")
        if not os.path.isfile(all_translation_3_file_path):
            print(f"{all_translation_3_file_path} doesn't exit")     
        if not os.path.isfile(all_rotation_1_file_path):
            print(f"{all_rotation_1_file_path} doesn't exit")
        if not os.path.isfile(all_rotation_2_file_path):
            print(f"{all_rotation_2_file_path} doesn't exit")
        if not os.path.isfile(all_rotation_3_file_path):
            print(f"{all_rotation_3_file_path} doesn't exit")
        for annId in list(range(annId_start, annId_end)):
            # 检查视角文件
            subfolder = scene_folder + '/{:04d}'.format(annId)
            if not os.path.isdir(subfolder):
                print(f"{subfolder} doesn't exit")
                continue
    print(f'==== finish checking grasp tsvs real ====')

    
def check_grasp_tsvs_real_encoded(root,sceneId_start,sceneId_end,annId_start,annId_end):
    for sceneId in list(range(sceneId_start, sceneId_end)):
        # 检查场景文件夹
        scene_folder = root+'/{:04d}'.format(sceneId)
        if not os.path.isdir(scene_folder):
            print(f"{scene_folder} doesn't exit")
            continue
        # 检查all_data.txt
        all_score_file_path = root+'/all_score.txt'
        all_width_file_path = root+'/all_width.txt'
        all_height_file_path = root+'/all_height.txt'
        all_translation_1_file_path = root+'/all_translation_1.txt'
        all_translation_2_file_path = root+'/all_translation_2.txt'
        all_translation_3_file_path = root+'/all_translation_3.txt'
        all_rotation_1_file_path = root+'/all_rotation_1.txt'
        all_rotation_2_file_path = root+'/all_rotation_2.txt'
        all_rotation_3_file_path = root+'/all_rotation_3.txt'
        if not os.path.isfile(all_score_file_path):
            print(f"{all_score_file_path} doesn't exit")
        if not os.path.isfile(all_width_file_path):
            print(f"{all_width_file_path} doesn't exit")
        if not os.path.isfile(all_height_file_path):
            print(f"{all_height_file_path} doesn't exit")
        if not os.path.isfile(all_translation_1_file_path):
            print(f"{all_translation_1_file_path} doesn't exit")
        if not os.path.isfile(all_translation_2_file_path):
            print(f"{all_translation_2_file_path} doesn't exit")
        if not os.path.isfile(all_translation_3_file_path):
            print(f"{all_translation_3_file_path} doesn't exit")     
        if not os.path.isfile(all_rotation_1_file_path):
            print(f"{all_rotation_1_file_path} doesn't exit")
        if not os.path.isfile(all_rotation_2_file_path):
            print(f"{all_rotation_2_file_path} doesn't exit")
        if not os.path.isfile(all_rotation_3_file_path):
            print(f"{all_rotation_3_file_path} doesn't exit")
        for annId in list(range(annId_start, annId_end)):
            # 检查视角文件
            subfolder = scene_folder + '/{:04d}'.format(annId)
            if not os.path.isdir(subfolder):
                print(f"{subfolder} doesn't exit")
                continue
    print(f'==== finish checking grasp tsvs real encoded ====')
    
def check_grasp_tsvs_train(root,sceneId_start,sceneId_end,annId_start,annId_end):
    for sceneId in list(range(sceneId_start, sceneId_end)):
        # 检查场景文件夹
        scene_folder = root+'/{:04d}'.format(sceneId)
        if not os.path.isdir(scene_folder):
            print(f"{scene_folder} doesn't exit")
            continue
        for annId in list(range(annId_start, annId_end)):
            # 检查视角文件
            subfolder = scene_folder + '/{:04d}.tsv'.format(annId)
            if not os.path.isfile(subfolder):
                print(f"{subfolder} doesn't exit")
                continue
    print(f'==== finish checking grasp tsvs train ====')
   
def check_else(root):
    all_obj_names_and_ids_final_file = root+'/all_obj_names_and_ids_final.json'
    if not os.path.isfile(all_obj_names_and_ids_final_file):
        print(f"{all_obj_names_and_ids_final_file} doesn't exit")
    prompts_engineering_6d_file = root+'/prompts_engineering_6d.json'
    if not os.path.isfile(prompts_engineering_6d_file):
        print(f"{prompts_engineering_6d_file} doesn't exit")

def main():
    graspnet_root = 'D:/dataset/graspnet' # 'D:/dataset/graspnet' # /Volumes/WD/dataset/graspnet # r'C:\Users\v-zhiwang2\Downloads\graspnet'
    train_or_test = 'train'
    camera = 'kinect'
    format = '6d'
    sceneId_start = 0
    sceneId_end = 100
    annId_start = 0
    annId_end = 256
    scene_sum = 100
    
    check_graspnet_1billion(graspnet_root)
    check_grasp_jsons(graspnet_root + '/grasp_jsons/' + camera + '/' + format ,sceneId_start,sceneId_end,annId_start,annId_end)
    check_grasp_tsvs_real(graspnet_root + '/grasp_tsvs_real/grasp_tsvs_real_'+camera+'_'+format+'_'+str(scene_sum),sceneId_start,sceneId_end,annId_start,annId_end)
    check_grasp_tsvs_real_encoded(r'C:\Users\v-zhiwang2\Downloads\graspnet' + '/grasp_tsvs_real_encoded/grasp_tsvs_real_encoded_'+camera+'_'+format+'_'+str(scene_sum),sceneId_start,sceneId_end,annId_start,annId_end)
    check_grasp_tsvs_train(graspnet_root + '/grasp_tsvs_train/grasp_tsvs_train_'+camera+'_'+format+'_'+str(scene_sum),sceneId_start,sceneId_end,annId_start,annId_end)
    check_else(graspnet_root+'/else')
    
    
    
if __name__ == "__main__":
    main()