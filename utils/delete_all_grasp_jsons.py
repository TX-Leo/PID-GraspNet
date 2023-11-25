import os
import shutil
def delete_graspInfo(graspnet_root,camera,sceneId_start,sceneId_end,annId_start,annId_end):
    '''
        删除指定场景、指定视角下的所有已保存的抓取信息
    '''
    
    # 遍历场景和视角
    for sceneId in list(range(sceneId_start, sceneId_end)):
        grasp_info_for_one_scene_path = graspnet_root + '/scenes/scene_' +'{:04d}'.format(sceneId)+'/'+camera+'/graspInfo'
        # 遍历视角
        for annId in list(range(annId_start, annId_end)):
            grasp_info_for_one_ann_in_one_scene_path = grasp_info_for_one_scene_path + '/{:03d}'.format(annId)
            # 如果存在就删除
            if os.path.exists(grasp_info_for_one_ann_in_one_scene_path):
                shutil.rmtree(grasp_info_for_one_ann_in_one_scene_path)
                print(f"'{grasp_info_for_one_ann_in_one_scene_path}' 文件夹已经被删除")  
        if os.path.exists(grasp_info_for_one_scene_path):
            shutil.rmtree(grasp_info_for_one_scene_path)
            print(f"'{grasp_info_for_one_scene_path}' 文件夹已经被删除")  
def main():
    delete_graspInfo(graspnet_root = 'D:/dataset/graspnet',camera = 'kinect',sceneId_start = 0,sceneId_end = 100,annId_start = 0,annId_end = 256)

def test():
    file_path = r'D:\dataset\graspnet\grasp_jsons\kinect\rect'
    if os.path.exists(file_path):
        shutil.rmtree(file_path)
        print(f"'{file_path}' 文件夹已经被删除")  

if __name__ == '__main__':
    # main()
    test()