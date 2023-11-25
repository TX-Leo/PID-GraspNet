__author__ = 'Zhi Wang'
__date__ = '2023.11.10'

import os
import json
import xml.etree.ElementTree as ET
import open3d as o3d
import cv2

from graspnetAPI import GraspNet,GraspGroup,RectGraspGroup

def generate_grasp_jsons_for_one_ann_in_one_scene_rect(g,sceneId,annId,camera,format,grasp_info_for_one_ann_in_one_scene_path,all_obj_names_and_ids_dict,fric_coef_thresh,show):
    '''
    @Function: get the top-10 rect grasp info for each object in a scene for an ann, and save them in json files.
        
        - rect grasp info:
            - rbg_img_path
            - depth_img_path
            - all_obj_names_and_ids_dict
            - object_id
            - object_name
            - score
            - height
            - center_point(2)
            - open_point(2)
            
    @Input:
        - g: a grapsing generator.
        - sceneId: int of the scene index.
        - annId: int of the annotation index.
        - camera: string of type of camera, "realsense" or "kinect".
        - format: string of grasp format, '6d' or 'rect'.
        - grasp_info_for_one_ann_in_one_scene_path: grasp_json file path in a scene for an ann.
        - all_obj_names_and_ids_dict: dict of all objects in the scene.
        - fric_coef_thresh: float of the frcition coefficient threshold of the grasp. 
        - show: bool of whether to show the scene and grasp.

    @Output: None
    '''
    
    rect_grasp_group = g.loadGrasp(sceneId = sceneId, format = format, annId = annId, camera = camera, fric_coef_thresh = fric_coef_thresh)
    # print(f'====== The number of rect_grasp_group: {len(rect_grasp_group)} ======')
    
    # ====== visualization ======
    if show:
        show_rect_grasp_group = RectGraspGroup()
        bgr = g.loadBGR(sceneId = sceneId, camera = camera, annId = annId)
    
    # ====== select top-10 rect grasp for every object ======
    id_data_dict = {}
    for data in rect_grasp_group:
        if data.object_id not in id_data_dict:
            id_data_dict[data.object_id] = []
        id_data_dict[data.object_id].append(data)
    for id, data_list in id_data_dict.items():
        sorted_data = sorted(data_list, key=lambda x: x.score, reverse=True)
        selected_data = sorted_data[:10]
        selected_rect_grasp_group = RectGraspGroup()
        for temp in selected_data:
            selected_rect_grasp_group.add(temp)
            # ====== add top-10 rect grasp for every object to the visualization ======
            if show:
                show_rect_grasp_group.add(temp)
            
        # ====== save top-10 rect grasp info in json file ======
        for i in range(len(selected_rect_grasp_group)):
            grasp_rect = selected_rect_grasp_group[i]
            object_name = all_obj_names_and_ids_dict[str(grasp_rect.object_id)]
            json_path = grasp_info_for_one_ann_in_one_scene_path + '/grasp_'+ '{:02d}'.format(grasp_rect.object_id) + '_{:02d}.json'.format(i)
            with open(json_path, "w") as json_file:
                grasp_data = {   
                    "rbg_img_path":'/scenes/scene_' +'{:04d}'.format(sceneId) + '/' + camera + '/rgb'+ '/{:04}'.format(annId)+'.png',
                    "depth_img_path":'/scenes/scene_' +'{:04d}'.format(sceneId) + '/' + camera + '/depth'+ '/{:04}'.format(annId)+'.png',
                    "all_obj_names_and_ids_dict":all_obj_names_and_ids_dict, 
                    "object_id":str(grasp_rect.object_id),
                    "object_name":object_name,
                    "score":grasp_rect.score,
                    "height":float(grasp_rect.height), # float32 --> float
                    "center_point":tuple(float(x) for x in grasp_rect.center_point), # float32 --> float
                    "open_point":tuple(float(x) for x in grasp_rect.open_point) # float32 --> float
                    }
                json.dump(grasp_data, json_file,indent=4)
                # print(f'====== grasp_data: {grasp_data} ======')
            # print(f'====== [scene_{sceneId},ann_{annId},id_{grasp_rect.object_id},NO_{i}] {camera}-{format}-grasp info is saved successfully! ======')
    print(f'====== [scene_{sceneId},ann_{annId}] {camera}-{format} grasp info is saved successfully! ======')

    # ====== visualization ======
    if show:
        img = show_rect_grasp_group.to_opencv_image(bgr)
        cv2.imshow('rect grasp', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    

def generate_grasp_jsons_for_one_ann_in_one_scene_6d(g,sceneId,annId,camera,format,grasp_info_for_one_ann_in_one_scene_path,all_obj_names_and_ids_dict,fric_coef_thresh,show):
    '''
    @Function: get the top-10 6d grasp info for each object in a scene for an ann, and save them in json files.
        
        - 6d grasp info:
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
            
    @Input:
        - g: a grapsing generator.
        - sceneId: int of the scene index.
        - annId: int of the annotation index.
        - camera: string of type of camera, "realsense" or "kinect".
        - format: string of grasp format, '6d' or 'rect'.
        - grasp_info_for_one_ann_in_one_scene_path: grasp_json file path in a scene for an ann.
        - all_obj_names_and_ids_dict: dict of all objects in the scene.
        - fric_coef_thresh: float of the frcition coefficient threshold of the grasp. 
        - show: bool of whether to show the scene and grasp.

    @Output: None
    '''
    
    _6d_grasp_group = g.loadGrasp(sceneId = sceneId, annId = annId, format = format, camera = camera, fric_coef_thresh = fric_coef_thresh) 
    # print(f'====== The number of _6d_grasp_group: {len(_6d_grasp_group)} ======')
    
    # ====== nms for reducing the grasp number ======
    _6d_grasp_group = _6d_grasp_group.nms(translation_thresh = 0.05, rotation_thresh = 20 / 180.0 * 3.1416)
    # print(f'====== After nms, the number of _6d_grasp_group: {len(_6d_grasp_group)} ======')
    
    # ====== add the point cloud of this scene to the visualization ======
    if show:
        geometry = []
        geometry.append(g.loadScenePointCloud(sceneId, camera, annId))
    
    # ====== select top-10 6d grasp for every object ======
    id_data_dict = {}
    for data in _6d_grasp_group:
        if data.object_id not in id_data_dict:
            id_data_dict[data.object_id] = []
        id_data_dict[data.object_id].append(data)
    for id, data_list in id_data_dict.items():
        sorted_data = sorted(data_list, key=lambda x: x.score, reverse=True)
        selected_data = sorted_data[:10]
        selected_6d_grasp_group = GraspGroup()
        for temp in selected_data:
            selected_6d_grasp_group.add(temp)

        # ====== save top-10 6d grasp info in json file ======
        for i in range(len(selected_6d_grasp_group)):
            grasp_6d = selected_6d_grasp_group[i]
            # ====== add top-10 6d grasp for every object to the visualization ======
            if show:
                geometry.append(grasp_6d.to_open3d_geometry())
            object_name = all_obj_names_and_ids_dict[str(grasp_6d.object_id)]
            json_path = grasp_info_for_one_ann_in_one_scene_path + '/grasp_'+ '{:02d}'.format(grasp_6d.object_id) + '_{:02d}.json'.format(i)
            with open(json_path, "w") as json_file:
                grasp_data = {   
                    "rbg_img_path":'/scenes/scene_' +'{:04d}'.format(sceneId) + '/' + camera + '/rgb'+ '/{:04}'.format(annId)+'.png',
                    "depth_img_path":'/scenes/scene_' +'{:04d}'.format(sceneId) + '/' + camera + '/depth'+ '/{:04}'.format(annId)+'.png',
                    "all_obj_names_and_ids_dict":all_obj_names_and_ids_dict, 
                    "object_id":str(grasp_6d.object_id),
                    "object_name":object_name,
                    "score":grasp_6d.score,
                    "width":grasp_6d.width,
                    "height":float(grasp_6d.height), # float32 --> float
                    "depth":float(grasp_6d.depth), # float32 --> float
                    "translation":grasp_6d.translation.tolist(), # ndarray --> list
                    "rotation_matrix":grasp_6d.rotation_matrix.tolist() # ndarray --> list
                    }
                json.dump(grasp_data, json_file,indent=4)
                # print(f'====== grasp_data: {grasp_data} ======')
            # print(f'====== [scene_{sceneId},ann_{annId},id_{grasp_6d.object_id},NO_{i}] {camera}-{format}-grasp info is saved successfully! ======')
    print(f'====== [scene_{sceneId},ann_{annId}] {camera}-{format} grasp info is saved successfully! ======')

    # ====== visualization ======
    if show:
        o3d.visualization.draw_geometries(geometry)   


def generate_grasp_jsons(graspnet_root,camera,format,sceneId_start,sceneId_end,annId_start,annId_end,fric_coef_thresh,show):
    '''    
    @Function: get all grasp info for the specified annotations in the specified scenes(top-10 for each object), and save them in json files.
            
    @Input:
        - graspnet_root: ROOT PATH FOR GRASPNET
        - camera: string of type of camera, "realsense" or "kinect".
        - format: string of grasp format, '6d' or 'rect'.
        - sceneId_start: int of the starting scene index.(sceneId max range is [0:180])
        - sceneId_end: int of the ending scene index.
        - annId_start: int of the starting annotation index.(annId max range is [0:256])
        - annId_end: int of the ending annotation index.
        - fric_coef_thresh: float of the frcition coefficient threshold of the grasp. 
        - show: bool of whether to show the scene and grasp.

    @Output: None
        - grasp_jsons
            - kinect
                - 6d
                    - 0000
                    - 0001
                        - 0000
                        - 0001
                            - grasp_05_00.json
                            - grasp_05_01.json

                            ....
                            - grasp_05_09.json
                        ....
                        - 0255
                    ....
                    - 0255
                - rect
    '''
    
    # ====== create a grapsing generator ======
    g = GraspNet(graspnet_root, camera, split='train')

    # ====== for every scene ======
    for sceneId in list(range(sceneId_start, sceneId_end)):
        grasp_info_for_one_scene_path = graspnet_root + '/grasp_jsons/' + camera + '/' + format + '/{:04d}'.format(sceneId)
        if not os.path.exists(grasp_info_for_one_scene_path):
            os.makedirs(grasp_info_for_one_scene_path)
        # ====== get and save all_obj_names_and_ids_dict for one scene ======
        with open(graspnet_root + '/else/all_obj_names_and_ids_final.json', 'r') as file:
            json_data = file.read()
        all_obj_names_and_ids_dict = json.loads(json_data)
        xml_path = graspnet_root + '/scenes/scene_' +'{:04d}'.format(sceneId)+'/'+camera+'/annotations/0000.xml'
        tree = ET.parse(xml_path)
        root = tree.getroot()
        obj_ids = []
        for obj in root.findall('obj'):
            obj_ids.append(obj.find('obj_id').text)
        all_obj_names_and_ids_dict =  {key: all_obj_names_and_ids_dict[key] for key in obj_ids if key in all_obj_names_and_ids_dict}
        with open(grasp_info_for_one_scene_path + '/all_obj_names_and_ids_dict.json', "w") as json_file:
            json.dump(all_obj_names_and_ids_dict, json_file,indent=4)
        print(f'====== For scene_{sceneId}, all_obj_names_and_ids_dict: ======\n{all_obj_names_and_ids_dict}')
            
        # ====== for every ann ======
        for annId in list(range(annId_start, annId_end)):
            grasp_info_for_one_ann_in_one_scene_path = grasp_info_for_one_scene_path + '/{:04d}'.format(annId)
            if not os.path.exists(grasp_info_for_one_ann_in_one_scene_path):
                os.makedirs(grasp_info_for_one_ann_in_one_scene_path)
            # ====== get all grasp info for this ann in this scene, and save them in json files ======
            if format == '6d':
                generate_grasp_jsons_for_one_ann_in_one_scene_6d(g = g,sceneId = sceneId, annId = annId, camera = camera, format = format, grasp_info_for_one_ann_in_one_scene_path = grasp_info_for_one_ann_in_one_scene_path,all_obj_names_and_ids_dict=all_obj_names_and_ids_dict,fric_coef_thresh=fric_coef_thresh,show=show)
            elif format == 'rect':
                generate_grasp_jsons_for_one_ann_in_one_scene_rect(g = g,sceneId = sceneId, annId = annId, camera = camera, format = format, grasp_info_for_one_ann_in_one_scene_path = grasp_info_for_one_ann_in_one_scene_path,all_obj_names_and_ids_dict=all_obj_names_and_ids_dict,fric_coef_thresh=fric_coef_thresh,show=show)
            else:
                print(f'====== !!!FORMAT ERROR!!! ======')
            
def main():
    graspnet_root = r'C:\Users\v-zhiwang2\Downloads\graspnet-v2' #'D:/dataset/graspnet' #'/mnt/msranlpintern/dataset/graspnet-v2' #
    camera = 'kinect'
    format = '6d'
    sceneId_start = 100
    sceneId_end = 190
    annId_start = 0
    annId_end = 256
    fric_coef_thresh=0.2
    show = False
    
    # generate_grasp_jsons
    generate_grasp_jsons(graspnet_root = graspnet_root , camera = camera , format = format , sceneId_start = sceneId_start,sceneId_end = sceneId_start + 1,annId_start = annId_start,annId_end = 256,fric_coef_thresh = fric_coef_thresh,show=show)
    generate_grasp_jsons(graspnet_root = graspnet_root , camera = camera , format = format , sceneId_start = sceneId_start + 1,sceneId_end = sceneId_end,annId_start = 0,annId_end = annId_end,fric_coef_thresh = fric_coef_thresh,show=show)

    
if __name__ == '__main__':
    main()