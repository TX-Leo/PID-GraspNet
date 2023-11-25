__author__ = 'Zhi Wang'
__date__ = '2023.11.10'

import numpy as np
import open3d as o3d
import os

from graspnetAPI.utils.utils import generate_scene_model, generate_scene_pointcloud

def generate_grasp_scene_point_clouds(graspnet_root,camera,sceneId_start,sceneId_end,align_to_table=True,show=False,voxel_size=0.005,if_save_pcd=True,if_save_npy_points=False,if_save_npy_points_and_colors=False):
    '''    
    @Function: generate point clouds for every scene
        - type of point cloud 
            - pcd: xyz and rgb
            - npy(points): xyz
            - npy(points and colors): xyz and rgb
            
    @Input:
        - graspnet_root: ROOT PATH FOR GRASPNET
        - camera: string of type of camera, "realsense" or "kinect".
        - sceneId_start: int of the starting scene index.(sceneId max range is [0:180])
        - sceneId_end: int of the ending scene index.
        - align_to_table: bool, transform to table coordinates if set to True
        - show: bool of whether to show the scene and grasp.
        - voxel_size: float, the size of voxel(down sample)
        - if_save_pcd: bool, save point clouds in pcd files if set to True
        - if_save_npy_points: bool, save point clouds(only points) in npy files if set to True
        - if_save_npy_points_and_colors: bool, save point clouds(points and colors) in npy files if set to True

    @Output: None
        - grasp_scene_point_clouds
            - kinect
                - pcd
                - npy_points
                - npy_points_and_colors
                    - 0000.npy
                    - 0001.npy

                    ....
                    - 0099.npy
    '''

    point_clouds_save_dir = graspnet_root + '/grasp_scene_point_clouds/' + camera     

    # ====== for every scene ======
    for sceneId in list(range(sceneId_start, sceneId_end)):
        if if_save_pcd:
            pcd_save_dir = point_clouds_save_dir + '/pcd/'
            if not os.path.exists(pcd_save_dir):
                os.makedirs(pcd_save_dir)
        if if_save_npy_points:
            npy_points_save_dir = point_clouds_save_dir + '/npy_points/'
            if not os.path.exists(npy_points_save_dir):
                os.makedirs(npy_points_save_dir)
        if if_save_npy_points_and_colors:
            npy_points_and_colors_save_dir = point_clouds_save_dir + '/npy_points_and_colors/'
            if not os.path.exists(npy_points_and_colors_save_dir):
                os.makedirs(npy_points_and_colors_save_dir)

        # ====== add point cloud of a scene ======
        point_cloud = generate_scene_pointcloud(graspnet_root, scene_name='scene_'+str(sceneId).zfill(4), anno_idx=0, align=align_to_table, camera=camera)
        point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
        
        # ====== add point clouds of all objects in a scene ======
        model_list, obj_list, pose_list = generate_scene_model(graspnet_root, scene_name='scene_'+str(sceneId).zfill(4), anno_idx=0, return_poses=True, align=align_to_table, camera=camera)
        for model in model_list:
            point_cloud += model
        
        # ====== visualization ======
        if show:
            o3d.visualization.draw_geometries([point_cloud, *model_list])
        
        # ====== save point clouds in .pcd file ======
        if if_save_pcd:
            point_cloud_for_one_scene =  pcd_save_dir + '/{:04d}.pcd'.format(sceneId)
            o3d.io.write_point_cloud(point_cloud_for_one_scene, point_cloud)
            # print(f'================[scene_{sceneId}}, type: pcd] point cloud is generated successfully!!!================')
        
        # ====== save point clouds in .npy file (only points) ======
        if if_save_npy_points:
            point_cloud_for_one_scene =  npy_points_save_dir + '/{:04d}.npy'.format(sceneId)
            np.save(point_cloud_for_one_scene, np.asarray(point_cloud.points))
            # print(f'================[scene_{sceneId}}, type: npy(points)] point cloud is generated successfully!!!================')
            
        # ====== save point clouds in .npy file (points and colors) ======
        if if_save_npy_points_and_colors:
            point_cloud_for_one_scene =  npy_points_and_colors_save_dir + '/{:04d}.npy'.format(sceneId)
            np.save(point_cloud_for_one_scene, {'points': np.asarray(point_cloud.points),'colors': np.asarray(point_cloud.colors)})
            # print(f'================[scene_{sceneId}}, type: npy(points and colors)] point cloud is generated successfully!!!================')
        
        print(f'================= [scene_{sceneId}] point cloud is saved successfully!!! There are {len(point_cloud.points)} points in total!!!! =============')

def main():
    graspnet_root = '/mnt/msranlpintern/dataset/graspnet-v2' # 'D:/dataset/graspnet'
    camera = 'kinect'
    sceneId_start = 0
    sceneId_end = 100
    align_to_table = True
    show = False
    voxel_size = 0.005
    if_save_pcd = False
    if_save_npy_points = True
    if_save_npy_points_and_colors = False
    
    generate_grasp_scene_point_clouds(graspnet_root=graspnet_root,camera=camera,sceneId_start=sceneId_start,sceneId_end=sceneId_end,align_to_table=align_to_table,show=show,voxel_size=voxel_size,if_save_pcd=if_save_pcd,if_save_npy_points=if_save_npy_points,if_save_npy_points_and_colors=if_save_npy_points_and_colors)
    
if __name__ == "__main__":
    main()