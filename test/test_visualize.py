'''可视化数据集的脚本，主要有四种
1.Show grasp labels on a object
2.Show 6D poses of objects in a scene.
3.Show Rectangle grasp labels in a scene.
4.Show 6D grasp labels in a scene.
'''
__author__ = 'mhgou'
__version__ = '1.0'

# GraspNetAPI example for visualization.
# change the graspnet_root path

####################################################################
graspnet_root = 'D:/dataset/graspnet' # ROOT PATH FOR GRASPNET
####################################################################

from graspnetAPI import GraspNet
import numpy as np
import open3d as o3d
import os

# initialize a GraspNet instance  
g = GraspNet(graspnet_root, camera='kinect', split='train')

# # ======================== 1.show object grasps======================== 
# g.showObjGrasp(objIds = 0, show=True)

# # ======================== 2.show 6d poses======================== 
# g.show6DPose(sceneIds = 0, show = True)
# from graspnetAPI.utils.utils import generate_scene_model, generate_scene_pointcloud
# graspnet_root = 'D:/dataset/graspnet' 
# camera = 'kinect'
# scene_id = 0
# scene_name = 'scene_'+str(scene_id).zfill(4)
# anno_idx = 0
# align_to_table=True
# save_folder = r"D:\dataset\graspnet\scenes\scene_0000\kinect/"

# model_list, obj_list, pose_list = generate_scene_model(graspnet_root, scene_name, anno_idx, return_poses=True, align=align_to_table, camera=camera)
# point_cloud = generate_scene_pointcloud(graspnet_root, scene_name, anno_idx, align=align_to_table, camera=camera)
# point_cloud = point_cloud.voxel_down_sample(voxel_size=0.005)
# # 将所有物体的点云添加到整个场景点云中
# for model in model_list:
#     point_cloud += model
# # 保存整个场景点云
# scene_point_cloud_filename = os.path.join(save_folder, '{}_{}_scene_point_cloud.pcd'.format(scene_name, camera))
# o3d.io.write_point_cloud(scene_point_cloud_filename, point_cloud)
# o3d.visualization.draw_geometries([point_cloud, *model_list])

# # ========================  3.show scene rectangle grasps======================== 
# g.showSceneGrasp(sceneId = 0, camera = 'realsense', annId = 0, format = 'rect', numGrasp = 20)

# ======================== 4.show scene 6d grasps(You may need to wait several minutes)======================== 
# g.showSceneGrasp(sceneId = 4, camera = 'kinect', annId = 2, format = '6d')

# sceneId = 0
# camera = 'kinect'
# annId = 0
# format = '6d'
# numGrasp = 20
# show_object = True
# coef_fric_thresh = 0.1

# geometries = []
# sceneGrasp = g.loadGrasp(sceneId = sceneId, annId = annId, camera = camera, format = '6d', fric_coef_thresh = coef_fric_thresh)
# sceneGrasp = sceneGrasp.random_sample(numGrasp = numGrasp)
# scenePCD = g.loadScenePointCloud(sceneId = sceneId, camera = camera, annId = annId, align = False)
# geometries.append(scenePCD)
# # geometries += sceneGrasp.to_open3d_geometry_list()
# if show_object:
#     objectPCD = g.loadSceneModel(sceneId = sceneId, camera = camera, annId = annId, align = False)
#     geometries += objectPCD
# # o3d.visualization.draw_geometries(geometries)
# # 合并所有点云对象
# merged_cloud = o3d.geometry.PointCloud()
# for point_cloud in geometries:
#     merged_cloud += point_cloud

# o3d.io.write_point_cloud("D:\dataset\graspnet\scenes\scene_0000\kinect\point_cloud\point_cloud.pcd", merged_cloud)

# # 提取点云数据并保存为NumPy数组
# points = np.asarray(merged_cloud.points)
# print(len(points))

# # 保存为.npy文件
# np.save("D:\dataset\graspnet\scenes\scene_0000\kinect\point_cloud\point_cloud.npy", points)

# ====================================== 5.show some specific 6d grasp in one scene ==================================================
# geometry = []
# geometry.append(g.loadScenePointCloud(sceneId, camera, annId))
# geometry.append(grasp.to_open3d_geometry())
# o3d.visualization.draw_geometries(geometry)

# ====================================== 5.show some specific rect grasp in one scene ==================================================
# g = GraspNet(graspnet_root, camera = camera, split = 'all')
# bgr = g.loadBGR(sceneId = sceneId, camera = camera, annId = annId)
# rect_grasp = rect_grasp_group.random_sample(1)[0]
# img = rect_grasp.to_opencv_image(bgr)
# cv2.imshow('rect grasp', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#=============================================== 6.show some 6d Grasp randomly in 6d format and visulize the result ========================================
# # load grasps of scene 1 with annotation id = 3, camera = kinect and fric_coef_thresh = 0.2
# _6d_grasp = g.loadGrasp(sceneId = 0, annId = 0, format = '6d', camera = 'kinect', fric_coef_thresh = 0.2)
# # print('6d grasp:\n{}'.format(_6d_grasp))

# # visualize the grasps using open3d
# geometries = []
# geometries.append(g.loadScenePointCloud(sceneId = 0, annId = 0, camera = 'kinect'))
# geometries += _6d_grasp.random_sample(numGrasp = 100).to_open3d_geometry_list()
# o3d.visualization.draw_geometries(geometries)

#=============================================== 7.show some rect Grasp randomly in 6d format and visulize the result ========================================
# load rectangle grasps of scene 1 with annotation id = 3, camera = realsense and fric_coef_thresh = 0.2
rect_grasp = g.loadGrasp(sceneId = 0, annId = 0, format = 'rect', camera = 'realsense', fric_coef_thresh = 0.2)
print('rectangle grasp:\n{}'.format(rect_grasp))

# visualize the rectanglegrasps using opencv
bgr = g.loadBGR(sceneId = 0, annId = 0, camera = 'realsense')
img = rect_grasp.to_opencv_image(bgr, numGrasp = 20)
cv2.imshow('rectangle grasps', img)
cv2.waitKey(0)
cv2.destroyAllWindows()