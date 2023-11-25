####################################################################
graspnet_root ='D:/dataset/graspnet' # ROOT PATH FOR GRASPNET
####################################################################

from graspnetAPI import GraspNet

sceneId = 1
annId = 1

# initialize a GraspNet instance  
g = GraspNet(graspnet_root, camera='kinect', split='train')

# show object grasps
# g.showObjGrasp(objIds = 0, show=True)

# load grasps of scene 1 with annotation id = 3, camera = kinect and fric_coef_thresh = 0.2
_6d_grasp = g.loadGrasp(sceneId = sceneId, annId = annId, format = '6d', camera = 'kinect', fric_coef_thresh = 0.2)
# print('6d grasp:\n{}'.format(_6d_grasp))

# index
grasp = _6d_grasp[0]
print(f'depth:{grasp.depth}')
grasp = _6d_grasp[100]
print(f'depth:{grasp.depth}')
grasp = _6d_grasp[1000]
print(f'depth:{grasp.depth}')

sceneId = 10
annId = 1

# initialize a GraspNet instance  
g = GraspNet(graspnet_root, camera='kinect', split='train')

# show object grasps
# g.showObjGrasp(objIds = 0, show=True)

# load grasps of scene 1 with annotation id = 3, camera = kinect and fric_coef_thresh = 0.2
_6d_grasp = g.loadGrasp(sceneId = sceneId, annId = annId, format = '6d', camera = 'kinect', fric_coef_thresh = 0.2)
# print('6d grasp:\n{}'.format(_6d_grasp))

# index
grasp = _6d_grasp[0]
print(f'depth:{grasp.depth}')
grasp = _6d_grasp[100]
print(f'depth:{grasp.depth}')
grasp = _6d_grasp[1000]
print(f'depth:{grasp.depth}')