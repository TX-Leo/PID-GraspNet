import numpy as np

from graspnetAPI import GraspNetEval

graspnet_root = r'D:\dataset\graspnet' # ROOT PATH FOR GRASPNET
camera = 'kinect'   
format = '6d'
scene_sum = 100
split = 'test'
specified_sceneId = 0

grasp_npys_eval_folder = r'C:\Users\v-zhiwang2\Downloads\graspnet-v2\grasp_npys_eval\grasp_npys_eval_kniect_6d_100'

# ====== Get GraspNetEval instances. ======
ge = GraspNetEval(root = graspnet_root, camera = camera, split = split)

# ====== eval a single scene ======
acc = ge.eval_scene(scene_id = specified_sceneId, dump_folder = grasp_npys_eval_folder,vis=True)
np_acc = np.array(acc)

print('\nEvaluating scene:{}, camera:{}'.format(specified_sceneId, camera))
print('mean accuracy:{}'.format(np.mean(np_acc)))