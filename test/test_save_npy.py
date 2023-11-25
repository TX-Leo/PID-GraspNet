from graspnetAPI import GraspGroup
import numpy as np

import transforms3d as tfs

grasp_array_list_for_one_ann_in_one_scene = []

translation = np.array([-0.1202927902340889,-0.031995952129364014,0.4719941318035126])
rotation_matrix = tfs.euler.euler2mat(2.5250257012718538,-1.3783649702436835,2.5222931445812367, 'sxyz')
object_id = '5'
grasp_array1 = np.concatenate([np.array((1.0, 0.0693683847784996, 0.019999999552965164, 0.009999999776482582)),rotation_matrix.reshape(-1), translation, np.array((object_id)).reshape(-1)]).astype(np.float64)#.tolist()

translation = np.array([-0.12755820155143738,0.007025174796581268,0.4731793701648712])
rotation_matrix = tfs.euler.euler2mat(-2.791554346448947,-0.7584211088766268,-1.3491289328027547, 'sxyz')
object_id = '5'
grasp_array2 = np.concatenate([np.array((1.0, 0.052317723631858826,0.019999999552965164,0.019999999552965164)),rotation_matrix.reshape(-1), translation, np.array((object_id)).reshape(-1)]).astype(np.float64)#.tolist()
	
translation = np.array([-0.12755820155143738,0.007025174796581268,0.4731793701648712])
rotation_matrix = tfs.euler.euler2mat(-1.157266218757188,-1.096550984922682,0.27948516800167617, 'sxyz')
object_id = '5'
grasp_array3 = np.concatenate([np.array((1.0, 0.05136888101696968,0.019999999552965164,0.019999999552965164)),rotation_matrix.reshape(-1), translation, np.array((object_id)).reshape(-1)]).astype(np.float64)#.tolist()


grasp_array_list_for_one_ann_in_one_scene.append(grasp_array1)
grasp_array_list_for_one_ann_in_one_scene.append(grasp_array2)
grasp_array_list_for_one_ann_in_one_scene.append(grasp_array3)


grasp_npys_eval_for_one_ann_in_one_scene_path = r'C:\Users\v-zhiwang2\Downloads\graspnet-v2\grasp_npys_eval\grasp_npys_eval_kniect_6d_100\scene_0000\kinect'
gg=GraspGroup(np.array(grasp_array_list_for_one_ann_in_one_scene))
for i in range(256):
    gg.save_npy(grasp_npys_eval_for_one_ann_in_one_scene_path+'/{:04d}.npy'.format(i))


