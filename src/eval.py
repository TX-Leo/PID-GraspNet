__author__ = 'Zhi Wang'
__date__ = '2023.11.10'

import numpy as np

from graspnetAPI import GraspNetEval

def eval_a_single_ann(graspnet_root,grasp_npys_eval_folder,camera,split,specified_sceneId,specified_annId):
    '''
    @Function: eval a single ann
            
    @Input:
        - graspnet_root: ROOT PATH FOR GRASPNET
        - grasp_npys_eval_folder: string of the folder that saves the npy files.
        - camera: string of type of camera, "realsense" or "kinect".
        - split: string of the date split.
        - specified_sceneId: int, the scene index
        - specified_annId: int, the ann index

    @Output: None
    '''
    # ====== Get GraspNetEval instances. ======
    ge = GraspNetEval(root = graspnet_root, camera = camera, split = split)

    # ====== eval a single scene ======
    acc = ge.eval_ann(scene_id = specified_sceneId,ann_id=specified_annId,dump_folder = grasp_npys_eval_folder)
    np_acc = np.array(acc)

    print('\nEvaluating scene:{}, ann:{}, camera:{}'.format(specified_sceneId, specified_annId, camera))
    print('mean accuracy:{}'.format(np.mean(np_acc)))

def eval_a_single_scene(graspnet_root,grasp_npys_eval_folder,camera,split,specified_sceneId):
    '''
    @Function: eval a single scene
            
    @Input:
        - graspnet_root: ROOT PATH FOR GRASPNET
        - grasp_npys_eval_folder: string of the folder that saves the npy files.
        - camera: string of type of camera, "realsense" or "kinect".
        - split: string of the date split.
        - specified_sceneId: int, the scene index

    @Output: None
    '''
    
    # ====== Get GraspNetEval instances. ======
    ge = GraspNetEval(root = graspnet_root, camera = camera, split = split)

    # ====== eval a single scene ======
    acc = ge.eval_scene(scene_id = specified_sceneId, dump_folder = grasp_npys_eval_folder)
    np_acc = np.array(acc)

    print('\nEvaluating scene:{}, camera:{}'.format(specified_sceneId, camera))
    print('mean accuracy:{}'.format(np.mean(np_acc)))
    
def eval_scenes(graspnet_root,grasp_npys_eval_folder,camera,split,sceneId_start,sceneId_end,proc):
    '''
    @Function: eval a list of int of scene index.
            
    @Input:
        - graspnet_root: ROOT PATH FOR GRASPNET
        - grasp_npys_eval_folder: string of the folder that saves the npy files.
        - camera: string of type of camera, "realsense" or "kinect".
        - split: string of the date split.
        - sceneId_start: int of the starting scene index.(sceneId max range is [0:180])
        - sceneId_end: int of the ending scene index.
        - proc: int of the number of processes to use to evaluate.

    @Output: None
    '''
    
    # ====== Get GraspNetEval instances. ======
    ge = GraspNetEval(root = graspnet_root, camera = camera, split = split)
    
    # ====== eval scenes ======
    res = np.array(ge.parallel_eval_scenes(scene_ids = list(range(sceneId_start, sceneId_end)), dump_folder = grasp_npys_eval_folder, proc = proc))
    ap = np.mean(res)
    
    print('\nEvaluation Result:\n----------\n{}, AP Seen={}'.format(camera, ap))

def eval_train_dataset(graspnet_root,grasp_npys_eval_folder,camera,split,proc):
    '''
    @Function: eval the scenes in train dataset(scene_0000 - scene_0090)
            
    @Input:
        - graspnet_root: ROOT PATH FOR GRASPNET
        - grasp_npys_eval_folder: string of the folder that saves the npy files.
        - camera: string of type of camera, "realsense" or "kinect".
        - split: string of the date split.
        - proc: int of the number of processes to use to evaluate.

    @Output: None
    '''
    
    # ====== Get GraspNetEval instances. ======
    ge = GraspNetEval(root = graspnet_root, camera = camera, split = split)
    
    # ====== eval scenes ======
    res = np.array(ge.parallel_eval_scenes(scene_ids = list(range(0, 90)), dump_folder = grasp_npys_eval_folder, proc = proc))
    ap = np.mean(res)
    
    print('\nEvaluation Result:\n----------\n{}, AP Seen={}'.format(camera, ap))

def eval_valid_dataset(graspnet_root,grasp_npys_eval_folder,camera,split,proc):
    '''
    @Function: eval the scenes in valid dataset(scene_0090 - scene_0100)
            
    @Input:
        - graspnet_root: ROOT PATH FOR GRASPNET
        - grasp_npys_eval_folder: string of the folder that saves the npy files.
        - camera: string of type of camera, "realsense" or "kinect".
        - split: string of the date split.
        - proc: int of the number of processes to use to evaluate.

    @Output: None
    '''
    
    # ====== Get GraspNetEval instances. ======
    ge = GraspNetEval(root = graspnet_root, camera = camera, split = split)
    
    # ====== eval scenes ======
    res = np.array(ge.parallel_eval_scenes(scene_ids = list(range(90, 100)), dump_folder = grasp_npys_eval_folder, proc = proc))
    ap = np.mean(res)
    
    print('\nEvaluation Result:\n----------\n{}, AP Seen={}'.format(camera, ap))

def eval_test_dataset(graspnet_root,grasp_npys_eval_folder,camera,split,proc):
    '''
    @Function: eval the scenes in test dataset(scene_0100 - scene_0190)
        - seen: scene_0100 - scene_0130
        - similar: scene_0130 - scene_0160
        - novel: scene_0160 - scene_0190
        
    @Input:
        - graspnet_root: ROOT PATH FOR GRASPNET
        - grasp_npys_eval_folder: string of the folder that saves the npy files.
        - camera: string of type of camera, "realsense" or "kinect".
        - split: string of the date split.
        - proc: int of the number of processes to use to evaluate.

    @Output: None
    '''
    
    # ====== Get GraspNetEval instances. ======
    ge = GraspNetEval(root = graspnet_root, camera = camera, split = split)

    # ====== eval 'seen' split (100-130) ======
    res, ap = ge.eval_seen(grasp_npys_eval_folder, proc = proc)
    print('\nEvaluation Result:\n----------\n{}, AP Seen={}'.format(camera, ap))
    
    # ====== eval 'similar' split (130-160) ======
    res, ap = ge.eval_similar(grasp_npys_eval_folder, proc = proc)
    print('\nEvaluation Result:\n----------\n{}, AP Similar={}'.format(camera, ap))
    
    # ====== eval 'novel' split (160-190) ======
    res, ap = ge.eval_novel(grasp_npys_eval_folder, proc = proc)
    print('\nEvaluation Result:\n----------\n{}, AP Novel={}'.format(camera, ap))
    

def eval_all_data(graspnet_root,grasp_npys_eval_folder,camera,split,proc):
    '''
    @Function: eval all data(scene_0100 - scene_0190)
            
    @Input:
        - graspnet_root: ROOT PATH FOR GRASPNET
        - grasp_npys_eval_folder: string of the folder that saves the npy files.
        - camera: string of type of camera, "realsense" or "kinect".
        - split: string of the date split.
        - proc: int of the number of processes to use to evaluate.

    @Output: None
    '''
    
    # ====== Get GraspNetEval instances. ======
    ge = GraspNetEval(root = graspnet_root, camera = camera, split = split)

    # ====== eval all data ======
    print('Evaluating kinect')
    res, ap = ge.eval_all(grasp_npys_eval_folder, proc = proc)
    print('\nEvaluation Result:\n----------\n{}, AP={}'.format(camera, ap))

def eval(graspnet_root,camera,format,scene_sum,sceneId_start,sceneId_end,split,proc,specified_sceneId,if_eval_a_single_scene,if_eval_scenes,if_eval_train_dataset,if_eval_valid_dataset,if_eval_test_dataset,if_eval_all_data):
    '''
    @Function: eval
        - 1.eval_a_single_scene
        - 2.eval_scenes
        - 3.eval_train_dataset
        - 4.eval_valid_dataset
        - 5.eval_test_dataset
        - 6.eval_all_data
            
    @Input:
        - graspnet_root: ROOT PATH FOR GRASPNET.
        - camera: string of type of camera, "realsense" or "kinect".
        - format: string of grasp format, '6d' or 'rect'.
        - scene_sum: int of the number of scenes in the dataset.
        - sceneId_start: int of the starting scene index.(sceneId max range is [0:180])
        - sceneId_end: int of the ending scene index.
        - split: string of the date split.
        - proc: int of the number of processes to use to evaluate.
        - specified_sceneId: int, the scene index

    @Output: None
    '''

    grasp_npys_eval_folder = graspnet_root + '/grasp_npys_eval/grasp_npys_eval_'+camera+'_'+format+'_'+str(scene_sum)
    
    if if_eval_a_single_scene:
        eval_a_single_scene(graspnet_root,grasp_npys_eval_folder,camera,split,specified_sceneId)
    
    if if_eval_scenes:
        eval_scenes(graspnet_root,grasp_npys_eval_folder,camera,split,sceneId_start,sceneId_end,proc)
    
    if if_eval_train_dataset:
        eval_train_dataset(graspnet_root,grasp_npys_eval_folder,camera,split,proc)

    if if_eval_valid_dataset:
        eval_valid_dataset(graspnet_root,grasp_npys_eval_folder,camera,split,proc)
    
    if if_eval_test_dataset:
        eval_test_dataset(graspnet_root,grasp_npys_eval_folder,camera,split,proc)
    
    if if_eval_all_data:
        eval_all_data(graspnet_root,grasp_npys_eval_folder,camera,split,proc)

def main():
    graspnet_root = '/home/gmh/graspnet' # ROOT PATH FOR GRASPNET
    camera = 'kinect'   
    format = '6d'
    scene_sum = 100
    sceneId_start = 0
    sceneId_end = 100
    
    split = 'test'
    proc = 24
    specified_sceneId = 121
    if_eval_a_single_scene = False
    if_eval_scenes = False
    if_eval_train_dataset = False
    if_eval_valid_dataset = False
    if_eval_test_dataset = False
    if_eval_all_data = False
    
    eval(graspnet_root=graspnet_root,camera=camera,format=format,scene_sum=scene_sum,sceneId_start=sceneId_start,sceneId_end=sceneId_end,split=split,proc=proc,specified_sceneId=specified_sceneId,if_eval_a_single_scene=if_eval_a_single_scene,if_eval_scenes=if_eval_scenes,if_eval_train_dataset=if_eval_train_dataset,if_eval_valid_dataset=if_eval_valid_dataset,if_eval_test_dataset=if_eval_test_dataset,if_eval_all_data=if_eval_all_data)

    
if __name__ == '__main__':
    main()

