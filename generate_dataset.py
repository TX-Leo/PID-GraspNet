__author__ = 'Zhi Wang'
__date__ = '2023.11.10'

import argparse

from src.check_dataset import check_dataset
from src.generate_grasp_scene_point_clouds import generate_grasp_scene_point_clouds
from src.generate_grasp_jsons import generate_grasp_jsons
from src.generate_grasp_tsvs_real import generate_grasp_tsvs_real
from src.generate_grasp_tsvs_train import generate_grasp_tsvs_train
from src.generate_grasp_dataloader_config import generate_grasp_dataloader_config
from src.generate_grasp_tsvs_predicted import generate_grasp_tsvs_predicted
from src.generate_grasp_npys_eval import generate_grasp_npys_eval
from src.eval import eval

def parse_args():
    parser = argparse.ArgumentParser(description='Generate dataset')
    
    # ====== which step ======
    parser.add_argument('--check_dataset', dest='if_check_dataset', action="store_true", help='')
    parser.add_argument('--generate_grasp_scene_point_clouds', dest='if_generate_grasp_scene_point_clouds', action="store_true", help='')
    parser.add_argument('--generate_grasp_jsons', dest='if_generate_grasp_jsons', action="store_true", help='')
    parser.add_argument('--generate_grasp_tsvs_real', dest='if_generate_grasp_tsvs_real', action="store_true", help='')
    parser.add_argument('--generate_grasp_tsvs_train', dest='if_generate_grasp_tsvs_train', action="store_true", help='')
    parser.add_argument('--generate_grasp_dataloader_config', dest='if_generate_grasp_dataloader_config', action="store_true", help='')
    parser.add_argument('--generate_grasp_tsvs_predicted', dest='if_generate_grasp_tsvs_predicted', action="store_true", help='')
    parser.add_argument('--generate_grasp_npys_eval', dest='if_generate_grasp_npys_eval', action="store_true", help='')
    parser.add_argument('--eval', dest='if_eval', action="store_true", help='')
    
    # ====== universal parameters ======
    parser.add_argument('--graspnet_root', type=str, default='/mnt/msranlpintern/dataset/graspnet-v2', help='')
    parser.add_argument('--train_or_test', type=str, default='train', choices=['train', 'test'],help='')
    parser.add_argument('--camera', type=str, default='kinect', choices=['kinect', 'realsense'], help='')
    parser.add_argument('--format', type=str, default='rect', choices=['rect', '6d'], help='')
    parser.add_argument('--scene_sum', type=int, default=100, help='')
    parser.add_argument('--sceneId_start', type=int, default=0, help='')
    parser.add_argument('--sceneId_end', type=int, default=100, help='')
    parser.add_argument('--annId_start', type=int, default=0, help='')
    parser.add_argument('--annId_end', type=int, default=256, help='')
    
    # ====== 0.for chech_dataset ======
    parser.add_argument('--check_graspnet_1billion', dest='if_check_graspnet_1billion', action="store_true", help='')
    parser.add_argument('--check_grasp_scene_point_clouds', dest='if_check_grasp_scene_point_clouds', action="store_true", help='')
    parser.add_argument('--check_grasp_jsons', dest='if_check_grasp_jsons', action="store_true", help='')
    parser.add_argument('--check_grasp_tsvs_real', dest='if_check_grasp_tsvs_real', action="store_true", help='')
    parser.add_argument('--check_grasp_tsvs_train', dest='if_check_grasp_tsvs_train', action="store_true", help='')
    parser.add_argument('--check_grasp_dataloader_config', dest='if_check_grasp_dataloader_config', action="store_true", help='')
    parser.add_argument('--check_else', dest='if_check_else', action="store_true", help='')
    
    # ====== 1.for generate_grasp_scene_point_clouds ======
    parser.add_argument('--align_to_table', type=bool, default=True, help='')
    parser.add_argument('--voxel_size', type=float, default=0.005, help='')
    parser.add_argument('--save_pcd', dest='if_save_pcd', action="store_true", help='')
    parser.add_argument('--save_npy_points', dest='if_save_npy_points', action="store_true", help='')
    parser.add_argument('--save_npy_points_and_colors', dest='if_save_npy_points_and_colors', action="store_true", help='')
    
    # ====== 2.for generate_grasp_jsons ======
    parser.add_argument('--fric_coef_thresh', type=float, default=0.2, help='')
    parser.add_argument('--no-show', dest ='show', action="store_false", help='')
    parser.add_argument('--specified_range_generate_grasp_jsons', action="store_true", help='')
    
    # ====== 3.for generate_grasp_tsvs_real ======
    parser.add_argument('--generate_grasp_tsvs_real_uncoded', dest='if_generate_grasp_tsvs_real_uncoded', action="store_true", help='')
    parser.add_argument('--generate_grasp_txt_real_uncoded', dest='if_generate_grasp_txt_real_uncoded', action="store_true", help='')
    parser.add_argument('--plot_grasp_txt_real_uncoded', dest='if_plot_grasp_txt_real_uncoded', action="store_true", help='')
    parser.add_argument('--generate_grasp_txt_real_encoded', dest='if_generate_grasp_txt_real_encoded', action="store_true", help='')
    parser.add_argument('--plot_grasp_txt_real_encoded', dest='if_plot_grasp_txt_real_encoded', action="store_true", help='')
    
    # ====== 4.for generate_grasp_tsvs_train ======
    parser.add_argument('--specified_range_generate_grasp_tsvs_train', action="store_true", help='')
    
    # ====== 5.for generate_grasp_dataloader_config ======
    parser.add_argument('--train_valid_proportion', type=int, default=90, help='')
    
    # ====== 6.for generate_grasp_tsvs_predicted ======
    parser.add_argument('--generate_grasp_tsvs_predicted_encoded', dest='if_generate_grasp_tsvs_predicted_encoded', action="store_true", help='')
    parser.add_argument('--generate_grasp_txt_predicted_encoded', dest='if_generate_grasp_txt_predicted_encoded', action="store_true", help='')
    parser.add_argument('--plot_grasp_txt_predicted_encoded', dest='if_plot_grasp_txt_predicted_encoded', action="store_true", help='')
    parser.add_argument('--generate_grasp_txt_predicted_uncoded', dest='if_generate_grasp_txt_predicted_uncoded', action="store_true", help='')
    parser.add_argument('--plot_grasp_txt_predicted_uncoded', dest='if_plot_grasp_txt_predicted_uncoded', action="store_true", help='')
    
    # ====== 7.for igenerate_grasp_npys_eval ======
    parser.add_argument('--specified_range_generate_grasp_npys_eval', action="store_true", help='')

    # ====== 8.for eval ======
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'],help='')
    parser.add_argument('--proc', type=int, default=24, help='')
    parser.add_argument('--eval_a_single_scene', dest='if_eval_a_single_scene', action="store_true", help='')
    parser.add_argument('--specified_sceneId', type=int, default=121, help='')
    parser.add_argument('--eval_scenes', dest='if_eval_scenes', action="store_true", help='')
    parser.add_argument('--eval_train_dataset', dest='if_eval_train_dataset', action="store_true", help='')
    parser.add_argument('--eval_valid_dataset', dest='if_eval_valid_dataset', action="store_true", help='')
    parser.add_argument('--eval_test_dataset', dest='if_eval_test_dataset', action="store_true", help='')
    parser.add_argument('--eval_all_data', dest='if_eval_all_data', action="store_true", help='')
    
    args = parser.parse_args()
    return args

def generate_dataset(args):
    '''
    @Function: 
        - 0.chech_dataset
        - 1.generate_grasp_scene_point_clouds
        - 2.generate_grasp_jsons
        - 3.generate_grasp_tsvs_real
        - 4.generate_grasp_tsvs_train
        - 5.generate_grasp_dataloader_config
        - 6.generate_grasp_tsvs_predicted
        - 7.generate_grasp_npys_eval
        - 8.eval
        
    @Input:
        - if_check_dataset
        - if_generate_grasp_scene_point_clouds
        - if_generate_grasp_jsons
        - if_generate_grasp_tsvs_real
        - if_generate_grasp_tsvs_train
        - if_generate_grasp_dataloader_config
        - if_generate_grasp_tsvs_predicted
        - if_generate_grasp_npys_eval
        - if_eval

    @Output: None
        - grasp_scene_point_clouds
        - grasp_jsons
        - grasp_tsvs_real
        - grasp_tsvs_train
        - grasp_dataloader_config
        - grasp_tsvs_predicted
        - grasp_npys_eval
    '''

    # if_check_dataset = False
    # if_generate_grasp_scene_point_clouds = False
    # if_generate_grasp_jsons = False
    # if_generate_grasp_tsvs_real = True
    # if_generate_grasp_tsvs_train = False
    # if_generate_grasp_dataloader_config = False
    # if_generate_grasp_tsvs_predicted = False
    # if_generate_grasp_npys_eval = False
    # if_eval = False
    
    # graspnet_root =  r'C:\Users\v-zhiwang2\Downloads\graspnet-v2' #'D:/dataset/graspnet' #'/mnt/msranlpintern/dataset/graspnet-v2' 
    # train_or_test = 'train'
    # camera = 'kinect'
    # format = 'rect' #'6d'
    # scene_sum = 100
    # sceneId_start = 0
    # sceneId_end = 1
    # annId_start = 0
    # annId_end = 1
    
    # # ====== 0.for chech_dataset ======
    # if_check_graspnet_1billion=False
    # if_check_grasp_scene_point_clouds=False
    # if_check_grasp_jsons=True
    # if_check_grasp_tsvs_real=True
    # if_check_grasp_tsvs_train=False
    # if_check_grasp_dataloader_config=False
    # if_check_else=False
    # # ====== 1.for generate_grasp_scene_point_clouds ======
    # align_to_table = True
    # voxel_size = 0.005
    # if_save_pcd = False
    # if_save_npy_points = True
    # if_save_npy_points_and_colors = False
    # # ====== 2.for generate_grasp_jsons ======
    # fric_coef_thresh=0.2
    # show = False
    # specified_range_generate_grasp_jsons = True
    # # ====== 3.for generate_grasp_tsvs_real ======
    # if_generate_grasp_tsvs_real_uncoded=False
    # if_generate_grasp_txt_real_uncoded=False
    # if_plot_grasp_txt_real_uncoded=False
    # if_generate_grasp_txt_real_encoded=False
    # if_plot_grasp_txt_real_encoded=True
    # # ====== 4.for generate_grasp_tsvs_train ======
    # specified_range_generate_grasp_tsvs_train = True
    # # ====== 5.for generate_grasp_dataloader_config ======
    # train_valid_proportion = 90
    # # ====== 6.for generate_grasp_tsvs_predicted ======
    # if_generate_grasp_tsvs_predicted_encoded=True
    # if_generate_grasp_txt_predicted_encoded=False
    # if_plot_grasp_txt_predicted_encoded=False
    # if_generate_grasp_txt_predicted_uncoded=False
    # if_plot_grasp_txt_predicted_uncoded=False
    # # ====== 7.for igenerate_grasp_npys_eval ======
    # specified_range_generate_grasp_npys_eval = True
    # # ====== 8.for eval ======
    # split = 'test'
    # proc = 24
    # specified_sceneId = 121
    # if_eval_a_single_scene = False
    # if_eval_scenes = False
    # if_eval_train_dataset = False
    # if_eval_valid_dataset = False
    # if_eval_test_dataset = False
    # if_eval_all_data = False
    
    # ====== which step ======
    if_check_dataset = args.if_check_dataset
    if_generate_grasp_scene_point_clouds = args.if_generate_grasp_scene_point_clouds
    if_generate_grasp_jsons = args.if_generate_grasp_jsons
    if_generate_grasp_tsvs_real = args.if_generate_grasp_tsvs_real
    if_generate_grasp_tsvs_train = args.if_generate_grasp_tsvs_train
    if_generate_grasp_dataloader_config = args.if_generate_grasp_dataloader_config
    if_generate_grasp_tsvs_predicted = args.if_generate_grasp_tsvs_predicted
    if_generate_grasp_npys_eval = args.if_generate_grasp_npys_eval
    if_eval = args.if_eval
    
    # ====== universal parameters ======
    graspnet_root = args.graspnet_root
    train_or_test = args.train_or_test
    camera = args.camera
    format = args.format
    scene_sum = args.scene_sum
    sceneId_start = args.sceneId_start
    sceneId_end = args.sceneId_end
    annId_start = args.annId_start
    annId_end = args.annId_end
    
    # ====== 0.for chech_dataset ======
    if_check_graspnet_1billion = args.if_check_graspnet_1billion
    if_check_grasp_scene_point_clouds = args.if_check_grasp_scene_point_clouds
    if_check_grasp_jsons = args.if_check_grasp_jsons
    if_check_grasp_tsvs_real = args.if_check_grasp_tsvs_real
    if_check_grasp_tsvs_train = args.if_check_grasp_tsvs_train
    if_check_grasp_dataloader_config = args.if_check_grasp_dataloader_config
    if_check_else = args.if_check_else
    
    # ====== 1.for generate_grasp_scene_point_clouds ======
    align_to_table = args.align_to_table
    voxel_size = args.voxel_size
    if_save_pcd = args.if_save_pcd
    if_save_npy_points = args.if_save_npy_points
    if_save_npy_points_and_colors = args.if_save_npy_points_and_colors
    
    # ====== 2.for generate_grasp_jsons ======
    fric_coef_thresh = args.fric_coef_thresh
    show = args.show
    specified_range_generate_grasp_jsons = args.specified_range_generate_grasp_jsons
    
    # ====== 3.for generate_grasp_tsvs_real ======
    if_generate_grasp_tsvs_real_uncoded = args.if_generate_grasp_tsvs_real_uncoded
    if_generate_grasp_txt_real_uncoded = args.if_generate_grasp_txt_real_uncoded
    if_plot_grasp_txt_real_uncoded = args.if_plot_grasp_txt_real_uncoded
    if_generate_grasp_txt_real_encoded = args.if_generate_grasp_txt_real_encoded
    if_plot_grasp_txt_real_encoded = args.if_plot_grasp_txt_real_encoded
    
    # ====== 4.for generate_grasp_tsvs_train ======
    specified_range_generate_grasp_tsvs_train = args.specified_range_generate_grasp_tsvs_train
    
    # ====== 5.for generate_grasp_dataloader_config ======
    train_valid_proportion = args.train_valid_proportion
    
    # ====== 6.for generate_grasp_tsvs_predicted ======
    if_generate_grasp_tsvs_predicted_encoded = args.if_generate_grasp_tsvs_predicted_encoded
    if_generate_grasp_txt_predicted_encoded = args.if_generate_grasp_txt_predicted_encoded
    if_plot_grasp_txt_predicted_encoded = args.if_plot_grasp_txt_predicted_encoded
    if_generate_grasp_txt_predicted_uncoded = args.if_generate_grasp_txt_predicted_uncoded
    if_plot_grasp_txt_predicted_uncoded = args.if_plot_grasp_txt_predicted_uncoded
    
    # ====== 7.for igenerate_grasp_npys_eval ======
    specified_range_generate_grasp_npys_eval = args.specified_range_generate_grasp_npys_eval
    
    # ====== 8.for eval ======
    split = args.split
    proc = args.proc
    specified_sceneId = args.specified_sceneId
    if_eval_a_single_scene = args.if_eval_a_single_scene
    if_eval_scenes = args.if_eval_scenes
    if_eval_train_dataset = args.if_eval_train_dataset
    if_eval_valid_dataset = args.if_eval_valid_dataset
    if_eval_test_dataset = args.if_eval_test_dataset
    if_eval_all_data = args.if_eval_all_data
    
    if if_check_dataset:
        # ====== 0. check dataset ======
        check_dataset(graspnet_root=graspnet_root,camera=camera,format=format,scene_sum=scene_sum,sceneId_start=sceneId_start,sceneId_end=sceneId_end,annId_start=annId_start,annId_end=annId_end,if_check_graspnet_1billion=if_check_graspnet_1billion,if_check_grasp_scene_point_clouds=if_check_grasp_scene_point_clouds,if_check_grasp_jsons=if_check_grasp_jsons,if_check_grasp_tsvs_real=if_check_grasp_tsvs_real,if_check_grasp_tsvs_train=if_check_grasp_tsvs_train,if_check_grasp_dataloader_config=if_check_grasp_dataloader_config,if_check_else=if_check_else)
        
    if if_generate_grasp_scene_point_clouds:
        # ====== 1. generate /grasp_scene_point_clouds ======
        generate_grasp_scene_point_clouds(graspnet_root=graspnet_root,camera=camera,sceneId_start=sceneId_start,sceneId_end=sceneId_end,align_to_table=align_to_table,show=show,voxel_size=voxel_size,if_save_pcd=if_save_pcd,if_save_npy_points=if_save_npy_points,if_save_npy_points_and_colors=if_save_npy_points_and_colors)
    
    if if_generate_grasp_jsons:
        # ====== 2. generate /grasp_jsons ======
        if specified_range_generate_grasp_jsons:
            generate_grasp_jsons(graspnet_root = graspnet_root , camera = camera , format = format , sceneId_start = sceneId_start,sceneId_end = sceneId_start + 1,annId_start = annId_start,annId_end = 256,fric_coef_thresh = fric_coef_thresh,show=show)
            generate_grasp_jsons(graspnet_root = graspnet_root , camera = camera , format = format , sceneId_start = sceneId_start + 1,sceneId_end = sceneId_end,annId_start = 0,annId_end = annId_end,fric_coef_thresh = fric_coef_thresh,show=show)
        else:
            generate_grasp_jsons(graspnet_root = graspnet_root , camera = camera , format = format , sceneId_start = sceneId_start,sceneId_end = sceneId_end,annId_start = annId_start,annId_end = annId_end,fric_coef_thresh = fric_coef_thresh,show=show)
            
    if if_generate_grasp_tsvs_real: 
        # ====== 3. generate /grasp_tsvs_real ======
        generate_grasp_tsvs_real(graspnet_root=graspnet_root,train_or_test=train_or_test,camera=camera,format=format,scene_sum=scene_sum,sceneId_start=sceneId_start,sceneId_end=sceneId_end,annId_start=annId_start,annId_end=annId_end,if_generate_grasp_tsvs_real_uncoded=if_generate_grasp_tsvs_real_uncoded,if_generate_grasp_txt_real_uncoded=if_generate_grasp_txt_real_uncoded,if_plot_grasp_txt_real_uncoded=if_plot_grasp_txt_real_uncoded,if_generate_grasp_txt_real_encoded=if_generate_grasp_txt_real_encoded,if_plot_grasp_txt_real_encoded=if_plot_grasp_txt_real_encoded)
    
    
    if if_generate_grasp_tsvs_train:
        # ====== 4. generate /grasp_tsvs_train ======
        if specified_range_generate_grasp_tsvs_train:
            generate_grasp_tsvs_train(graspnet_root = graspnet_root ,train_or_test = train_or_test, camera = camera , format = format , scene_sum=scene_sum,sceneId_start = sceneId_start,sceneId_end = sceneId_start + 1,annId_start = annId_start,annId_end = 256)
            generate_grasp_tsvs_train(graspnet_root = graspnet_root ,train_or_test = train_or_test, camera = camera , format = format , scene_sum=scene_sum,sceneId_start = sceneId_start + 1,sceneId_end = sceneId_end,annId_start = 0,annId_end = annId_end)
        else:
            generate_grasp_tsvs_train(graspnet_root = graspnet_root ,train_or_test = train_or_test, camera = camera , format = format , scene_sum=scene_sum,sceneId_start = sceneId_start,sceneId_end = sceneId_end,annId_start = annId_start,annId_end = annId_end)
            
    if if_generate_grasp_dataloader_config:
        # ====== 5. generate /grasp_dataloader_config ======
        generate_grasp_dataloader_config(graspnet_root = graspnet_root, camera = camera , format = format , scene_sum=scene_sum,sceneId_start = sceneId_start,sceneId_end = sceneId_end,annId_start = annId_start,annId_end = annId_end,train_valid_proportion=train_valid_proportion)

    if if_generate_grasp_tsvs_predicted:
        # ====== 6.generate /grasp_tsvs_predicted ======
        generate_grasp_tsvs_predicted(graspnet_root=graspnet_root,train_or_test=train_or_test,camera=camera,format=format,scene_sum=scene_sum,sceneId_start=sceneId_start,sceneId_end=sceneId_end,annId_start=annId_start,annId_end=annId_end,if_generate_grasp_tsvs_predicted_encoded=if_generate_grasp_tsvs_predicted_encoded,if_generate_grasp_txt_predicted_encoded=if_generate_grasp_txt_predicted_encoded,if_plot_grasp_txt_predicted_encoded=if_plot_grasp_txt_predicted_encoded,if_generate_grasp_txt_predicted_uncoded=if_generate_grasp_txt_predicted_uncoded,if_plot_grasp_txt_predicted_uncoded=if_plot_grasp_txt_predicted_uncoded)
    
    if if_generate_grasp_npys_eval:
        # ====== 7.generate /grasp_npys_eval ======
        if specified_range_generate_grasp_npys_eval:
            generate_grasp_npys_eval(graspnet_root=graspnet_root,train_or_test=train_or_test,camera=camera,format=format,scene_sum=scene_sum,sceneId_start=sceneId_start,sceneId_end=sceneId_start+1,annId_start=annId_start,annId_end=256)
            generate_grasp_npys_eval(graspnet_root=graspnet_root,train_or_test=train_or_test,camera=camera,format=format,scene_sum=scene_sum,sceneId_start=sceneId_start+1,sceneId_end=sceneId_end,annId_start=0,annId_end=annId_end)
        else:
            generate_grasp_npys_eval(graspnet_root=graspnet_root,train_or_test=train_or_test,camera=camera,format=format,scene_sum=scene_sum,sceneId_start=sceneId_start,sceneId_end=sceneId_end,annId_start=annId_start,annId_end=annId_end)
    
    if if_eval:
        # ====== 8.eval ======
        eval(graspnet_root=graspnet_root,camera=camera,format=format,scene_sum=scene_sum,sceneId_start=sceneId_start,sceneId_end=sceneId_end,split=split,proc=proc,specified_sceneId=specified_sceneId,if_eval_a_single_scene=if_eval_a_single_scene,if_eval_scenes=if_eval_scenes,if_eval_train_dataset=if_eval_train_dataset,if_eval_valid_dataset=if_eval_valid_dataset,if_eval_test_dataset=if_eval_test_dataset,if_eval_all_data=if_eval_all_data)


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(args)