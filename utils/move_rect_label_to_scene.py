'''
实现从官网直接下载的rect_labels文件夹下，把每个对应从scene的一些rect复制到scenes对应的里面（看官网的数据集结构）
'''

import os
import shutil
def move_rect_label_to_scene(kinect_or_realsense):
    root_path = 'D:/dataset/'
    for i in range(256):
        num = '{:04d}'.format(i)
        rect_labels_path = root_path+'./graspnet/rect_labels/scene_'+num+'/'+kinect_or_realsense
        
        scene_rect_path = root_path+'./graspnet/scenes/scene_'+num+'/'+kinect_or_realsense+'/rect/'

        if not os.path.exists(scene_rect_path):
            os.makedirs(scene_rect_path)
        
        # 遍历源文件夹下的所有文件和子文件夹
        for root, dirs, files in os.walk(rect_labels_path):
            # 遍历文件夹中的所有文件
            for file in files:
                # 构造源文件和目标文件的路径
                src_path = os.path.join(root, file)
                dst_path = os.path.join(scene_rect_path, file)
                # 移动文件到目标文件夹下
                shutil.move(src_path, dst_path)
        print('scene_'+num+' 移动成功!')

def main():
    move_rect_label_to_scene(kinect_or_realsense='kinect')
    move_rect_label_to_scene(kinect_or_realsense='realsense')

if __name__ == '__main__':
    main()