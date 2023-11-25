import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer

# sys.path.append(os.path.abspath("src"))
from generate_grasp_tsvs_real import load_grasp_txt_real

all_score = []
all_width = []
all_height = []
all_translation_1 = []
all_translation_2 = []
all_translation_3 = []
all_rotation_1 = []
all_rotation_2 = []
all_rotation_3 = []

def get_all_grasp_data_real(grasp_tsvs_real_file_path):
    global all_score
    global all_width
    global all_height
    global all_translation_1
    global all_translation_2
    global all_translation_3
    global all_rotation_1
    global all_rotation_2
    global all_rotation_3
    all_score_file_path = grasp_tsvs_real_file_path+'/all_score.txt'
    all_width_file_path = grasp_tsvs_real_file_path+'/all_width.txt'
    all_height_file_path = grasp_tsvs_real_file_path+'/all_height.txt'
    all_translation_1_file_path = grasp_tsvs_real_file_path+'/all_translation_1.txt'
    all_translation_2_file_path = grasp_tsvs_real_file_path+'/all_translation_2.txt'
    all_translation_3_file_path = grasp_tsvs_real_file_path+'/all_translation_3.txt'
    all_rotation_1_file_path = grasp_tsvs_real_file_path+'/all_rotation_1.txt'
    all_rotation_2_file_path = grasp_tsvs_real_file_path+'/all_rotation_2.txt'
    all_rotation_3_file_path = grasp_tsvs_real_file_path+'/all_rotation_3.txt'
    
    with open(all_score_file_path, "r") as file:
        lines = file.readlines()
        all_score = [float(line.strip()) for line in lines]
        file.close()
    with open(all_width_file_path, "r") as file:
        lines = file.readlines()
        all_width = [float(line.strip()) for line in lines]
        file.close()
    with open(all_height_file_path, "r") as file:
        lines = file.readlines()
        all_height = [float(line.strip()) for line in lines]
        file.close()
    with open(all_translation_1_file_path, "r") as file:
        lines = file.readlines()
        all_translation_1 = [float(line.strip()) for line in lines]
        file.close()
    with open(all_translation_2_file_path, "r") as file:
        lines = file.readlines()
        all_translation_2 = [float(line.strip()) for line in lines]
        file.close()
    with open(all_translation_3_file_path, "r") as file:
        lines = file.readlines()
        all_translation_3 = [float(line.strip()) for line in lines]
        file.close()
    with open(all_rotation_1_file_path, "r") as file:
        lines = file.readlines()
        all_rotation_1 = [float(line.strip()) for line in lines]
        file.close()
    with open(all_rotation_2_file_path, "r") as file:
        lines = file.readlines()
        all_rotation_2 = [float(line.strip()) for line in lines]
        file.close()
    with open(all_rotation_3_file_path, "r") as file:
        lines = file.readlines()
        all_rotation_3 = [float(line.strip()) for line in lines]
        file.close()
    print(f'==== There are {len(all_score)} data totally!!! ====')
    # all_score.sort()
    # all_width.sort()
    # all_height.sort()
    # all_translation_1.sort()
    # all_translation_2.sort()
    # all_translation_3.sort()
    # all_rotation_1.sort()
    # all_rotation_2.sort()  
    # all_rotation_3.sort()

def encode_and_plot(all_data,your_data,caption,save_dir):
    # 原始数据
    all_data = np.array(all_data)  # 你的数据列表，包含100多万条数据
    # 将all_data和your_data合并为一个数组
    data = np.concatenate([all_data, [your_data]])
    # 使用自适应分箱编码将数据编码到0-255的范围内
    encoder = KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile')
    data_encoded = encoder.fit_transform(data.reshape(-1, 1))
    # 获取all_data_encoded和your_data_encoded
    all_data_encoded = data_encoded[:-1]  # 去除your_data的编码
    your_data_encoded = data_encoded[-1]  # your_data的编码
    print(f"===={caption} your_data:{your_data}  your_data_encoded:{your_data_encoded}====")

    # 绘制原始数据的分布
    plt.figure(figsize=(10, 5))
    plt.hist(all_data, bins=100, color='blue', alpha=0.7, label='Original Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(caption+' Distribution of Original Data')
    plt.legend()
    # 保存原始数据的分布图
    plt.savefig(save_dir+'/'+caption+'original_data_distribution.png')
    # 绘制编码后数据的分布
    plt.figure(figsize=(10, 5))
    plt.hist(all_data_encoded, bins=256, color='red', alpha=0.7, label='Encoded Data')
    plt.xlabel('Encoded Value')
    plt.ylabel('Frequency')
    plt.title(caption+' Distribution of Encoded Data')
    plt.legend()
    plt.savefig(save_dir+'/'+caption+'encoded_data_distribution.png')
    # # 显示图形
    # plt.show()
    
    return your_data_encoded

def decode(all_data,encoded_value,caption):  
    all_data = np.array(all_data)  # 你的数据列表，包含100多万条数据
    # 使用自适应分箱编码将数据编码到0-255的范围内
    encoder = KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile')
    # 拟合编码器
    encoder.fit(all_data.reshape(-1, 1))
    # 解码编码后的数据
    decoded_value = encoder.inverse_transform(np.array([encoded_value]).reshape(1, -1))[0]
    print(f'===={caption} encoded_value:{encoded_value} decoded_value:{decoded_value}====')
    return decoded_value

def compute_accuracy(original_data, decoded_data):
    # 计算绝对误差
    absolute_error = np.abs(original_data - decoded_data)
    # 计算平均绝对误差
    mean_absolute_error = np.mean(absolute_error)
    # 计算相对误差（相对于原始数据）
    relative_error = absolute_error / np.abs(original_data)
    # 计算平均相对误差
    mean_relative_error = np.mean(relative_error)

    return mean_absolute_error, mean_relative_error

def test():
    graspnet_root = r'C:\Users\v-zhiwang2\Downloads\graspnet-v2' #'D:/dataset/graspnet' # 'D:/dataset/graspnet' # /Volumes/WD/dataset/graspnet
    camera = 'kinect'
    format = '6d'
    scene_sum = 100
    
    # ====== get all grasp predicted encoded data ======
    all_width_encoded = [i for i in range(256)]

    # ====== get all grasp real uncoded data ======
    grasp_tsvs_real_file_path=graspnet_root + '/grasp_tsvs_real/grasp_tsvs_real_'+camera+'_'+format+'_'+str(scene_sum)
    all_data_real = load_grasp_txt_real(grasp_tsvs_real_file_path,uncoded_or_encoded='uncoded')
    all_score_uncoded_real,all_width_uncoded_real,all_height_uncoded_real,all_depth_uncoded_real,all_translation_1_uncoded_real,all_translation_2_uncoded_real,all_translation_3_uncoded_real,all_rotation_1_uncoded_real,all_rotation_2_uncoded_real,all_rotation_3_uncoded_real = all_data_real

    all_width_uncoded = KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='quantile').fit(all_width_uncoded_real.reshape(-1, 1)).inverse_transform(np.array(all_width_encoded).reshape(-1, 1)).flatten().tolist()

    print(f'all_width_uncoded:\n{all_width_uncoded}')
    print(type(all_width_uncoded))
    
    def plot(data,caption,save_dir): 
        ''' plot image in speicified save directory'''
        plt.figure(figsize=(10, 5))
        plt.hist(data, bins=100, color='blue', alpha=0.7, label='Uncoded Data')
        plt.xlabel('Uncoded Value')
        plt.ylabel('Frequency')
        plt.title(caption+' Distribution of Uncoded Data')
        plt.legend()
        plt.savefig(save_dir+'/'+caption+'_distribution.png')
        print(f'====== {caption} distribution is ploted successfully!!! ======') 
        
    plot(data=all_width_uncoded,caption='all_width_uncoded',save_dir = r'C:\Users\v-zhiwang2\Downloads')
    
    
def main():
    # ======== get all original data ======== 
    graspnet_root = 'D:/dataset/graspnet' # 'D:/dataset/graspnet' # /Volumes/WD/dataset/graspnet
    camera = 'kinect'
    format = '6d'
    scene_sum = 100
    grasp_tsvs_real_file_path = graspnet_root+'/grasp_tsvs_real/grasp_tsvs_real_'+camera+'_'+format+'_'+str(scene_sum)
    get_all_grasp_data_real(grasp_tsvs_real_file_path)
    
    # ======== you data which is to be encoded =======
    you_data_score = 0.9000000357627869
    your_data_width=0.07
    your_data_height=0.019999999552965164
    your_data_translation_1=-0.05
    your_data_translation_2=0.05
    your_data_translation_3=0.45
    your_data_rotation_1=0
    your_data_rotation_2=-0.6
    your_data_rotation_3=-1
    
    # # ======== encode ======== 
    # your_data_encoded_score = encode_and_plot(all_data=all_score,your_data=you_data_score,caption='[Score]',save_dir = grasp_tsvs_real_file_path)
    # your_data_encoded_width = encode_and_plot(all_data=all_width,your_data=your_data_width,caption='[Width]',save_dir = grasp_tsvs_real_file_path)
    # your_data_encoded_height = encode_and_plot(all_data=all_height,your_data=your_data_height,caption='[Height]',save_dir = grasp_tsvs_real_file_path)
    # your_data_encoded_translation_1 = encode_and_plot(all_data=all_translation_1,your_data=your_data_translation_1,caption='[Translation-1]',save_dir = grasp_tsvs_real_file_path)
    # your_data_encoded_translation_2 = encode_and_plot(all_data=all_translation_2,your_data=your_data_translation_2,caption='[Translation-2]',save_dir = grasp_tsvs_real_file_path)
    # your_data_encoded_translation_3 = encode_and_plot(all_data=all_translation_3,your_data=your_data_translation_3,caption='[Translation-3]',save_dir = grasp_tsvs_real_file_path)
    # your_data_encoded_rotation_1 = encode_and_plot(all_data=all_rotation_1,your_data=your_data_rotation_1,caption='[Rotation-1]',save_dir = grasp_tsvs_real_file_path)
    # your_data_encoded_rotation_2 = encode_and_plot(all_data=all_rotation_2,your_data=your_data_rotation_2,caption='[Rotation-2]',save_dir = grasp_tsvs_real_file_path)
    # your_data_encoded_rotation_3 = encode_and_plot(all_data=all_rotation_3,your_data=your_data_rotation_3,caption='[Rotation-3]',save_dir = grasp_tsvs_real_file_path)
    # # plt.show()

    # ======== decode ======== 
    your_data_decoded_score = decode(all_data=all_score,encoded_value=0,caption='[Score]')
    your_data_decoded_width = decode(all_data=all_width,encoded_value=156,caption='[Width]')
    your_data_decoded_height = decode(all_data=all_height,encoded_value=0,caption='[Height]')
    your_data_decoded_translation_1 = decode(all_data=all_translation_1,encoded_value=152,caption='[Translation-1]')
    your_data_decoded_translation_2 =decode(all_data=all_translation_2,encoded_value=135,caption='[Translation-2]')
    your_data_decoded_translation_3 =decode(all_data=all_translation_3,encoded_value=154,caption='[Translation-3]')
    your_data_decoded_rotation_1 = decode(all_data=all_rotation_1,encoded_value=129,caption='[Rotation-1]')
    your_data_decoded_rotation_2 = decode(all_data=all_rotation_2,encoded_value=119,caption='[Rotation-2]')
    your_data_decoded_rotation_3 = decode(all_data=all_rotation_3,encoded_value=127,caption='[Rotation-3]')
    
    # ========排除score（基本上都是1）和height（基本上都是0.019999999552965164）的影响，评估一下encode-decode的误差=======
    # 原始数据
    your_data = [your_data_width,your_data_translation_1,your_data_translation_2,your_data_translation_3,your_data_rotation_1,your_data_rotation_2,your_data_rotation_3]  # 编码前的值
    your_data_decoded = [your_data_decoded_width,your_data_decoded_translation_1,your_data_decoded_translation_2,your_data_decoded_translation_3,your_data_decoded_rotation_1,your_data_decoded_rotation_2,your_data_decoded_rotation_3]  # 编码后再解码的值
    print(your_data,your_data_decoded)
    # # 计算编码和解码的精度
    # mae, mre = compute_accuracy(your_data_width, your_data_decoded_width)
    # print("Mean Absolute Error (MAE):", mae)
    # print("Mean Relative Error (MRE):", mre)
    
    
if __name__ == "__main__":
    # main()
    test()