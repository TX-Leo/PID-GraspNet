import os
import json
import xml.etree.ElementTree as ET

def get_object_names(xml_file):
    '''
        读取annotations下的一个xml文件来获取这个场景下的所有出现的物体，以字典形式返回{obj_id:obj_name}
    '''
    
    # 解析xml文件
    tree = ET.parse(xml_file)
    root = tree.getroot()

    all_obj_names_dict = {}
 
    # 遍历每个obj标签
    for obj in root.findall('obj'):
        obj_id = int(obj.find('obj_id').text)
        obj_name = obj.find('obj_name').text
        # 先去除物体名称的".ply" 后缀
        obj_name = obj_name.split('.')[0]            
        # 去除物体名称的下划线 和 前面有些会有的标号(014这种)
        obj_name = obj_name.split('_')
        for i in obj_name:
            if '0'<= i <= '9':
                obj_name.remove(i)
        res = obj_name[0]
        for i in range(1,len(obj_name)):
            res = res+ ' ' + obj_name[i]
        # 添加这个图文
        all_obj_names_dict[obj_id] = res

    return all_obj_names_dict


def get_all_obj_names_and_ids_json(graspnet_root,camera,sceneId_start,sceneId_end):
    '''
        获取所有场景出现过的所有物体的id和name键值对，保存为一个json文件，（其中id为75-87的，都出现在scene170-189里，而且没名字，没法知道，根据官网知道是DexNet adversarial objects
    '''
    # ========================所有物体的id和name的dict========================
    all_obj_names_and_ids_dict = {}
    # ========================遍历每个指定的场景和视角========================
    for sceneId in list(range(sceneId_start, sceneId_end)):
        # print(f'====[scene_{sceneId}] start processing====')
        if sceneId in [51,116]+list(range(170,190)): continue
        # ========================获取这个场景的所有物体（一个字典）========================
        xml_path = graspnet_root + '/scenes/scene_' +'{:04d}'.format(sceneId)+'/'+camera+'/annotations/0000.xml'
        all_obj_names_dict = get_object_names(xml_path)
        # print(f'=====scene_{sceneId}这个场景下的物体有：======\n{all_obj_names_dict}')
        
        # ========================更新所有物体========================
        all_obj_names_and_ids_dict.update(all_obj_names_dict)
        print(f'====[scene_{sceneId}] the names and ids are saved successfully!')

    # ======================== 保存 ========================
    all_obj_names_and_ids_dict = dict(sorted(all_obj_names_and_ids_dict.items(), key=lambda x: x[0]))
    all_obj_names_and_ids_json_path = graspnet_root + '/else/all_obj_names_and_ids.json'
    with open(all_obj_names_and_ids_json_path,"w") as json_file:
        json.dump(all_obj_names_and_ids_dict, json_file,indent=4)
    missing_ids = [id for id in range(88) if id not in all_obj_names_and_ids_dict]
    print(f'====[all_obj_names_and_ids.json] is saved successfully!====\n====[missing ids] {missing_ids}  ')
    
def check_if_collect_all_obj_names_and_ids(graspnet_root,camera,sceneId_start,sceneId_end):
    '''
        检查所有场景出现过的所有物体的id是否在json文件里保存过了
    '''
    # ========================从 JSON 文件中读取all_obj_names_and_ids_dict========================
    with open(graspnet_root + '/else/all_obj_names_and_ids.json', 'r') as file:
        json_data = file.read()
    # 解析 JSON 数据为字典
    all_obj_names_and_ids_dict = json.loads(json_data)

    # ========================遍历每个指定的场景和视角========================
    for sceneId in list(range(sceneId_start, sceneId_end)):
        # print(f'====[scene_{sceneId}] start processing====')
        # ========================获取这个场景的所有物体（一个字典）========================
        xml_path = graspnet_root + '/scenes/scene_' +'{:04d}'.format(sceneId)+'/'+camera+'/annotations/0000.xml'
        # 解析xml文件
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # 遍历每个obj标签
        for obj in root.findall('obj'):
            # 遍历每个obj的id
            obj_id = obj.find('obj_id').text
            if obj_id not in all_obj_names_and_ids_dict:
                print(f'XXXXXXX [scene_{sceneId}] id_{obj_id} is not in the all_obj_names_and_ids_json file XXXXXXX')
        print(f'====[scene_{sceneId}] is checked successfully!')

def find_id(graspnet_root,camera,sceneId_start,sceneId_end):
    '''
        检查所有场景出现过的所有物体的id是否在json文件里保存过了
    '''
    specific_id = 57
    # ========================遍历每个指定的场景和视角========================
    for sceneId in list(range(sceneId_start, sceneId_end)):
        # print(f'====[scene_{sceneId}] start processing====')
        # ========================获取这个场景的所有物体（一个字典）========================
        xml_path = graspnet_root + '/scenes/scene_' +'{:04d}'.format(sceneId)+'/'+camera+'/annotations/0000.xml'
        # 解析xml文件
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # 遍历每个obj标签
        for obj in root.findall('obj'):
            # 遍历每个obj的id
            obj_id = obj.find('obj_id').text
            if int(obj_id) == specific_id:
                print(f'==== [scene_{sceneId}] id_{obj_id} ====')


def test():
    # obj_name = '072-i_toy_airplane.ply'
    # obj_name = 'kispa_cleanser.ply'
    # obj_name = '032_knife.ply'
    # obj_name = '003_cracker_box.ply'
    # obj_name = 'tape.ply'
    obj_name = '065-b_cups.ply'
    # 先去除物体名称的".ply" 后缀
    obj_name = obj_name.split('.')[0]            
    # 去除物体名称的下划线 和 前面有些会有的标号(014这种)
    obj_name = obj_name.split('_')
    for i in obj_name:
        if '0'<= i <= '9':
            obj_name.remove(i)
    res = obj_name[0]
    for i in range(1,len(obj_name)):
        res = res+ ' ' + obj_name[i]
    print(res) 

        
def main():
    graspnet_root = 'D:/dataset/graspnet' # 'D:/dataset/graspnet' # /Volumes/WD/dataset/graspnet
    camera = 'kinect'
    sceneId_start = 0
    sceneId_end = 190
    # get_all_obj_names_and_ids_json(graspnet_root=graspnet_root,camera = camera, sceneId_start=sceneId_start,sceneId_end=sceneId_end)
    # check_if_collect_all_obj_names_and_ids(graspnet_root=graspnet_root,camera = camera, sceneId_start=sceneId_start,sceneId_end=sceneId_end)
    find_id(graspnet_root=graspnet_root,camera = camera, sceneId_start=sceneId_start,sceneId_end=sceneId_end)

if __name__ == '__main__':
    main()
    # test()