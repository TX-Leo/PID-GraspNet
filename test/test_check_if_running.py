import subprocess
import time
import psutil

# script_path = "../generate_grasp_jsons_v3.py"  # 脚本的路径

# while True:
#     # 检测脚本是否在运行
#     process = subprocess.Popen(["tasklist", "/FI", f'IMAGENAME eq {script_path}'], stdout=subprocess.PIPE)
#     output, _ = process.communicate()

#     if script_path.lower() in output.decode().lower():
#         print("脚本正在运行...")
#     else:
#         print("脚本停止运行，重新启动脚本...")
#         # subprocess.Popen(["python", script_path])

#     time.sleep(3)  # 检测间隔时间

script_name = "generate_grasp_jsons_v3.py"
while True:
    # 获取当前正在运行的进程列表
    process_list = [p.name() for p in psutil.process_iter()]
    print(process_list)
    if script_name in process_list:
        print("脚本正在运行...")
    else:
        print("脚本停止运行，重新启动脚本...")
        # 执行重新启动脚本的操作

    time.sleep(5)  # 检测间隔时间