from pynvml import *

def show_gpu(simple=True):
    #initialize
    nvmlInit()
    #get gpu count
    deviceCount = nvmlDeviceGetCount()
    total_memory = 0
    total_free = 0
    total_used = 0
    gpu_names = ""
    gpu_num = deviceCount

    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        gpu_name = nvmlDeviceGetName(handle).decode('utf-8')

        if not simple:
            print("[ GPU{}: {}".format(i, gpu_name), end="    ")
            print("总共显存: {}G".format((info.total // 1048576) / 1024), end="    ")
            print("空余显存: {}G".format((info.free // 1048576) / 1024), end="    ")
            print("已用显存: {}G".format((info.used // 1048576) / 1024), end="    ")
            print("显存占用率: {}%".format(info.used / info.total), end="    ")
            print("运行温度: {}摄氏度 ]".format(nvmlDeviceGetTemperature(handle, 0)))
        gpu_names += gpu_name + ' '
        total_memory += (info.total // 1048576) / 1024
        total_free += (info.free // 1048576) / 1024
        total_used += (info.used // 1048576) / 1024
    print("显卡名称：[{}]，显卡数量：[{}]，总共显存；[{}G]，空余显存：[{}G]，已用显存：[{}G]，显存占用率：[{}%]。".format(gpu_names, gpu_num, total_memory, total_free, total_used, (total_used/total_memory)))
    #close
    nvmlShutdown()

