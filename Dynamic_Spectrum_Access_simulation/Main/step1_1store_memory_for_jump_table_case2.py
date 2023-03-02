import os
import sys
path1=os.path.abspath('.')
path2=os.path.abspath('..')
sys.path.append(path2)

from Main.Core_functions import *
import os
from Environments.JumpTable import env_jumpTable
import dill as pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


result_folder = os.path.abspath("..") + '\\Stores\\results\\'
param_folder = os.path.abspath("..") + '\\Stores\\parameters\\'
# result_xls_manager(file_path, xls_name_prefix)

xls_manager_for_memory_pool = result_xls_manager(result_folder, "explore_for_memory")

TIME_UPPER_BOUND = 15
SLIP_WINDOW_SIZE = 1
SENSE_TIME_COST = 1


"""


    pu_probs = [[0.85, 0.8, 0.9, 0.5, 0.7, 0.95, 0.7, 0.5],  
                [0.55, 0.5, 0.5, 0.55, 0.5, 0.6, 0.3, 0.5]]



    pu_probs = [[0.85, 0.8, 0.9, 0.5, 0.7, 0.95, 0.7, 0.5], 
                [0.55, 0.5, 0.5, 0.55, 0.5, 0.6, 0.3, 0.5]]

"""


pu_jump_table_source = [
[0, 1, 1, 0, 4, 1, 1, 1, 7, 7, 4, 3, 3, 0, 1, 1, 2, 1, 2, 1, 1, 1, 4, 6, 3, 1, 1, 7, 3, 8],
[1, 1, 1, 0, 4, 1, 1, 1, 7, 7, 4, 3, 3, 0, 1, 1, 2, 2, 2, 1, 1, 1, 4, 6, 3, 1, 1, 7, 3, 8],
[3, 1, 1, 1, 4, 1, 1, 1, 7, 7, 4, 3, 3, 2, 1, 1, 2, 3, 2, 1, 1, 1, 4, 6, 3, 1, 1, 7, 3, 8],
[4, 1, 1, 2, 4, 3, 2, 1, 7, 7, 4, 3, 4, 4, 2, 1, 2, 4, 2, 1, 1, 1, 4, 6, 3, 1, 1, 7, 3, 8],
[5, 3, 1, 4, 4, 4, 3, 2, 7, 7, 4, 4, 6, 6, 3, 2, 4, 6, 2, 1, 1, 1, 4, 6, 3, 1, 2, 7, 6, 8],
[6, 5, 3, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 3, 7, 7, 3, 7, 1, 4, 4, 6, 3, 3, 7, 7, 7, 8],
]
pu_jump_table_target = [
[0, 1, 1, 0, 4, 1, 1, 1, 7, 7, 4, 3, 3, 0, 1, 1, 2, 1, 2, 1, 1, 1, 4, 6, 3, 1, 1, 7, 3, 8],
[1, 1, 1, 0, 4, 2, 1, 1, 5, 7, 4, 4, 3, 0, 2, 1, 2, 2, 2, 3, 1, 1, 4, 6, 3, 2, 1, 7, 3, 8],
[3, 1, 1, 1, 4, 1, 1, 1, 7, 7, 4, 3, 3, 2, 1, 1, 2, 3, 2, 1, 1, 1, 4, 6, 3, 1, 1, 7, 3, 8],
[4, 1, 1, 2, 4, 3, 2, 1, 7, 7, 4, 3, 4, 4, 2, 1, 2, 4, 2, 1, 1, 1, 4, 6, 3, 1, 1, 7, 3, 8],
[5, 3, 1, 4, 4, 4, 3, 2, 7, 7, 4, 4, 6, 6, 3, 2, 4, 6, 2, 1, 1, 1, 4, 6, 3, 1, 2, 7, 6, 8],
[6, 5, 3, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 3, 7, 7, 3, 7, 1, 4, 4, 6, 3, 3, 7, 7, 7, 8],
]
     
     
n_pu_source = len(pu_jump_table_source)
pu_max_stay_time_source = [1]*n_pu_source

n_pu_target = len(pu_jump_table_target)
pu_max_stay_time_target = [1]*n_pu_target
env_source = env_jumpTable(n_channel=n_channel, n_su=1, n_pu=n_pu_source, pu_max_stay_time=pu_max_stay_time_source, pu_jump_table=pu_jump_table_source, sense_time_slots=SENSE_TIME_COST, sense_error_prob=0)
env_target = env_jumpTable(n_channel=n_channel, n_su=1, n_pu=n_pu_target, pu_max_stay_time=pu_max_stay_time_target, pu_jump_table=pu_jump_table_target, sense_time_slots=SENSE_TIME_COST, sense_error_prob=0)
memory_size_source = 4096*4
memory_size_target = 1024*4
transfer_batch_source = 4
transfer_batch_target = 16

get_memory_explore_time = 5*1000*4


if __name__ == '__main__':

    memory_source = normal2Transfer(getMemoryPool(memory_size_source,
                                                  env_source,
                                                  get_memory_explore_time,
                                                  TIME_UPPER_BOUND,
                                                  SLIP_WINDOW_SIZE,
                                                  xls_manager_for_memory_pool),
                                    0.9)
    memory_source.batch_size = transfer_batch_source
    memory_target = normal2Transfer(getMemoryPool(memory_size_target,
                                                  env_target,
                                                  get_memory_explore_time,
                                                  TIME_UPPER_BOUND,
                                                  SLIP_WINDOW_SIZE,
                                                  xls_manager_for_memory_pool),
                                    0.9)
    memory_target.batch_size = transfer_batch_target
    with open(param_folder + 'memory_source'+'.pkl', 'wb') as file:
        pickle.dump(memory_source, file)
        print('源经验池 memory_source 保存成功：经验池大小%d' % memory_size_source)
    with open(param_folder + 'memory_target' + '.pkl', 'wb') as file:
        pickle.dump(memory_target, file)
        print("目标经验池 memory_target 保存成功: 目标经验池大小：%d" % memory_size_target)
    with open(param_folder + 'env_source'+'.pkl', 'wb') as file:
        pickle.dump(env_source, file)
        print("源仿真环境 env_source 保存成功")
    with open(param_folder + 'env_target' + '.pkl', 'wb') as file:
        pickle.dump(env_target, file)
        print("目标仿真环境 env_target 保存成功")
    with open(param_folder + 'TIME_UPPER_BOUND' + '.pkl', 'wb') as file:
        pickle.dump(TIME_UPPER_BOUND, file)
        print("参数 TIME_UPPER_BOUND 保存成功")
    with open(param_folder + 'SLIP_WINDOW_SIZE' + '.pkl', 'wb') as file:
        pickle.dump(SLIP_WINDOW_SIZE, file)
        print("参数 SLIP_WINDOW_SIZE 保存成功")
    # xls_manager_for_memory_pool.save_and_close()

""":cvar

pu_jump_table_source = [[0, 1, 1, 7, 7, 2, 2, 7, 7, 0, 0, 7, 7, 0, 0, 1, 1],
                        [1, 2, 2, 1, 1, 3, 3, 3, 3, 7, 7, 0, 0, 1, 1, 0, 0],
                        [2, 3, 3, 2, 2, 7, 7, 4, 4, 5, 5, 1, 1, 2, 2, 3, 3],
                        [4, 4, 4, 4, 4, 4, 4, 5, 5, 6, 6, 6, 2, 3, 3, 2, 2],
                        [3, 2, 2, 3, 3, 6, 6, 6, 6, 0, 0, 0, 0, 7, 7, 5, 5],
                        [5, 2, 2, 6, 6, 5, 5, 5, 5, 5, 5, 6, 6, 6, 2, 2, 2],
                        [4, 4, 7, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 2, 6, 6, 6]]
pu_jump_table_target =[
[0, 0, 0, 7, 7, 2, 2, 7, 7, 0, 0, 7, 7, 0, 0, 1, 1],
[1, 2, 2, 0, 0, 3, 3, 3, 3, 7, 7, 0, 0, 1, 1, 0, 0],
[2, 3, 3, 2, 2, 7, 7, 4, 4, 5, 5, 1, 1, 2, 2, 4, 4],
[4, 4, 4, 4, 4, 4, 4, 5, 5, 6, 6, 6, 2, 4, 4, 2, 2],
[3, 2, 2, 3, 3, 6, 6, 6, 6, 0, 0, 0, 0, 7, 7, 5, 5],
[5, 2, 2, 6, 6, 5, 5, 5, 5, 5, 5, 6, 6, 6, 2, 2, 2],
[4, 4, 7, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 2, 6, 6, 6]
]
pu_jump_table_target = [[0, 1, 2, 3, 4, 5, 6, 7],
                        [7, 6, 5, 2, 1, 0],
                        [4, 3],
                        [1, 3, 2, 4, 7, 5, 6, 0]]
"""

"""
pu_jump_table_source = [
[0, 1, 1, 7, 7, 2, 2, 7, 7, 0, 0, 7, 7, 0, 0, 1, 1],
[1, 2, 2, 1, 1, 3, 3, 3, 3, 7, 7, 0, 0, 1, 1, 0, 0],
[2, 3, 3, 2, 2, 7, 7, 4, 4, 5, 5, 1, 1, 2, 2, 3, 3],
[4, 4, 4, 4, 4, 4, 4, 5, 5, 6, 6, 6, 2, 3, 3, 2, 2],
[3, 2, 2, 3, 3, 6, 6, 6, 6, 0, 0, 0, 0, 7, 7, 5, 5],
[5, 2, 2, 6, 6, 5, 5, 5, 5, 5, 5, 6, 6, 6, 2, 2, 2],
[4, 4, 7, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 2, 6, 6, 6]
]

pu_jump_table_target =[
[0, 0, 0, 7, 7, 2, 2, 7, 7, 0, 0, 7, 7, 0, 0, 1, 1],
[1, 2, 2, 0, 0, 3, 3, 3, 3, 7, 7, 0, 0, 1, 1, 0, 0],
[2, 3, 3, 2, 2, 7, 7, 4, 4, 5, 5, 1, 1, 2, 2, 4, 4],
[4, 4, 4, 4, 4, 4, 4, 5, 5, 6, 6, 6, 2, 4, 4, 2, 2],
[3, 2, 2, 3, 3, 6, 6, 6, 6, 0, 0, 0, 0, 7, 7, 5, 5],
[5, 2, 2, 6, 6, 5, 5, 5, 5, 5, 5, 6, 6, 6, 2, 2, 2],
[4, 4, 7, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 2, 6, 6, 6]
]

pu_jump_table_target =[
[0, 1, 1, 7, 7, 2, 2, 7, 7, 0, 0, 7, 7, 0, 0, 1, 1],
[1, 2, 2, 1, 1, 3, 3, 3, 3, 7, 7, 0, 0, 1, 1, 0, 0],
[2, 3, 3, 2, 2, 7, 7, 4, 4, 5, 5, 1, 1, 2, 2, 3, 3],
[4, 4, 4, 4, 4, 4, 4, 5, 5, 6, 6, 6, 2, 3, 3, 2, 2],
[3, 2, 2, 3, 3, 6, 6, 6, 6, 0, 0, 0, 0, 7, 7, 5, 5],
[5, 2, 2, 6, 6, 5, 5, 5, 5, 5, 5, 6, 6, 6, 2, 2, 2],
]

[
[0, 1, 1, 7, 7, 2, 2, 7, 7, 0, 0, 7, 7, 0, 0, 1, 1],
[1, 2, 2, 1, 1, 3, 3, 3, 3, 7, 7, 0, 0, 1, 1, 0, 0],
[2, 3, 3, 2, 2, 7, 7, 4, 4, 5, 5, 1, 1, 2, 2, 3, 3],
[4, 4, 4, 4, 4, 4, 4, 5, 5, 6, 6, 6, 2, 3, 3, 2, 2],
[3, 2, 2, 3, 3, 6, 6, 6, 6, 0, 0, 0, 0, 7, 7, 5, 5],
[5, 2, 2, 6, 6, 5, 5, 5, 5, 5, 5, 6, 6, 6, 2, 2, 2],
[4, 4, 7, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 2, 6, 6, 6],
[4, 4, 2, 2, 2, 5, 5, 5, 5, 2, 2, 2, 1, 2, 2, 2, 2],
]

[
[0, 1, 1, 7, 7, 2, 2, 7, 7, 0, 0, 7, 7, 0, 0, 1, 1],
[1, 2, 2, 1, 1, 3, 3, 3, 3, 7, 7, 0, 0, 1, 1, 0, 0],
[2, 3, 3, 2, 2, 7, 7, 4, 4, 5, 5, 1, 1, 2, 2, 3, 3],
[4, 4, 4, 4, 4, 4, 4, 5, 5, 6, 6, 6, 2, 3, 3, 2, 2],
[3, 2, 2, 3, 3, 6, 6, 6, 6, 0, 0, 0, 0, 7, 7, 5, 5],
[5, 2, 2, 6, 6, 5, 5, 5, 5, 5, 5, 6, 6, 6, 2, 2, 2],
[4, 4, 7, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 2, 6, 6, 6],
[4, 4, 2, 2, 2, 5, 5, 5, 2, 3, 5, 5, 6, 7, 7, 7, 2]
]


pu_jump_table_source = [
[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0],
[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0, 0, 1],
[2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0, 0, 1, 1, 2],
[3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0, 0, 1, 1, 2, 2, 3],
[5, 5, 5, 5, 7, 7, 0, 0, 0, 0, 4, 4, 5, 5, 5, 5, 5],
[5, 5, 5, 5, 7, 7, 0, 1, 1, 1, 3, 3, 3, 3, 5, 5, 5]
]

pu_jump_table_target =[
[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0],
[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0, 0, 1],
[2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0, 0, 1, 1, 2],
[3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0, 0, 2, 2, 2, 2, 3],
[5, 5, 5, 5, 7, 7, 0, 0, 0, 0, 4, 4, 5, 5, 5, 5, 5],
[5, 5, 5, 5, 7, 7, 0, 2, 2, 2, 3, 3, 3, 3, 5, 5, 5]
]

pu_jump_table_target = [
[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0],
[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0, 0, 1],
[2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0, 0, 1, 1, 2],
[3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0, 0, 1, 1, 2, 2, 3],
[5, 5, 5, 5, 7, 7, 0, 0, 0, 0, 4, 4, 5, 5, 5, 5, 5]
]

pu_jump_table_target = [
[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0],
[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0, 0, 1],
[2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0, 0, 1, 1, 2],
[3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0, 0, 1, 1, 2, 2, 3],
[5, 5, 5, 5, 7, 7, 0, 0, 0, 0, 4, 4, 5, 5, 5, 5, 5],
[5, 5, 5, 5, 7, 7, 0, 1, 1, 1, 3, 3, 3, 3, 5, 5, 5],
[5, 5, 5, 5, 7, 7, 1, 0, 0, 0, 1, 1, 5, 5, 5, 5, 5]
]

pu_jump_table_source = [
[1,2],
[6,0,7,0],
[0,1,2,3,4,5,6,7],
[4,3,3,3,8,5,5,5,5,8,8,8,4,4,4,8],
[7,8,5,8]
]
t=[2,2,1,4,1]

pu_jump_table_target =[
[1,2],
[6,0,7,0],
[0,1,2,3,4,5,6,7],
[4,3,3,3,8,5,5,5,5,8,8,8,4,4,4,8],
[8,7,8,5]
]
t=[2,2,1,4,1]
pu_jump_table_target =[
[1,2],
[6,0,7,0],
[0,1,2,3,4,5,6,7],
[4,3,3,3,8,5,5,5,5,8,8,8,4,4,4,8],
]
t=[2,2,1,4]

pu_jump_table_target = [
[1,2],
[6,0,7,0],
[0,1,2,3,4,5,6,7],
[4,3,3,3,8,5,5,5,5,8,8,8,4,4,4,8],
[7,8,5,8],
[8,5,8,8,8,8,7,8]
]
t=[2,2,1,4,1,1]

pu_jump_table_source = [[7, 6, 5, 4, 3, 2, 1, 0],
                        [6, 5, 4, 3, 2, 1, 0, 7],
                        [5, 4, 3, 2, 1, 0, 7, 6],
                        [4, 3, 2, 1, 0, 7, 6, 5]
                        ]
pu_jump_table_target = [[7, 6, 5, 4, 3, 2, 1, 0],
                        [6, 5, 4, 3, 2, 1, 0, 7],
                        [5, 4, 3, 2, 1, 0, 7, 6],
                        [4, 3, 2, 1, 0, 7, 6, 5]
                        ]
pu_jump_table_source = [[0, 0, 0, 0, 1, 1, 1, 1],
                        [2, 2, 2, 2, 3, 3, 3, 3],
                        [4, 4, 4, 4, 5, 5, 5, 5],
                        [6, 6, 6, 6, 7, 7, 7, 7]
                        ]
pu_jump_table_target = [[7, 6, 5, 4, 3, 2, 1, 0],
                        [6, 5, 4, 3, 2, 1, 0, 7],
                        [5, 4, 3, 2, 1, 0, 7, 6],
                        [4, 3, 2, 1, 0, 7, 6, 5]
                        ]
"""

print('Step1 store memory case2 ')

