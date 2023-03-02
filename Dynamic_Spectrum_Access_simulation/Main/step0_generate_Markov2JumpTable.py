import os
import sys
path1=os.path.abspath('.') 
path2=os.path.abspath('..')
sys.path.append(path2)
from Environments.Markov import env_Markov
import numpy as np



if __name__ == "__main__":

    jump_table_length = 30 
    n_channel = 8 
    # pu_probs = [[0.1, 0.9, 0.95, 0.95, 0.1, 0.05, 0.8, 0.5],  
    #             [0.9, 0.3, 0.95, 0.05, 0.55, 0.95, 0.8, 0.5]]  
    
    
    # pu_probs = [[0.7, 0.5, 0.6, 0.55, 0.5, 0.7, 0.5, 0.5],  
    #             [0.4, 0.3, 0.6, 0.55, 0.7, 0.4, 0.7, 0.5]] 
    

    # pu_probs = [[0.7, 0.5, 0.6, 0.55, 0.5, 0.6, 0.5, 0.5],  
    #             [0.3, 0.3, 0.3, 0.55, 0.7, 0.3, 0.7, 0.5]] 
    
    # pu_probs = [[0.85, 0.8, 0.9, 0.5, 0.7, 0.95, 0.7, 0.5],  
    #             [0.55, 0.5, 0.5, 0.55, 0.5, 0.6, 0.3, 0.5]]  
    
    pu_probs = [[0.75, 0.8, 0.9, 0.5, 0.7, 0.7, 0.7, 0.5],  
                [0.55, 0.5, 0.5, 0.55, 0.5, 0.6, 0.3, 0.5]]  
    
    init_channel_state = np.random.randint(0,2,n_channel)  
    # init_channel_state = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    print("初始信道状态为：", init_channel_state)
    env = env_Markov(n_channel=n_channel, n_su=1, initChannelState=init_channel_state, stays_prob=pu_probs)
    states = []  
    for i in range(jump_table_length):
        states.append(list(env.sense())) 
        env.render() 
    print("生成的信道状态序列为：", states) 

    n_pu = n_channel  
    channels_active_in_each_slot = [[] for i in range(jump_table_length)]
    num_active_in_each_slot = [0]*jump_table_length
    for time in range(jump_table_length):
        for channel in range(n_channel):
            if states[time][channel]==1:
                channels_active_in_each_slot[time].append(channel)
        num_active_in_each_slot[time] = len(channels_active_in_each_slot[time])
    n_pu = max(num_active_in_each_slot)  
    # print(num_active_in_each_slot)
    num_pu_in_same_channel_each_slot = n_pu - np.array(num_active_in_each_slot)
    print("每个时隙对应的活跃信道如下：", channels_active_in_each_slot)
    print("pu个数为：", n_pu)
    jump_table = np.zeros([n_pu, jump_table_length])
    for time in range(jump_table_length):
        position = 0 
        for ith_pu in range(n_pu):
            # print(position,end="and")
            # print(time)
            if len(channels_active_in_each_slot[time]) == 0:  
                jump_table[ith_pu][time] = n_channel
                continue
            if num_pu_in_same_channel_each_slot[time] == 0:  
                
                jump_table[ith_pu][time] = channels_active_in_each_slot[time][position]
                position += 1  
            else:  
                
                jump_table[ith_pu][time] = channels_active_in_each_slot[time][position]  
                num_pu_in_same_channel_each_slot[time] -= 1  
    print("生成的跳表为：\n", jump_table)

    print("可直接命令行复制并使用的跳表如下：")
    print("[")
    for ith_pu in range(n_pu):
        print("[",end="")
        for time in range(jump_table_length):
            print(np.int32(jump_table[ith_pu][time]),end=", ")
        print("\b\b],",end="\n")  
    print("]")

