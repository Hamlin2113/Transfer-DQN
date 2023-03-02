# /usr/bin/env python
# coding:utf-8
import numpy as np


class PU:
    def __init__(self, n_channel, max_stay_time=25, jump_table=[0]):
        self.n_channel = n_channel  
        self.jump_table = jump_table  
        self.jump_table_loc = 0 
        self.max_stay_time = max_stay_time - 1  
        
        self.stay_time = 0  
        
        self.action = jump_table[0]

    def step(self):
        

        if self.stay_time >= self.max_stay_time:

            self.stay_time = 0
            self.jump_table_loc += 1

            if self.jump_table_loc >= len(self.jump_table):
                self.jump_table_loc = 0
            self.action = self.jump_table[self.jump_table_loc]
        else:
     
            self.stay_time += 1

    def rand_pu_state(self, random_seed):
    
        
        np.random.seed(random_seed)
        self.stay_time = np.random.randint(0, self.max_stay_time)
        self.jump_table_loc = np.random.randint(0, len(self.jump_table))

    def get_PU_action(self):
        
        return self.action


class env_jumpTable:

    def __init__(self, n_channel, n_su, n_pu, pu_max_stay_time, pu_jump_table,
                 sense_time_slots=1, sense_error_prob=0, hasRounds=False, RoundMaxTime=1000, randProb=False):
        self.n_channel = n_channel  
        self.n_su = n_su  
        self.n_pu = n_pu  
        self.n_features = n_channel  
        self.n_actions = self.n_channel + 1  
        self.sense_error_prob = sense_error_prob  
        self.sense_time = sense_time_slots  

        self.PU_list = self.init_PU(self.n_pu, pu_max_stay_time, pu_jump_table)

        self._init_channel_state(randProb)
        
        self.setRoundByTimes(hasRounds, RoundMaxTime)

    def init_PU(self, n_pu, pu_max_stay_time, pu_jump_table):

        
        PU_list = []
        for i in range(n_pu):
            
            pu_imp = PU(self.n_channel, pu_max_stay_time[i], pu_jump_table[i])
            PU_list.append(pu_imp)
        self.PU_list = PU_list
    
        return self.PU_list

    def _init_channel_state(self, randProb=False):
        
        
        self.channel_state = np.zeros(self.n_channel)
        if randProb:  
            np.random.seed(10)  
            
            
            PU_randSeed_list = np.random.randint(0, 100, self.n_pu)
            
            for i in range(self.n_pu):
                self.PU_list[i].rand_pu_state(PU_randSeed_list[i])
        
        for i in range(self.n_pu):
            self.channel_state[self.PU_list[i].get_PU_action()] = 1
        
        return self.channel_state

    def render(self):

        
        pu_action = np.zeros(self.n_pu)  
        for i in range(self.n_pu):
            self.PU_list[i].step()  
            # print("there",self.PU_list[i].action)
            pu_action[i] = self.PU_list[i].action  
        self.channel_state = np.zeros(self.n_channel)  
        for i in range(self.n_pu):
            if pu_action[i] < self.n_channel:

                self.channel_state[np.int(pu_action[i])] = 1  

    def sense(self):

        tmp_dice = np.random.uniform(0, 1, size=(self.n_su, self.n_channel))
        error_index = tmp_dice <= self.sense_error_prob

        self.sensing_result = self.channel_state*(1-error_index) + (1-self.channel_state)*error_index

        return self.sensing_result[0, :] 

    def _successAccess(self, action_channel):

        if self.channel_state[action_channel] == 1:
            return 0  # access fail
        else:
            return 1  # access success

    def _puOccupyTimesCounter(self):

        num_pu_active_channel = 0
        for single_channel_state in self.channel_state:
            if single_channel_state == 1:
                num_pu_active_channel += 1
        return num_pu_active_channel

    def reset(self):

        self.stop_counter = 0  
        
        s = []
        for _ in range(self.sense_time):
            s.append(self.sense())
        s = np.array(s).flatten()  
        return s

    def setRoundByTimes(self, hasRound=False, RoundMaxTime=1000):
 
        self.hasRound = hasRound
        self.RoundMaxTime = RoundMaxTime

    def step(self, action):

        count = [0, 0, 0, 0, 0]  
        for i in range(action[1].astype(int)):
            
            self.render()  
            count[3] += self._puOccupyTimesCounter()  
            count[4] += self.n_channel  
            self.stop_counter += 1  
            
            if action[0].astype(int) == self.n_features:
                count[2] += 1
            elif self._successAccess(action[0].astype(int)) == 1:
                count[0] += 1
            else:
                count[1] += 1
        s_ = []  

        for i in range(self.sense_time):
            
            self.render()
            count[3] += self._puOccupyTimesCounter()   
            count[4] += self.n_channel  
            s_.append(self.sense())  
            self.stop_counter += 1
        s_ = np.array(s_).flatten()  
        
        r = self._countReward(count)
        if self.hasRound: 
           done = 1 if self.stop_counter >= self.RoundMaxTime else 0
        else:
            done = 0
        info = 0  
        self.count = count  
        return s_, r, done, info

    def step_use_for_dynamic_figure(self, action):

        count = [0, 0, 0, 0, 0]
        actions = []  
        states = []  
        for i in range(action[1].astype(int)):
            self.render()  
            count[3] += self._puOccupyTimesCounter()  
            count[4] += self.n_channel  

            states.append(self.channel_state)
            actions.append(action[0])

            self.stop_counter += 1
            if action[0].astype(int) == self.n_features:
                count[2] += 1
            elif self._successAccess(action[0].astype(int)) == 1:
                count[0] += 1
            else:
                count[1] += 1

        s_ = []  
        for i in range(self.sense_time):
            self.render()
            count[3] += self._puOccupyTimesCounter()  
            count[4] += self.n_channel  
            states.append(self.channel_state)
            actions.append(self.n_channel)
            s_.append(self.sense())
            self.stop_counter += 1

        s_ = np.array(s_).flatten()  
        r = self._countReward(count)
        if self.hasRound:
            done = 1 if self.stop_counter >= self.RoundMaxTime else 0
        else:
            done = 0
        info = 0
        self.count = count
        return s_, r, done, info, actions, states

    def _countReward(self, count, ):

        return count[0]*1 + count[1]*-3 + count[2]*-2 


if __name__ == '__main__':

    pu = PU(3, 2, [0, 1, 2])
    
    for i in range(12):
        
        print(pu.get_PU_action(), end=", ")
        
        pu.step()
    print("")
    from Environments.JumpTable import env_jumpTable
    import numpy as np


    sense_time_slots = 1
    env = env_jumpTable(n_channel=5, n_su=1,  
                        
                        n_pu=2, pu_max_stay_time=[2, 2], pu_jump_table=[[0, 1, 2], [1, 2, 0]],
                        sense_time_slots=sense_time_slots, sense_error_prob=0,  
                        hasRounds=False,  
                        RoundMaxTime=1000  
                        )

    s = env.reset()
    cur_state = env.sense()
    cur_time = 1
    print("------------执行su的决策[4,3]之前----------------")
    print("当前时隙：", cur_time, "；SU的感知结果：\n", s)
    print("当前时隙：", cur_time, "；当前信道状态：\n", cur_state)

    su_action = np.array([4, 3])  

    s_, r, done, info, actions, states = env.step_use_for_dynamic_figure(su_action)

    print("--------------执行su的决策[4,3]之后-----------------")
    cur_time = cur_time + su_action[1] + sense_time_slots 
    cur_state = env.sense()
    print("当前时隙：", cur_time, "；SU在env中执行决策后下一次的感知结果：\n", s_)
    print("当前时隙：", cur_time, "；SU当前的感知结果：（与step输出的s_保持一致,更方便随时获取当前时隙信道状态）\n", s_)
    print("当前时隙：", cur_time, "；SU在每一个时隙对应的信道使用情况，可与SU的action作对比\n", actions)
    print("SU的决策==n_channel时，用以表示当前SU正处于感知状态中，而非接入状态,SU的n_channel ==", env.n_channel)
    print("当前时隙：", cur_time, "；在每一个时隙的即时信道占用状态,即SU用频时+用频后的PU信道使用情况感知状态\n", states)