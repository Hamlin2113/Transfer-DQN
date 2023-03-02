#! /usr/bin/env python
# coding:utf-8
import numpy as np


class env_Markov:
    def __init__(self, n_channel, n_su,
                 initChannelState=None, stays_prob=None,
                 sense_time_slots=1, sense_error_prob=0,
                 hasRounds=False, RoundMaxTime=1000):
        self.n_channel = n_channel 
        self.n_features = n_channel 
        self.n_su = n_su  
        self.n_actions = self.n_channel + 1  
  
        self.sense_error_prob = sense_error_prob 
        self.sense_time = sense_time_slots  

        self.build_Markov_channel(initChannelState, stays_prob)
        self.setRoundByTimes(hasRounds, RoundMaxTime)

    def _init_channel_state(self, initChannelState=None):
        if initChannelState is None:
            self.channel_state = np.random.choice(2, self.n_channel)
        else:
            self.channel_state = initChannelState

    def build_Markov_channel(self, initChannelState=None, stays_prob=None):  # zero-zero and one-one prob
        self._init_channel_state(initChannelState)  
        if stays_prob is None:
            self.stayZero_prob = np.random.uniform(0.8, 0.99, self.n_channel)  # zero-to-zero prob
            self.stayOne_prob = np.random.uniform(0.8, 0.99, self.n_channel)  # one-to-one prob
        else: 
            stays_prob = np.array(stays_prob)
            self.stayZero_prob = stays_prob[0]
            self.stayOne_prob = stays_prob[1]
        self.ZeroToOne_prob = 1 - self.stayZero_prob  
        self.OneToZero_prob = 1 - self.stayOne_prob  

    def render(self):

        stay_prob = self.channel_state * self.stayOne_prob + (1 - self.channel_state) * self.stayZero_prob

        tmp_dice = np.random.uniform(0, 1, self.n_channel)  # roll the dice between 0 and 1

        stay_index = tmp_dice < stay_prob   # choose out the index of the keep the same state

        self.channel_state = self.channel_state * stay_index + (1 - self.channel_state) * (1 - stay_index)
        self.channel_state = np.array(self.channel_state)

    def sense(self):
        tmp_dice = np.random.uniform(0, 1, size=(self.n_su, self.n_channel))
        error_index = tmp_dice <= self.sense_error_prob
        self.channel_state = np.array(self.channel_state)  
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
            # 00000000000000000000000000000000000000 self.render()
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
        return count[0]*5 + count[1]*-15 + count[2]*-10


if __name__ =='__main__':
    env = env_Markov(n_channel=4, n_su=1,
                     initChannelState=[0, 0, 0, 0], stays_prob=[[1, 0.5, 0.5, 0], [0, 0.5, 0.5, 1]],
                     sense_time_slots=0, sense_error_prob=0
                     )

    time = 10
    for i in range(time):
        print("第%d时隙, 信道状态为 [" % i, end="")
        for k in range(env.n_channel):
            print("%2d, " % env.channel_state[k], end="")
        print("\b\b]", end=", ")
        print("su的感知结果为", env.sense())
        env.render()  