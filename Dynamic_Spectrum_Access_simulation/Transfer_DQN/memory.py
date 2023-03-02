import numpy as np
import math
from Transfer_DQN.sum_tree import SumTree



class Memory(object):
    def __init__(self, batch_size, max_size, beta):
        self.batch_size = batch_size  
        self.max_size = 2**math.ceil(math.log2(max_size)) 
        self.beta = beta

        self._sum_tree = SumTree(max_size)

    def store_transition(self, s, a, r, s_, done):
        self._sum_tree.add((s, a, r, s_, done))

    def store_transition_with_prior_one(self, s, a, r, s_, done):
        self._sum_tree.add_prior_one((s, a, r, s_, done))

    def get_mini_batches(self):
        n_sample = self.batch_size if self._sum_tree.size >= self.batch_size else self._sum_tree.size
        total = self._sum_tree.get_total()

        step = total // n_sample

        points_transitions_probs = []
        for i in range(n_sample):
            v = np.random.uniform(i * step, (i + 1) * step - 1)

            t = self._sum_tree.sample(v)
            points_transitions_probs.append(t)

        points, transitions, probs = zip(*points_transitions_probs)

        # print(self._sum_tree.get_min())

        # temp = n_sample * self._sum_tree.get_min()
        # print("the temp ", temp)
        # while temp == 0:
        #     temp = n_sample * self._sum_tree.get_min()
        # max_importance_ratio = temp ** -self.beta
             
        # max_importance_ratio = (n_sample * self._sum_tree.get_min() + 1e-8)**-self.beta
        # importance_ratio = [(n_sample * probs[i])**-self.beta / max_importance_ratio
        #                     for i in range(len(probs))]
        
        
        importance_ratio = [1 for i in range(len(probs))] 
        
        # min_prob = self._sum_tree.get_min() / self._sum_tree.get_total()
        # if min_prob == 0:
        #     importance_ratio = [0 for i in range(len(probs))] 
        # else:
        #    
        #     importance_ratio = [n_sample*np.power(probs[i]/min_prob, -self.beta) for i in range(len(probs))] 
        #     # importance_ratio = [((n_sample * probs[i] + 1e-15) / min_prob) ** -self.beta for i in range(len(probs))]

        return points, tuple(np.array(e) for e in zip(*transitions)), importance_ratio

    def update(self, points, td_error):
        for i in range(len(points)):
            self._sum_tree.update(points[i], td_error[i])

    def update_source(self, points, td_error):
        for i in range(len(points)):
            # _, _, pre_weight = self._sum_tree.get_transition(points[i])
            # print(pre_weight* self._sum_tree.get_total(),end="     ")
            idx = points[i] + self._sum_tree.capacity - 1
            pre_weight = self._sum_tree.tree[idx]
            cur_weight = pre_weight / (1+td_error[i]/1000)
            # print(cur_weight)
            if cur_weight < 0.01:
                cur_weight = 0.01
            # print(cur_weight)
            self._sum_tree.update(points[i], cur_weight) ##############################
        # print("source", self._sum_tree.tree[idx])

    def update_target(self, points, td_error):
        for i in range(len(points)):
            idx = self._sum_tree.capacity - 1 + points[i]
            pre_weight = self._sum_tree.tree[idx]
            cur_weight = pre_weight + td_error[i]/5000
            # _, _, pre_weight = self._sum_tree.get_transition(points[i])
            # # print(pre_weight)
            # cur_weight = pre_weight * self._sum_tree.get_total() + td_error[i]/100000
            if cur_weight > 5:
                cur_weight = 5
            self._sum_tree.update(points[i], cur_weight) #####################################
            # print("target", self._sum_tree.tree[idx])

    def get_weights(self, points):
        weights = []
        for i in range(len(points)):
            idx = points[i] + self._sum_tree.capacity - 1
            # print(len(weights), idx, self._sum_tree.tree[idx])
            # print(len(weights), idx, self._sum_tree.tree[idx])
            # print(len(weights), idx, self._sum_tree.tree[idx])
            # print(len(weights), idx, self._sum_tree.tree[idx])
            weights.append(self._sum_tree.tree[idx])
            # print(len(weights), idx, self._sum_tree.tree[idx])
        return weights

    def get_memory_all(self):
        n_all = self.max_size
        total = self._sum_tree.get_total()
        points_transitions_all = []

        for i in range(n_all):
            t = self._sum_tree.get_transition(i)
            points_transitions_all.append(t)

        points, transitions, probs = zip(*points_transitions_all)
        # print(tuple(np.array(e) for e in zip(*transitions)))
        # transitions = tuple(transitions)
        # print(type(transitions))
        min_prob = self._sum_tree.get_min() / self._sum_tree.get_total()
        # transitions = np.array(transitions)
        
        importance_ratio = [1 for i in range(len(probs))]
        
        # if min_prob == 0:
        #     importance_ratio = [0 for i in range(len(probs))]
        # else:
        #     importance_ratio = [n_all * np.power(probs[i] / min_prob + 1e-15, -self.beta) for i in range(len(probs))]
        # # print(np.array(transitions).shape)
        # # print(transitions[0:-1][2001])
        # # return points, self._unpark(transitions), importance_ratio
        return points, tuple(np.array(e) for e in zip(*transitions)), importance_ratio

    def _unpark(self, transitions):
        length = len(transitions)
        print(length)
        s = []
        a = []
        r = []
        s_ = []
        done = []
        for i in range(length):
            print(len(s))
            s.append(transitions[0][i])
            a.append(transitions[1][i])
            r.append(transitions[2][i])
            s_.append(transitions[3][i])
            done.append(transitions[4][i])
        return tuple([s, a, r, s_, done])

