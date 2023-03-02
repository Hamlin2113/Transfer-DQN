import gym
import numpy as np
import tensorflow as tf
from DQN.memory import *


initializer_helper = { 
    'kernel_initializer': tf.random_normal_initializer(0., 0.3),
    'bias_initializer': tf.constant_initializer(0.1)
}


class DQN(object):
    def __init__(self, sess, s_dim, a_dim, batch_size, gamma, lr, epsilon, replace_target_iter, memory_size=10000):
        self.sess = sess
        self.s_dim = s_dim 
        self.a_dim = a_dim 
        self.gamma = gamma
        self.lr = lr  # learning rate
        self.epsilon = epsilon
        self.replace_target_iter = replace_target_iter
        self.memory = Memory(batch_size, memory_size)
        self._learn_step_counter = 0
        self._generate_model()

    def choose_action(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.a_dim)
        else:
            q_eval_z = self.sess.run(self.q_eval_z, feed_dict={
                self.s: s[np.newaxis, :]
            })
            return q_eval_z.squeeze().argmax()

    def _generate_model(self):
        # 构建DQN的价值网络模型
        self.s = tf.placeholder(tf.float32, shape=(None, self.s_dim), name='s')  
        self.a = tf.placeholder(tf.float32, shape=(None, self.a_dim), name='a')  
        self.r = tf.placeholder(tf.float32, shape=(None, 1), name='r')
        self.s_ = tf.placeholder(tf.float32, shape=(None, self.s_dim), name='s_') 
        self.done = tf.placeholder(tf.float32, shape=(None, 1), name='done')  

        self.q_eval_z = self._build_net(self.s, 'eval_net', True)
        self.q_target_z = self._build_net(self.s_, 'target_net', False)
        q_target = self.r + self.gamma \
            * tf.reduce_max(self.q_target_z, axis=1, keepdims=True) * (1 - self.done)
        q_eval = tf.reduce_sum(self.a * self.q_eval_z, axis=1, keepdims=True)

        self.loss = tf.reduce_mean(tf.squared_difference(q_target, q_eval))  
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)  
        # self.optimizer = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)  
        # self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)  
        # self.optimizer = tf.train.MomentumOptimizer(self.lr).minimize(self.loss)  
        # self.optimizer = tf.train.AdagradOptimizer(self.lr).minimize(self.loss)  
        self.param_target = tf.global_variables(scope='target_net')  # 保存参数
        self.param_eval = tf.global_variables(scope='eval_net')


        self.target_replace_ops = [tf.assign(t, e) for t, e in zip(self.param_target, self.param_eval)]

    def _build_net(self, s, scope, trainable):

        with tf.variable_scope(scope):  # relu sigmoid tanh
            # l = tf.layers.dense(s, 20, activation=tf.nn.relu, trainable=trainable, **initializer_helper)
            # k = tf.layers.dense(s, 30, activation=tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(s, 20, activation=tf.nn.relu, trainable=trainable, **initializer_helper)
            q_z = tf.layers.dense(l, self.a_dim, trainable=trainable, **initializer_helper)

        return q_z

    def store_transition_and_learn(self, s, a, r, s_, done):
        if self._learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_ops)

        one_hot_action = np.zeros(self.a_dim)
        one_hot_action[a] = 1

        self.memory.store_transition(s, one_hot_action, [r], s_, [done])
        loss = self._learn()
        self._learn_step_counter += 1
        return loss

    def _learn(self):
        s, a, r, s_, done = self.memory.get_mini_batches()
        # print("S: ", s, "A: ", a, "r: ", r, "S_: ", s_)
        loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict={
            self.s: s,
            self.a: a,
            self.r: r,
            self.s_: s_,
            self.done: done
        })
        return loss

    def make_action(self, s):
        q_eval_z = self.sess.run(self.q_eval_z, feed_dict={
            self.s: s[np.newaxis, :]
        })
        return q_eval_z.squeeze().argmax()

    def store_transition_and_count_loss(self, s, a, r, s_, done):
        one_hot_action = np.zeros(self.a_dim)
        one_hot_action[a] = 1
        self.memory.store_transition(s, one_hot_action, [r], s_, [done])
        ss, aa, rr, ss_, donee = self.memory.get_mini_batches()
        # print("S: ", s, "A: ", a, "r: ", r, "S_: ", s_)
        loss = self.sess.run(self.loss, feed_dict={
            self.s: ss,
            self.a: aa,
            self.r: rr,
            self.s_: ss_,
            self.done: donee
        })
        return loss

