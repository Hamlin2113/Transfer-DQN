import gym
import numpy as np
import tensorflow as tf
from Transfer_DQN.memory import *


initializer_helper = {
    'kernel_initializer': tf.random_normal_initializer(0., 0.3),
    'bias_initializer': tf.constant_initializer(0.1)
}


class DQN(object):

    def __init__(self, sess, s_dim, a_dim, batch_size, gamma, lr, epsilon, replace_target_iter, memory_size=16384):
        self.sess = sess
        self.s_dim = s_dim  
        self.a_dim = a_dim 
        self.gamma = gamma
        self.lr = lr  # learning rate
        self.epsilon = epsilon  # epsilon-greedy
        self.replace_target_iter = replace_target_iter  

        self.memory = Memory(batch_size, memory_size, 0.9)
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
        self.s = tf.placeholder(tf.float32, shape=(None, self.s_dim), name='s')
        self.a = tf.placeholder(tf.float32, shape=(None, self.a_dim), name='a')
        self.r = tf.placeholder(tf.float32, shape=(None, 1), name='r')
        self.s_ = tf.placeholder(tf.float32, shape=(None, self.s_dim), name='s_')
        self.done = tf.placeholder(tf.float32, shape=(None, 1), name='done')

        self.importance_ratio = tf.placeholder(tf.float32, shape=(None, 1), name='importance_ratio')

        self.q_eval_z = self._build_net(self.s, 'eval_net', True)
        self.q_target_z = self._build_net(self.s_, 'target_net', False)

        # argmax(Q)
        max_a = tf.argmax(self.q_eval_z, axis=1)
        one_hot_max_a = tf.one_hot(max_a, self.a_dim)

        # y = R + gamma * Q_(S, argmax(Q))
        q_target = self.r + self.gamma \
            * tf.reduce_sum(one_hot_max_a * self.q_target_z, axis=1, keepdims=True) * (1 - self.done)
        q_target = tf.stop_gradient(q_target)

        q_eval = tf.reduce_sum(self.a * self.q_eval_z, axis=1, keepdims=True)

        self.td_error = tf.abs(q_target - q_eval)
        self.temp = tf.squared_difference(q_target, q_eval) * self.importance_ratio
        self.loss = tf.reduce_mean(tf.squared_difference(q_target, q_eval) * self.importance_ratio)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.param_target = tf.global_variables(scope='target_net')
        self.param_eval = tf.global_variables(scope='eval_net')

        self.target_replace_ops = [tf.assign(t, e) for t, e in zip(self.param_target, self.param_eval)]

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):  # sigmoid, tanh relu
            # l = tf.layers.dense(s, 20, activation=tf.nn.relu, trainable=trainable, ** initializer_helper)
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
        # print("loss", loss)
        return loss

    def _learn(self):
        points, (s, a, r, s_, done), importance_ratio = self.memory.get_mini_batches()

        td_error, _, loss = self.sess.run([self.td_error, self.optimizer, self.loss], feed_dict={
            self.s: s,
            self.a: a,
            self.r: r,
            self.s_: s_,
            self.done: done,
            self.importance_ratio: np.array([importance_ratio]).T
        })
        self.memory.update(points, td_error.squeeze(axis=1)) 
        return loss


    def input_source_memory_and_set_train_param(self, memory, batch_source, batch_target):
        self.source_memory = memory
        self.source_memory.batch_size = batch_source
        self.memory.batch_size = batch_target

    def learn_from_only_source_memory(self):

        if self._learn_step_counter % self.replace_target_iter == 0:
            
            self.sess.run(self.target_replace_ops)
        points_source, (s, a, r, s_, done), importance_ratio = self.source_memory.get_mini_batches()
        # print(np.array(points_source).shape)
        td_error_source, _, loss_source = self.sess.run([self.td_error, self.optimizer, self.loss], feed_dict={
            self.s: s,
            self.a: a,
            self.r: r,
            self.s_: s_,
            self.done: done,
            self.importance_ratio: np.array([importance_ratio]).T
        })
        self.source_memory.update_source(points_source, td_error_source.squeeze(axis=1))
        self.points_source = points_source
        self.weights_source = self.source_memory.get_weights(self.points_source)
        return loss_source

    def get_points_and_weights_only_source_memory(self):
        return self.points_source, self.weights_source

    def learn_from_only_target_memory(self):

        if self._learn_step_counter % self.replace_target_iter == 0:
            # 每replace_target_iter步更新一次eval_net参数
            self.sess.run(self.target_replace_ops)
        points_target, (s, a, r, s_, done), importance_ratio = self.memory.get_mini_batches()
        td_error, _, loss, temp, importance = self.sess.run(
            [self.td_error, self.optimizer, self.loss, self.temp, self.importance_ratio], feed_dict={
                self.s: s,
                self.a: a,
                self.r: r,
                self.s_: s_,
                self.done: done,
                self.importance_ratio: np.array([importance_ratio]).T
            })
        self.memory.update_target(points_target, td_error.squeeze(axis=1))
        self.points_target = points_target
        self.weights_target = self.memory.get_weights(self.points_target)
        return loss

    def get_points_and_weights_only_target_memory(self):
        return self.points_target, self.weights_target

    def learn_from_source_and_target_memory(self):
        if self._learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_ops)
        loss, loss_source = self._learn_in_two_memory()
        return loss, loss_source

    def _learn_in_two_memory(self):
        points_target, (s, a, r, s_, done), importance_ratio = self.memory.get_mini_batches()
        td_error, _, loss, temp, importance = self.sess.run([self.td_error, self.optimizer, self.loss, self.temp, self.importance_ratio], feed_dict={
            self.s: s,
            self.a: a,
            self.r: r,
            self.s_: s_,
            self.done: done,
            self.importance_ratio: np.array([importance_ratio]).T
        })
        self.memory.update_target(points_target, td_error.squeeze(axis=1))
        self.points_target = points_target
        # print(type(td_error))
        self.weights_target = self.memory.get_weights(self.points_target)

        points_source, (s, a, r,  s_, done), importance_ratio = self.source_memory.get_mini_batches()
        td_error_source, _, loss_source = self.sess.run([self.td_error, self.optimizer, self.loss], feed_dict={
            self.s: s,
            self.a: a,
            self.r: r,
            self.s_: s_,
            self.done: done,
            self.importance_ratio: np.array([importance_ratio]).T
        })
        self.source_memory.update_source(points_source, td_error_source.squeeze(axis=1))
        self.points_source = points_source
        # print(points_source)
        self.weights_source = self.source_memory.get_weights(self.points_source)
        return loss, loss_source

    def get_points_and_weights(self):
        return self.points_source, self.weights_source, self.points_target, self.weights_target

   
    def update_all_weight_in_two_memory(self):
        points_target, (s, a, r, s_, done), importance_ratio = self.memory.get_memory_all()
        td_error, loss = self.sess.run([self.td_error,  self.loss], feed_dict={
            self.s: s,
            self.a: a,
            self.r: r,
            self.s_: s_,
            self.done: done,
            self.importance_ratio: np.array([importance_ratio]).T
        })
        self.memory.update_target(points_target, td_error.squeeze(axis=1))

        points_source, (s, a, r, s_, done), importance_ratio = self.source_memory.get_memory_all()
        td_error_source, loss_source = self.sess.run([self.td_error, self.loss], feed_dict={
            self.s: s,
            self.a: a,
            self.r: r,
            self.s_: s_,
            self.done: done,
            self.importance_ratio: np.array([importance_ratio]).T
        })
        self.source_memory.update_source(points_source, td_error_source.squeeze(axis=1))

    def update_all_weight_in_target_memory(self):
        points_target, (s, a, r, s_, done), importance_ratio = self.memory.get_memory_all()
        td_error, loss = self.sess.run([self.td_error, self.loss], feed_dict={
            self.s: s,
            self.a: a,
            self.r: r,
            self.s_: s_,
            self.done: done,
            self.importance_ratio: np.array([importance_ratio]).T
        })
        # print("importance_ratio", importance_ratio)
        self.memory.update_target(points_target, td_error.squeeze(axis=1))

    def update_all_weight_in_source_memory(self):
        points_source, (s, a, r, s_, done), importance_ratio = self.source_memory.get_memory_all()
        td_error_source, loss_source = self.sess.run([self.td_error, self.loss], feed_dict={
            self.s: s,
            self.a: a,
            self.r: r,
            self.s_: s_,
            self.done: done,
            self.importance_ratio: np.array([importance_ratio]).T
        })
        self.source_memory.update_source(points_source, td_error_source.squeeze(axis=1))
