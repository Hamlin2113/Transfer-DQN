import os
import sys
path1=os.path.abspath('.')
path2=os.path.abspath('..')
sys.path.append(path2)
from Environments.Markov import env_Markov
import numpy as np


from Main.Core_functions import *
import os
import dill as pickle
from Transfer_DQN.dqn import DQN as Transfer_DQN

import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


result_folder = os.path.abspath("..") + '\\Stores\\results\\'
param_folder = os.path.abspath("..") + '\\Stores\\parameters\\'
xls_manager = result_xls_manager(result_folder, "transfer_")

transfer_batch_source = 40
transfer_batch_target = 160
replace_target_iter = 100
transfer_record_step = 1000
transfer_total_step = 20*1000




memory_size_test = 512
batch_size = 64
record_steps = 20
total_steps = 20*2000
init_explore_rate = 0.05
explore_reduce_rate = 0.998

save_agent_preTrain_or_final = True


if __name__ == '__main__':

    with open(param_folder + 'memory_source' + '.pkl', 'rb') as file:
        memory_source = pickle.load(file)
        print("源经验池 memory_source 加载成功")
    with open(param_folder + 'memory_target' + '.pkl', 'rb') as file:
        memory_target = pickle.load(file)
        print("目标经验池 memory_target 加载成功")
    with open(param_folder + 'env_target' + '.pkl', 'rb') as file:
        env_target = pickle.load(file)
        print("目标场景 env_target 加载成功")
    with open(param_folder + 'TIME_UPPER_BOUND' + '.pkl', 'rb') as file:
        TIME_UPPER_BOUND = pickle.load(file)
        print("参数 TIME_UPPER_BOUND 加载成功")
    with open(param_folder + 'SLIP_WINDOW_SIZE' + '.pkl', 'rb') as file:
        SLIP_WINDOW_SIZE = pickle.load(file)
        print("参数 SLIP_WINDOW_SIZE 加载成功")

    tf.reset_default_graph()
    sess = tf.Session()


    agent_transfer_learn = Transfer_DQN(sess=sess,
                                        s_dim=env_target.n_features * SLIP_WINDOW_SIZE * env_target.sense_time,
                                        a_dim=(env_target.n_features + 1) * TIME_UPPER_BOUND,
                                        batch_size=64, gamma=0.95, lr=0.01, epsilon=1,
                                        replace_target_iter=replace_target_iter,
                                        memory_size=memory_size_test)  # ------core-1--------------------
    # --core-2----
    agent_transfer_learn.memory = memory_target
    agent_transfer_learn.input_source_memory_and_set_train_param(memory=memory_source,
                                                                 batch_source=transfer_batch_source,
                                                                 batch_target=transfer_batch_target)
    print("batch_source : batch_target = ",
          agent_transfer_learn.source_memory.batch_size, " : ", agent_transfer_learn.memory.batch_size)
    train_Tranfer(sess, agent_transfer_learn,
                  record_step=transfer_record_step, total_steps=transfer_total_step,
                  xls_manager=xls_manager,
                  close_sess=False, init_param=True)  # ------core-3----------------

    # ------------------core-4-------------------------
    eval_net_param = [sess.run(t) for t in agent_transfer_learn.param_eval]
    target_net_param = [sess.run(t) for t in agent_transfer_learn.param_target]
    if save_agent_preTrain_or_final:
        with open(param_folder + 'eval_net_param' + '.pkl', 'wb') as file:
            pickle.dump(eval_net_param, file)
        with open(param_folder + 'target_net_param' + '.pkl', 'wb') as file:
            pickle.dump(target_net_param, file)
            print("双经验池迁移预学习结果：[eval_net_param, target_net_param]网络参数保存成功")

    sess.close()
    tf.reset_default_graph()
    sess = tf.Session()

    agent_target_env = Normal_DQN(sess=sess,
                              s_dim=env_target.n_features * SLIP_WINDOW_SIZE * env_target.sense_time,
                              a_dim=(env_target.n_features + 1) * TIME_UPPER_BOUND,
                              batch_size=batch_size, gamma=0.95, lr=0.01, epsilon=init_explore_rate,
                              replace_target_iter=100, memory_size=memory_size_test)
    with sess.as_default():
        tf.global_variables_initializer().run()
    eval_param_replace = [t.assign(e) for t, e in zip(agent_target_env.param_eval, eval_net_param)]
    target_param_replace = [t.assign(e) for t, e in zip(agent_target_env.param_target, target_net_param)]
    sess.run([eval_param_replace, target_param_replace])
    train_DQN(sess, agent_target_env, env_target, record_steps=record_steps, total_steps=total_steps,
              epsilon_reduction_rate=explore_reduce_rate,
              SLIP_WINDOW_SIZE=SLIP_WINDOW_SIZE, TIME_UPPER_BOUND=TIME_UPPER_BOUND,
              xls_manager=xls_manager, close_sess=False, init_param=False)

    if not save_agent_preTrain_or_final:
        eval_net_param = [sess.run(t) for t in agent_target_env.param_eval]
        target_net_param = [sess.run(t) for t in agent_target_env.param_target]  # ------core-4------
        with open(param_folder + 'eval_net_param' + '.pkl', 'wb') as file:
            pickle.dump(eval_net_param, file)
        with open(param_folder + 'target_net_param' + '.pkl', 'wb') as file:
            pickle.dump(target_net_param, file)
            print("双经验池迁移学习最终结果：[eval_net_param, target_net_param]网络参数保存成功")

    sess.close()
    xls_manager.save_and_close()

import winsound
winsound.Beep(1500, 1000)

print('Step2 完成transfer test')

