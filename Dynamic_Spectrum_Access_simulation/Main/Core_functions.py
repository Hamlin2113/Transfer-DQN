import numpy as np
import tensorflow as tf
from DQN.dqn import DQN as Normal_DQN
from Transfer_DQN.memoryUtils import normal2Transfer, storeAllMemory,\
    storeBatchMemory, storeBatchSourceMemory, storeMemorySumValue
from Utils.excelWriter import dataWriter
import time

def oneHot_to_action(oneHot, TIME_UPPER_BOUND):

    return np.array([oneHot // TIME_UPPER_BOUND, oneHot % TIME_UPPER_BOUND + 1]) 


class result_xls_manager:
    

    def __init__(self, file_path, file_prefix):
        self.xls_writer = dataWriter(file_name=file_path + file_prefix + 'train_record.xlsx')
        self.xls_writer_for_loss = dataWriter(file_name=file_path + file_prefix + 'loss_record.xlsx')
        self.xls_writer_memory_source = dataWriter(file_name=file_path + file_prefix + 'memory_source_record.xlsx')
        self.xls_writer_memory_target = dataWriter(file_name=file_path + file_prefix + 'memory_target_record.xlsx')
        self.xls_counter = 0  

    def create_new_xls(self):
        self.xls_counter = self.xls_counter + 1

    def save_and_close(self):
        self.xls_writer.close()
        self.xls_writer_for_loss.close()
        self.xls_writer_memory_source.close()
        self.xls_writer_memory_target.close()


def train_DQN(sess,
              agent, env,
              record_steps, total_steps, epsilon_reduction_rate,
              SLIP_WINDOW_SIZE, TIME_UPPER_BOUND,
              xls_manager: result_xls_manager,
              close_sess=True, init_param=True, learn_or_use=True):

    print("sheet_name = ", 'train_%d' % xls_manager.xls_counter)
    xls_manager.xls_writer.create_sheet(sheet_name='train_%d' % xls_manager.xls_counter)
    xls_manager.xls_writer_for_loss.create_sheet(sheet_name='train_%d' % xls_manager.xls_counter)
           #
    state_dim = env.n_features * SLIP_WINDOW_SIZE * env.sense_time  
    sense_dim = env.n_features * env.sense_time

    with sess.as_default():
        tf.set_random_seed(1)
        if init_param:
            tf.global_variables_initializer().run()

        s_windowed = np.zeros(state_dim)
        s_windowed_ = np.zeros(state_dim)

        s = env.reset()  # --------core-①---------
        
        count_record = np.array([0, 0, 0, 0, 0])  
        time_steps = 0  
        average_reward = 0  
        action_t_not0_record = 0  
        action_t_not0_nums = 0  
        make_decision_average_time_cost = 0  

        while time_steps <= total_steps:
            
            s_windowed = np.hstack([s_windowed[sense_dim:], s[:]])  # -------core-②----------

            t_start = time.clock()
            a_oneHot = agent.choose_action(s_windowed)  # ---------core-③-----------
            t_end = time.clock()
            make_decision_average_time_cost += (t_end - t_start)  

            a = oneHot_to_action(a_oneHot, TIME_UPPER_BOUND)  # -----------core-④------------
            s_, r, done, _ = env.step(a)  # -----------core-⑤-------------------
            
            s_windowed_ = np.hstack([s_windowed[sense_dim:], s_[:]])  # -----------core-⑥----------
            
            if learn_or_use: 
                loss = agent.store_transition_and_learn(s_windowed, a_oneHot, r, s_windowed_, done)  # ---core-⑦----
                xls_manager.xls_writer_for_loss.write_into_lable(time_steps + 1, loss, 'loss')  

            count_record += env.count  
            average_reward += r  
            time_steps += 1  
            if a[0] < env.n_features and a[1] > 0:
                action_t_not0_record += a[1]
                action_t_not0_nums += 1
            if time_steps % record_steps == 0:
                agent.epsilon *= epsilon_reduction_rate
                action_t_not0_nums += 1
                print("epsilon: %-6.5f" % agent.epsilon, "\tstep times: %-5d" % time_steps,
                      "\t[success collision give_up pu_active total]: [%-3d %-3d %-3d %-3d %-4d]"
                      % (count_record[0], count_record[1], count_record[2], count_record[3], count_record[4]),
                      "\taverage_reward: %-7.2f" % (average_reward / record_steps), "\tuse_spectrum_average_t: %-3.2f" %
                      (action_t_not0_record / action_t_not0_nums), "\taverage_decision_time_cost: %-6.4f ms" %
                      ((make_decision_average_time_cost/record_steps)*1000))
                rows = time_steps//record_steps + 2  
                xls_manager.xls_writer.write_into_lable(rows, count_record[0], "success_num")
                xls_manager.xls_writer.write_into_lable(rows, count_record[1], "collision_num")
                xls_manager.xls_writer.write_into_lable(rows, count_record[2], "give_up_num")
                xls_manager.xls_writer.write_into_lable(rows, count_record[3], "pu_active_num")
                xls_manager.xls_writer.write_into_lable(rows, count_record[4], "total_num")
                xls_manager.xls_writer.write_into_lable(rows, (count_record[0]+count_record[3])/count_record[4], "occupy_rate")
                xls_manager.xls_writer.write_into_lable(rows, count_record[3]/count_record[4], "PU_occupy_rate")
                xls_manager.xls_writer.write_into_lable(rows, count_record[0]/(count_record[0]+count_record[1]+count_record[2]),"success_rate")
                xls_manager.xls_writer.write_into_lable(rows, count_record[1] / (count_record[0] + count_record[1] + count_record[2]), "collision_rate")
                xls_manager.xls_writer.write_into_lable(rows, agent.epsilon, "epsilon")
                xls_manager.xls_writer.write_into_lable(rows, time_steps, "step_times")
                xls_manager.xls_writer.write_into_lable(rows, average_reward/record_steps, "average_reward")
                xls_manager.xls_writer.write_into_lable(rows, action_t_not0_record / action_t_not0_nums, 'use_spectrum_average_t')
                
                count_record = np.array([0, 0, 0, 0, 0])
                action_t_not0_nums = 0
                action_t_not0_record = 0
                average_reward = 0
                make_decision_average_time_cost = 0  
            s = s_  # --------------------core-⑧--------------------
        if close_sess:
            sess.close()
        xls_manager.xls_counter = xls_manager.xls_counter + 1


def train_Tranfer(sess, agent,
                  record_step, total_steps, xls_manager: result_xls_manager,
                  close_sess=True, init_param=True):

    xls_manager.xls_writer_for_loss.create_sheet(sheet_name='two_memory_transfer_%d' % xls_manager.xls_counter)
    xls_manager.xls_writer_memory_source.create_sheet(sheet_name='memory_source')
    xls_manager.xls_writer_memory_target.create_sheet(sheet_name='memory_target')
    storeAllMemory(xls_manager.xls_writer_memory_source, agent.source_memory)
    storeAllMemory(xls_manager.xls_writer_memory_target, agent.memory)

    with sess.as_default():
        if init_param:
            tf.global_variables_initializer().run()
        time_steps = 0
        print("迁移学习训练开始：")
        time_cost_preTrain = 0
        while time_steps <= total_steps:
            if time_steps % record_step == 0:
                print("step: ", time_steps)
            t_start = time.clock()
            loss_target, loss_source = agent.learn_from_source_and_target_memory()  # -----------core----------
            t_end = time.clock()
            time_cost_preTrain += (t_end-t_start) 
            points_source, weights_source, points, weights = agent.get_points_and_weights()  
            xls_manager.xls_writer_for_loss.write_into_lable(time_steps + 2, loss_target, "loss_target")
            xls_manager.xls_writer_for_loss.write_into_lable(time_steps + 2, loss_source, "loss_source")
            storeBatchSourceMemory(xls_manager.xls_writer_memory_source, points_source, weights_source, time_steps)  
            storeBatchMemory(xls_manager.xls_writer_memory_target, points, weights, time_steps)  
            storeMemorySumValue(xls_manager.xls_writer_memory_target, agent.memory, time_steps)  
            storeMemorySumValue(xls_manager.xls_writer_memory_source, agent.source_memory, time_steps) 
            time_steps += 1
        if close_sess:
            sess.close()
        print("迁移学习训练结束，预训练消耗时长为 %7.3f s" % time_cost_preTrain)
        xls_manager.xls_counter += 1


def train_Transfer_only_source_memory(sess,  
                                      agent,  
                                      memory_source, 
                                      batch_source,
                                      record_step,  
                                      total_steps,  
                                      xls_manager: result_xls_manager,
                                      close_sess=True,  
                                      init_param=True  
                                      ):

    xls_manager.xls_writer_for_loss.create_sheet(sheet_name='two_memory_transfer_%d' % xls_manager.xls_counter)
    xls_manager.xls_writer_memory_source.create_sheet(sheet_name='memory_source')
    with sess.as_default():
        if init_param:
            tf.global_variables_initializer().run()
        agent.memory = memory_source  # ----------代码①-------
        agent.memory.batch_size = batch_source
        print("batch_source: ", agent.memory.batch_size)
        print("经验池权值初始化中...")  
        # agent.update_all_weight_in_target_memory()  
        storeAllMemory(xls_manager.xls_writer_memory_source, agent.memory)
        print("经验池权值初始化完成，仅源经验池学习开始...")
        time_steps = 0
        time_cost_preTrain = 0
        while time_steps <= total_steps:
            if time_steps % record_step == 0:
                print("step: ", time_steps)
            t_start = time.clock()
            loss = agent.learn_from_only_target_memory()  # ----core------
            t_end = time.clock()
            time_cost_preTrain += (t_end-t_start)  
            points, weights = agent.get_points_and_weights_only_target_memory()  
            xls_manager.xls_writer_for_loss.write_into_lable(time_steps + 2, loss, "weights_source")
            storeBatchMemory(xls_manager.xls_writer_memory_source, points, weights, time_steps)  
            storeMemorySumValue(xls_manager.xls_writer_memory_source, agent.memory, time_steps)  
            time_steps += 1
        if close_sess:
            sess.close()
        print("迁移学习训练结束，预训练消耗时长为 %7.3f s" % time_cost_preTrain)
        xls_manager.xls_counter += 1


def train_Transfer_only_target_memory(sess,  
                                      agent,  
                                      memory_target,  
                                      batch_target,
                                      record_step,  
                                      total_steps, 
                                      xls_manager: result_xls_manager,
                                      close_sess=True,  
                                      init_param=True  
                                      ):
   

    xls_manager.xls_writer_for_loss.create_sheet(sheet_name='two_memory_transfer_%d' % xls_manager.xls_counter)
    xls_manager.xls_writer_memory_target.create_sheet(sheet_name='memory_target')

    with sess.as_default():
        if init_param:
            tf.global_variables_initializer().run()
      
        agent.memory = memory_target  # ---------代码①------------
        agent.memory.batch_size = batch_target
        print("batch_target: ", agent.memory.batch_size)
        print("经验池权值初始化中...")
        # agent.update_all_loss_in_target_memory()  
        storeAllMemory(xls_manager.xls_writer_memory_target, agent.memory)
        print("经验池权值初始化完成，仅目标经验池学习开始...")
        time_steps = 0
        time_cost_preTrain = 0
        while time_steps <= total_steps:
            if time_steps % record_step == 0:
                print("step: ", time_steps)
            t_start = time.clock()
            loss = agent.learn_from_only_target_memory()  # ----------core-------------
            t_end = time.clock()
            time_cost_preTrain += (t_end-t_start)  
            points, weights = agent.get_points_and_weights_only_target_memory() 
            xls_manager.xls_writer_for_loss.write_into_lable(time_steps + 2, loss, "weights_target")
            storeBatchMemory(xls_manager.xls_writer_memory_target, points, weights, time_steps)  
            storeMemorySumValue(xls_manager.xls_writer_memory_target, agent.memory, time_steps)  
            time_steps += 1
        if close_sess:
            sess.close()
        print("迁移学习训练结束，预训练消耗时长为 %7.3f s" % time_cost_preTrain)
        xls_manager.xls_counter += 1


def getMemoryPool(memory_size,
                  env,
                  memory_explore_time,
                  TIME_UPPER_BOUND,  
                  SLIP_WINDOW_SIZE,   
                  xls_manager
                  ):

    tf.reset_default_graph()
    sess = tf.Session()
    
    print('探索已获取经验池样本（探索次数：%d）' % memory_explore_time)
    agent = Normal_DQN(  
        sess=sess,
        s_dim=env.n_features * SLIP_WINDOW_SIZE * env.sense_time,

        a_dim=(env.n_features + 1) * TIME_UPPER_BOUND,  
        batch_size=64,
        gamma=0.95,
        lr=0.01,
        epsilon=1,  # ---------------core-1----------------
        replace_target_iter=100,
        memory_size=memory_size
    )
    # ---------core-2------------
    train_DQN(sess, agent, env, record_steps=1000, total_steps=memory_explore_time, epsilon_reduction_rate=1,
              SLIP_WINDOW_SIZE=SLIP_WINDOW_SIZE, TIME_UPPER_BOUND=TIME_UPPER_BOUND, xls_manager=xls_manager
              )
    return agent.memory


def train_Tranfer_batch_changes(sess, agent,
                  record_step, total_steps, batch_changes_step, batchs, xls_manager: result_xls_manager,
                  close_sess=True, init_param=True):
    xls_manager.xls_writer_for_loss.create_sheet(sheet_name='two_memory_transfer_%d' % xls_manager.xls_counter)
    xls_manager.xls_writer_memory_source.create_sheet(sheet_name='memory_source')
    xls_manager.xls_writer_memory_target.create_sheet(sheet_name='memory_target')
    storeAllMemory(xls_manager.xls_writer_memory_source, agent.source_memory)
    storeAllMemory(xls_manager.xls_writer_memory_target, agent.memory)

    with sess.as_default():
        if init_param:
            tf.global_variables_initializer().run()
        time_steps = 0
        point_batch = -1  
        print("迁移学习训练开始：")
        while time_steps <= total_steps:
            if time_steps % batch_changes_step == 0 and point_batch < (len(batchs)-1):
                point_batch += 1
                batch_source = batchs[point_batch][0]
                batch_target = batchs[point_batch][1]
                agent.source_memory.batch_size = batch_source
                agent.memory.batch_size = batch_target
            if time_steps % record_step == 0: 
                print("step: %-5d, batch_source: %-3d, batch_target: %-3d" % (time_steps, agent.source_memory.batch_size, agent.memory.batch_size))
            if batch_source == 0 and batch_target == 0:  
                pass
            elif batch_target == 0:  
                loss_source = agent.learn_from_only_source_memory()
                points_source, weights_source = agent.get_points_and_weights_only_source_memory()
                xls_manager.xls_writer_for_loss.write_into_lable(time_steps + 2, loss_source, "loss_source")
                storeBatchSourceMemory(xls_manager.xls_writer_memory_source, points_source, weights_source, time_steps)
                storeMemorySumValue(xls_manager.xls_writer_memory_source, agent.source_memory, time_steps)  
            elif batch_source == 0:  
                loss_target = agent.learn_from_only_target_memory()
                points_target, weights_target = agent.get_points_and_weights_only_target_memory()
                xls_manager.xls_writer_for_loss.write_into_lable(time_steps + 2, loss_target, "loss_target")
                storeBatchMemory(xls_manager.xls_writer_memory_target, points_target, weights_target, time_steps)
                storeMemorySumValue(xls_manager.xls_writer_memory_target, agent.memory, time_steps)  
        
            else: 
                loss_target, loss_source = agent.learn_from_source_and_target_memory()  # -----------core----------

                points_source, weights_source, points, weights = agent.get_points_and_weights()  
                xls_manager.xls_writer_for_loss.write_into_lable(time_steps + 2, loss_target, "loss_target")
                xls_manager.xls_writer_for_loss.write_into_lable(time_steps + 2, loss_source, "loss_source")

                storeBatchSourceMemory(xls_manager.xls_writer_memory_source, points_source, weights_source, time_steps) 
                storeBatchMemory(xls_manager.xls_writer_memory_target, points, weights, time_steps)  
                storeMemorySumValue(xls_manager.xls_writer_memory_target, agent.memory, time_steps)  
                storeMemorySumValue(xls_manager.xls_writer_memory_source, agent.source_memory, time_steps)  
            time_steps += 1
        if close_sess:
            sess.close()
        print("迁移学习训练结束。")
        xls_manager.xls_counter += 1