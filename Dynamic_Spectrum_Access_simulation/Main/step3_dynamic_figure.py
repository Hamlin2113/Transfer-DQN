import os
import sys
path1=os.path.abspath('.')
path2=os.path.abspath('..')
sys.path.append(path2)
from Environments.Markov import env_Markov
import numpy as np


import numpy as np
import dill as pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Main.Core_functions import *
import os

FIGURE_SU = True

if FIGURE_SU:
    GIF_title = 'Spectrum waterfall'
else:
    GIF_title =  'Spectrum waterfall'

FIGURE_ENV_SOURCE_NOT_TARGET = False


tf.reset_default_graph()

if __name__ == '__main__':


    param_folder = os.path.abspath("..") + '\\Stores\\parameters\\'
    env_filename = "env_source" if FIGURE_ENV_SOURCE_NOT_TARGET else "env_target"
    with open(param_folder + env_filename + '.pkl', 'rb') as file:
        env = pickle.load(file)
        print("场景 ", env_filename, " 加载成功")
    with open(param_folder + 'TIME_UPPER_BOUND' + '.pkl', 'rb') as file:
        TIME_UPPER_BOUND = pickle.load(file)
        print("参数 TIME_UPPER_BOUND 加载成功")
    with open(param_folder + 'SLIP_WINDOW_SIZE' + '.pkl', 'rb') as file:
        SLIP_WINDOW_SIZE = pickle.load(file)
        print("参数 SLIP_WINDOW_SIZE 加载成功")
    with open(param_folder + 'eval_net_param' + '.pkl', 'rb') as file:
        eval_net_param = pickle.load(file)
    with open(param_folder + 'target_net_param' + '.pkl', 'rb') as file:
        target_net_param = pickle.load(file)
        print("[eval_net_param, target_net_param]网络参数 加载成功")

    session = tf.Session()
    memory_size_test = 512
    agent = Normal_DQN(sess=session, s_dim=env.n_features * SLIP_WINDOW_SIZE * env.sense_time,
                       a_dim=(env.n_features + 1) * TIME_UPPER_BOUND, batch_size=64, gamma=0.95, lr=0.01,
                       epsilon=0.05, replace_target_iter=100, memory_size=memory_size_test)
    with session.as_default():
        tf.global_variables_initializer().run()
    eval_param_replace = [t.assign(e) for t, e in zip(agent.param_eval, eval_net_param)]
    target_param_replace = [t.assign(e) for t, e in zip(agent.param_target, target_net_param)]
    session.run([eval_param_replace, target_param_replace])

    result_folder = os.path.abspath("..") + '\\Stores\\results\\'
    xls_writer = dataWriter(file_name=result_folder + 'dynamic_figure_average_counter.xlsx')
    xls_writer.create_sheet("average_test")

    bottom_color = np.array([248, 248, 255]) / 255
    PU_color = np.array([216, 216, 216]) / 255
    SU_color = np.array([250, 128, 114]) / 255
    SU_sense_color = np.array([255, 250, 205]) / 255
    n_channel = env.n_channel


    image_high = 50
    multiply_channel = 3
    image_width = multiply_channel * n_channel


    bottom_layer = np.ones([image_high, image_width, 3])
    for i in range(3):
        bottom_layer[:, :, i] = bottom_color[i]

    state_dim = env.n_features * SLIP_WINDOW_SIZE * env.sense_time
    sense_dim = env.n_features * env.sense_time
    s_windowed = np.zeros(state_dim)
    s_windowed_ = np.zeros(state_dim)
    s = env.reset()

    def make_action_and_get_states_and_step(env, agent):

        global s_windowed, state_dim, s_windowed_, s
        s_windowed = np.hstack([s_windowed[sense_dim:], s[:]])
        a = oneHot_to_action(agent.make_action(s_windowed), TIME_UPPER_BOUND)
        s_, r, done, _, actions, states = env.step_use_for_dynamic_figure(a)
        s_windowed_ = np.hstack([s_windowed[sense_dim:], s_[:]])
        s = s_
        return actions, states

    def env_action_2_image(current_image,
                           env_state, action,
                           mul_channel, rgb_color_list, under_color):

        num_channel = len(env_state)

        def set_color_and_fit_size(array, multi_channel, rgb, bottom_rgb):

            len_su = len(array)
            array = np.reshape(np.array([array for i in range(multi_channel)]).T, [len_su * multi_channel, 1])
            out_array = np.dot(array, np.reshape(np.array(rgb - bottom_rgb), [1, 3]))
            return out_array

        new_rgb_array = set_color_and_fit_size(np.ones([num_channel]), mul_channel, under_color, np.array([0, 0, 0]))
        new_rgb_array += set_color_and_fit_size(env_state, mul_channel, rgb_color_list[0], under_color)
        if FIGURE_SU:
            action_state = np.zeros(num_channel)
            if action < num_channel:
                action_state[action] = 1
                new_rgb_array += set_color_and_fit_size(action_state, mul_channel, rgb_color_list[1], under_color)
            if action == num_channel:
                action_state[:] = 1
                new_rgb_array += set_color_and_fit_size(action_state, mul_channel, rgb_color_list[2], under_color)
        else:
            pass
        current_image = np.insert(current_image, image_high, new_rgb_array, 0)
        current_image = np.delete(current_image, 0, 0)
        return current_image
    fig = plt.figure()
    plt.ion
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    ax = fig.add_subplot(1, 1, 1)
    plt.xlabel("Channel")
    plt.ylabel("Time slots")
    plt.title(GIF_title)
    plt.xticks(np.floor(np.linspace(0, image_width, n_channel + 1))+0.9, [i for i in range(1, n_channel+2)])
    plt.yticks(np.floor(np.linspace(0, image_high, 10)), [i for i in range(0,image_high, 5)])
    image_show = bottom_layer
    fig_agent_env = ax.imshow(image_show)
    pos = 0
    actions, states = make_action_and_get_states_and_step(env, agent)

    slots_time = 0
    success_time = 0
    fail_time = 0
    collision_time = 0
    sense_time = 0

    def average_counter(action, state):
        global slots_time, success_time, fail_time, collision_time, sense_time
        slots_time += 1
        if action == n_channel:
            sense_time += 1
            fail_time += 1
        elif state[action] == 1:
            collision_time += 1
            fail_time += 1
        else:
            success_time += 1
        success_rate = success_time / slots_time
        xls_writer.write_into_lable(slots_time, slots_time, "time_slots")
        xls_writer.write_into_lable(slots_time, success_time, "success_time")
        xls_writer.write_into_lable(slots_time, fail_time, "fail_time")
        xls_writer.write_into_lable(slots_time, collision_time, "collision_time")
        xls_writer.write_into_lable(slots_time, sense_time, "sense_time")

        xls_writer.write_into_lable(slots_time, success_rate, "success_rate")
        # print(sense_time)
        if collision_time != 0 or success_time != 0:
            collision_rate = (collision_time / (success_time + collision_time))
        else:
            collision_rate = 0
        xls_writer.write_into_lable(slots_time, collision_rate, "collision_rate")
        sense_use_ratio = sense_time / slots_time  # slots_time = (sense_time + success_time + collision_time)
        xls_writer.write_into_lable(slots_time, sense_use_ratio, "sense_use_ratio")
        if sense_use_ratio>0:
            use_sense_ratio = 1/sense_use_ratio
            print("success_rate: %.3f" % success_rate, " collision_rate: %.3f" % collision_rate,
                  " use_sense_ratio: %.3f" % use_sense_ratio)


    def updata(i):
        global image_show, env, actions, states, n_channel, bottom_color, PU_color, SU_color, pos
        pos += 1
        if pos == len(actions):
            pos = 0
            actions, states = make_action_and_get_states_and_step(env, agent)
        average_counter(actions[pos], states[pos])
        image_show = env_action_2_image(image_show, states[pos], actions[pos], multiply_channel,
                                        [PU_color, SU_color, SU_sense_color],
                                        bottom_color)
        fig_agent_env.set_data(image_show)
        return [fig_agent_env]

    ani = FuncAnimation(fig, updata, frames=np.arange(0, 10), interval=100)
    ani.save(result_folder+'average_couter_and_result_test.gif', writer='imagemagick')
    plt.show()
    xls_writer.close()
