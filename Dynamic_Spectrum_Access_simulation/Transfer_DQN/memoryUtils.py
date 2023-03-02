import numpy as np
import math
from DQN.memory import Memory as normal_memory
from Transfer_DQN.memory import Memory as transfer_memory
from Utils.excelWriter import dataWriter




def normal2Transfer(source_memory: normal_memory, beta):
    memory = transfer_memory(source_memory.batch_size, source_memory.max_size, beta)
    stored_memory = source_memory.get_memory_deque()
    while len(stored_memory) != 0:
        out = stored_memory.popleft()
        s, a, r, s_, done = out
        memory.store_transition_with_prior_one(s, a, r, s_, done)
    return memory


def storeAllMemory(xls_writer: dataWriter, memory: transfer_memory):
    tree = memory._sum_tree.tree

    datas = memory._sum_tree.data
    # print(type(datas))
    capacity = memory._sum_tree.capacity
    # print(tree[capacity:-1])

    for i in range(capacity):
        xls_writer.write_into_lable(0, datas[i].__str__(), str(i))
        xls_writer.write_into_lable(1, tree[capacity - 1 + i], str(i))


def storeBatchMemory(xls_writer: dataWriter, points, weights, row):
    for i in range(len(points)):
        xls_writer.write_into_lable(row+2, weights[i], str(points[i]))


def storeBatchSourceMemory(xls_writer: dataWriter, points, weights, row):
    for i in range(len(points)):
        xls_writer.write_into_lable(row+2, weights[i], str(points[i]))


def storeMemorySumValue(xls_writer: dataWriter, memory: transfer_memory, row):
    tree = memory._sum_tree.tree
    xls_writer.write_into_lable(row+2, tree[0], 'sum_memory')
    # print("sum_memory's value: ", row)