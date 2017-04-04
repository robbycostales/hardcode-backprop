import math
import random
import copy

class Net:
    def __int__(self, active_node_list, t_set, n_inputs):
        """
        :param active_node_list: nested list of node structure and type.
        1 = adaline, 2 = step, 3...
        :param t_set: nested list of sub-lists of (input + expected_output) length
        :param n_inputs: number of input values (NOT INCLUDING THRESHOLD)
        """
        self.t_set = t_set
        self.num_inputs = n_inputs
        self.nodes = copy.deepcopy(active_node_list)

        for i in active_node_list:      # i == layer
            for j in i:                 # j == node
                 self.nodes[i][j] = Node(j, 32)


        return 0


    def new_node_connections(self):
        return 0


    def train(self):
        return 0


    def test(self):
        return 0

class Node:
    def __init__(self, function, prev):
        print(0)



xor_set = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
xor_active_node_list = [[1, 1], [1]]
xor_num_inputs = 2

XOR = Net()