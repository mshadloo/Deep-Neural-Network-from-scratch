import tensorflow as tf

def linear(x):
    return x
activation_mapping = {"sigmoid":tf.nn.sigmoid,"relu":tf.nn.relu,"tanh":tf.nn.tanh,"linear":linear}
# def Input(dim):
#     return tf.placeholder(tf.float64, [dim, None])
#
#
# class Dense:
#
#     def __init__(self, units_num, activation = "linear"):
#         self.units_num,  self.activation = units_num,  activation
#     def __call__(self, inp):
#         n_x = inp.shape[0].value
#         W = tf.get_variable(shape=[self.units_num, n_x],
#                                                initializer=tf.contrib.layers.xavier_initializer(seed=1))
#         b = tf.get_variable(shape= [self.units_num, 1],
#                                                initializer=tf.zeros_initializer())
#         Z = tf.add(tf.matmul(W,inp),b)
#         if self.activation =="softmax":
#             A = tf.nn.softmax(Z,axis=0)
#         else:
#             A = activation_mapping[self.activation](Z)
#
#         return A

class Input:
    def __init__(self, shape):
        self.shape = shape
class Dense:
    def __init__(self, units_num, activation = "linear"):
        self.units_num, self.activation = units_num, activation
        self.parameters = {}
    def setVariables(self,W_name, b_name,n_l_1):
        W = tf.get_variable(W_name, [self.units_num,n_l_1],
                                                        initializer=tf.contrib.layers.xavier_initializer(seed=1))
        b = tf.get_variable(b_name, [self.units_num, 1], initializer=tf.zeros_initializer())
        self.parameters["W"] = W
        self.parameters["b"] = b
        return W,b