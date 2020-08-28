import tensorflow as tf
mnist = tf.keras.datasets.mnist
from utils import convert_to_one_hot
from model import Model
from layers import Input,Dense
(train_set_x_orig, Y_train_orig), (test_set_x_orig, Y_test_orig) = mnist.load_data()
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

#normalize:
X_train = train_set_x_flatten/255.
X_test = test_set_x_flatten/255.
print(max(Y_train_orig))
print(Y_train_orig.shape)
Y_train = convert_to_one_hot(Y_train_orig, 10)
Y_test = convert_to_one_hot(Y_test_orig, 10)
print(Y_train.shape)


#model:
inp  = Input(X_train.shape[0])
h = Dense(512,"relu")
out = Dense(10,"softmax")

model = Model(inp,[h,out])
model.fit(X_train, Y_train, X_test, Y_test, loss="categorical_crossentropy", learning_rate=0.01)