import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.minist import load_mnist
from PIL import Image

def step_function(x) :
    y = x > 0
    return y.astype(np.int)

def relu(x):
    return np.maximum(0, x);

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def _(c):
    print(c)

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def img_show(img) :
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten = True, normalize = False)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)
