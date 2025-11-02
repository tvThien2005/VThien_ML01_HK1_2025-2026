import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import random as rn
print(f"tensorFlow: {tf.__version__}")
# print(f"Keras: {keras.__version__}")
(x_train,y_train), (x_text,y_text) = keras.datasets.fashion_mnist.load_data()
print(type(x_train))
# x_train = x_train.reshape(60000,28*28).astype("float32")/255
print(x_train.shape,y_train.shape)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(class_names[y_train[2]])

# plt.imshow(x_train[0])
# plt.show()

def load_data(x_data , y_data, ):
    rows, cols = 6,6
    fig , axes = plt.subplots(rows,cols,figsize=(10,8))
    len = x_data.shape[0]
    for i in range(rows * cols):
        # index = rn.randint(0,len-1)
        ax = axes[i//cols,i%cols]
        img = rn.randint(0,len)
        ax.imshow(x_data[img],cmap='gray')
        # hiển thị tên lớp
        ax.set_title(class_names[y_data[img]],fontsize=10,color='green')
       
        # tắt trục toa độ
        ax.axis('off')
load_data(x_train,y_train)
plt.show()


# print(y_train[0])