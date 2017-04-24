import tensorflow as tf
import numpy as np

from mmi.dataset import MMIMNIST
from mmi.loss    import BinaryCrossEntropyLoss, BinaryAccuracy,\
                        L1Regularizer, L2Regularizer
from mmi.model   import Sigmoid, ReLU, Softmax, Convolution2D,\
                        MaxPooling2D,Dropout, Flatten, BatchNormalization,\
                        LinearLayer, BagLayerMax, Model

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

X_train = np.vstack((mnist.train.images, mnist.validation.images))\
          .reshape((-1,28,28))
X_test  = mnist.test.images.reshape((-1,28,28))
y_train = np.concatenate(( mnist.train.labels, mnist.validation.labels))
y_test  = mnist.test.labels 

train = MMIMNIST(X_train, y_train, seed=43)
test  = MMIMNIST(X_test,  y_test, seed=43)

train.create_dataset(2500,2500)
test.create_dataset(2500,2500)

layers = [
    Convolution2D(32, kernel_size=[5,5], strides=[1,1], padding='VALID'),
    BatchNormalization(),
    ReLU(),
    MaxPooling2D(),
    Dropout(0.5),
    Convolution2D(64, kernel_size=[5,5], strides=[1,1], padding='VALID'),
    BatchNormalization(),
    ReLU(),
    MaxPooling2D(),
    Dropout(0.5),
    Flatten(),
    LinearLayer(1024),
    ReLU(),
    Dropout(0.5),
    LinearLayer(100),
    BagLayerMax(),
    ReLU(),
    LinearLayer(100),
    BagLayerMax(),
    ReLU(),
    LinearLayer(1, init_type='he_uniform')

]

model = Model()
for layer in layers:
    model.add_layer(layer)

input_shape = {
    'x': {'shape':[28,28,1], 'sparse':False, 'type':tf.float32},
    'y': {'shape':[], 'sparse':False, 'type':tf.float32},
    'segment_ids': {'shape':[2], 'sparse':False, 'type':tf.int32}
}

model.compile_(input_shape, BinaryCrossEntropyLoss(),\
               callbacks=[BinaryAccuracy(), L2Regularizer(5e-4)],\
               optimizer='adam', learning_rate=1e-3,\
               debug_folder='debug/')

for epoch in range(200):
    for xb,mb_dim,yb,ib in train.get_minibatches(20, shuffle=True):
        model.train_on_batch(xb,mb_dim,yb,ib)
    if (epoch+1)%5 == 0:
        for xb,mb_dim,yb,ib in train.get_minibatches(200, shuffle=False):
            model.update_metrics(xb, mb_dim, yb, ib)
        train_stats = model.save_merged('train', epoch, True)
        for xb,mb_dim,yb,ib in test.get_minibatches(200, shuffle=False):
            model.update_metrics(xb, mb_dim, yb, ib)
        test_stats  = model.save_merged('test', epoch, True)
        print 'End of Epoch %03d' % (epoch + 1)
        print 'Train - Loss: %.3f Accuracy: %.3f' %\
              (train_stats['loss'],train_stats['binaryaccuracy'])
        print 'Test  - Loss: %.3f Accuracy: %.3f' %\
              (test_stats['loss'],test_stats['binaryaccuracy'])
        print '-'*35
model.save_model(epoch+1)
