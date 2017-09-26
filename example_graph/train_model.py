import tensorflow as tf
import numpy as np
import h5py

from mmi.citation import CitationDataset
from mmi.loss     import SoftmaxCrossEntropyLoss, SoftmaxAccuracy,\
                         L1Regularizer, L2Regularizer
from mmi.model    import Sigmoid, ReLU, Softmax, Convolution2D,\
                         MaxPooling2D,Dropout, Flatten, BatchNormalization,\
                         LinearLayer, BagLayerMax, Model, SparseLinearLayer

data  = np.load('cora.npy')[()]
vocabulary = np.load('glove.npy')[()]
splits = h5py.File('splits.h5')

from nltk.corpus import stopwords
sw = stopwords.words('english')

for s in sw:
    if s in vocabulary:
        del vocabulary[s]  

type_ = 'mmi'
seg_dim = 1
if type_ == 'mmi':
    seg_dim += 1

for k in range(len(splits['train'])):

    layers = [
        LinearLayer(2000),
        BagLayerMax(),
        ReLU(),
        LinearLayer(2000),
        BagLayerMax(),
        ReLU(),
        LinearLayer(7)
    ]
    
    model = Model()
    for layer in layers:
        model.add_layer(layer)
    
    input_shape = {
        'x': {'shape':[300], 'sparse':False, 'type':tf.float32},
        'y': {'shape':[], 'sparse':False, 'type':tf.float32},
        'segment_ids': {'shape':[seg_dim], 'sparse':False, 'type':tf.int32}
    }
    
    model.compile_(input_shape, SoftmaxCrossEntropyLoss(),\
                   callbacks=[SoftmaxAccuracy()],\
                   optimizer='adam', learning_rate=1e-4,\
                   debug_folder='debug/')

    train_indices = splits['train/split_%d' % k][:]
    test_indices = splits['test/split_%d' % k][:]
    val_indices = splits['val/split_%d' % k][:]

    train = CitationDataset(data, vocabulary, train_indices, type_=type_, include_node=True)
    test  = CitationDataset(data, vocabulary, test_indices,  type_=type_, include_node=True)
    val   = CitationDataset(data, vocabulary, val_indices,   type_=type_, include_node=True)

    for epoch in range(400):
        for xb,mb_dim,yb,ib in train.get_minibatches(10, shuffle=True):
            model.train_on_batch(xb,mb_dim,yb,ib)
        if (epoch+1)%5 == 0:
            for xb,mb_dim,yb,ib in train.get_minibatches(200, shuffle=False):
                model.update_metrics(xb, mb_dim, yb, ib)
            train_stats = model.save_merged('train', epoch, True)
            for xb,mb_dim,yb,ib in test.get_minibatches(200, shuffle=False):
                model.update_metrics(xb, mb_dim, yb, ib)
            test_stats  = model.save_merged('test', epoch, True)
            print('End of Epoch %03d' % (epoch + 1))
            print('Train - Loss: %.3f Accuracy: %.3f' %\
                  (train_stats['loss'],train_stats['softmaxaccuracy']))
            print('Test  - Loss: %.3f Accuracy: %.3f' %\
                  (test_stats['loss'],test_stats['softmaxaccuracy']))
            print('-'*35)
    model.save_model(epoch+1)
