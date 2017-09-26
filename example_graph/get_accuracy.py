import os
import numpy as np
import h5py
import tensorflow as tf
from natsort import natsorted
from nltk.corpus import stopwords

from mmi.model    import Model
from mmi.citation import CitationDataset
from mmi.loss     import SoftmaxCrossEntropyLoss, SoftmaxAccuracy,\
                         L2Regularizer

data  = np.load('cora.npy')[()] 
vocabulary = np.load('glove.npy')[()] 
splits = h5py.File('splits.h5')
sw = stopwords.words('english')
type_ = 'mmi'
seg_dim = 2 

for s in sw: 
    if s in vocabulary:
        del vocabulary[s]


acc = []
for k,f in enumerate(natsorted(os.listdir('debug/'))):
    test_indices = splits['test/split_%d' % k][:]
    test  = CitationDataset(data, vocabulary, test_indices,  type_=type_, include_node=True)
    m = Model()
    m.load_from_json('./debug/%s/model.json' % f)
    input_shape = { 'x': {'shape':[300], 'sparse':False, 'type':tf.float32},\
                    'y': {'shape':[], 'sparse':False, 'type':tf.float32},\
                    'segment_ids': {'shape':[seg_dim], 'sparse':False, 'type':tf.int32} }
    m.compile_(input_shape, SoftmaxCrossEntropyLoss(),\
               callbacks=[SoftmaxAccuracy()],optimizer='adam',\
               learning_rate=1e-4,debug_folder='debug2/')
    m.restore('debug/%s/logs/model-401' % f)
    for xb,mb_dim,yb,ib in test.get_minibatches(200, shuffle=False):
        m.update_metrics(xb, mb_dim, yb, ib)
    test_stats  = m.save_merged('test', '444', True)
    acc += [test_stats['softmaxaccuracy']]
print(np.mean(acc), np.std(acc))
