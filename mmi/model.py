import json
import tensorflow as tf
import numpy as np
import time
import os

from mmi.loss import Regularizer

primitives = (bool, str, int, float, list, dict)

def get_inizializer(name, fan_in, fan_out):
    if name == 'he_normal':
        return tf.random_normal_initializer(mean=0, stddev=(np.sqrt(2.0)/np.sqrt(1.*fan_in)))
    if name == 'he_uniform':
        return tf.random_uniform_initializer(
                minval=(-np.sqrt(6.0)/np.sqrt(1.*fan_in + fan_out)), 
                maxval=( np.sqrt(6.0)/np.sqrt(1.*fan_in + fan_out)), 
                dtype=tf.float32)

def conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID'):
    return tf.nn.conv2d(x, W, strides=strides, padding=padding)

def max_pool_2x2(x, kernel_size=[1,2,2,1], strides=[1,2,2,1], padding='VALID'):
    return tf.nn.max_pool(x, ksize=kernel_size, strides=strides, padding=padding)

class Layer(object):
    def activation(self, *args, **kwargs):
        raise NotImplementedError( 'Should have implemented this' )
    def build(*args, **kwargs):
        raise NotImplementedError( 'Should have implemented this' )
    def get_summaries(self):
        raise NotImplementedError( 'Should have implemented this' )
    def get_debug(self):
        raise NotImplementedError( 'Should have implemented this' )
    def get_json(self):
        if not hasattr(self, 'name'):
            return {}
        json_layer = {}
        json_layer['layer'] = self.__class__.__name__
        for attrname in vars(self):
            if isinstance(getattr(self, attrname), primitives):
                json_layer[attrname] = getattr(self, attrname)
        return json_layer

class Activation(Layer):
    def __init__(self, *args, **kwargs):
        self.reduce_length = False
        self.activation_ = None

    def build(self, *args, **kwargs):
        # args[0] is the name
        # args[1] is the input_shape
        # args[2] is the input
        assert len(args) == 3
        self.name = args[0]
        self.input_shape = args[1]
        self.output_shape = self.input_shape
        return self.activation(args[2])

    def get_json(self):
        if not hasattr(self, 'name'):
            return {}
        json_layer = {}
        json_layer['layer'] = self.__class__.__name__
        for attrname in vars(self):
            if isinstance(getattr(self, attrname), primitives):
                json_layer[attrname] = getattr(self, attrname)
        return json_layer


    def get_summaries(self):
        return []
    
    def get_activation(self):
        return self.activation_


class Sigmoid(Activation):
    def activation(self, input_):
        self.activation_ = tf.sigmoid(input_)
        return self.activation_

class Softmax(Activation):
    def activation(self, input_):
        self.activation_ = tf.nn.softmax(input_)
        return self.activation_

class ReLU(Activation):
    def activation(self, input_):
        self.activation_ = tf.nn.relu(input_)
        return self.activation_
    
class PReLU(Activation):
    def build(self, *args, **kwargs):
        assert len(args) == 3
        self.name = args[0]
        self.input_shape = args[1]
        self.output_shape = self.input_shape
        input_ = args[2]
        self.alphas = tf.get_variable(self.name + '/alpha', self.input_shape,
                       initializer=tf.constant_initializer(0.0),
                       dtype=tf.float32)
        return self.activation(input_)

    def activation(self, input_):
        pos = tf.nn.relu(input_)
        neg = self.alphas * (input_ - abs(input_)) * 0.5
        self.activation_ = pos + neg
        return self.activation_

class Dropout(Activation):
    def __init__(self, *args, **kwargs):
        super(Dropout, self).__init__()
        assert ('keep_proba' in kwargs) != ( len(args) == 1)
        if len(args) == 1:
            self.keep_proba = args[0]
        else:
            self.keep_proba = kwargs['keep_proba']

    def build(self, *args, **kwargs):
        assert len(args) == 3
        assert 'is_training' in kwargs
        self.name = args[0]
        self.input_shape = args[1]
        self.output_shape = self.input_shape
        input_ = args[2]
        is_training = kwargs['is_training']
        return self.activation(input_, is_training)

    def activation(self, input_, is_training):
        self.activation_ = tf.cond(is_training, lambda: tf.nn.dropout(input_,self.keep_proba), lambda: tf.nn.dropout(input_, 1.0))
        return self.activation_

class BatchNormalization(Activation):
    def build(self, *args, **kwargs):
        assert len(args) == 3
        assert 'is_training' in kwargs
        self.name = args[0]
        self.input_shape = args[1]
        self.output_shape = self.input_shape
        input_ = args[2]
        is_training = kwargs['is_training']
        return self.activation(input_, is_training)

    def activation(self, input_, is_training):
        self.activation_ = tf.contrib.layers.batch_norm(input_, scope='batch_'+self.name, is_training=is_training)
        return self.activation_

class BagLayerMax(Activation):
    def __init__(self, *arg, **kwargs):
        super(BagLayerMax, self).__init__()
        self.reduce_length = True

    def build(self, *args, **kwargs):
        assert len(args) == 3
        assert 'lengths' in kwargs
        self.name = args[0]
        self.input_shape = args[1]
        self.output_shape = self.input_shape
        input_ = args[2]
        lengths = kwargs['lengths']
        return self.activation(input_, lengths)

    def activation(self, input_, lengths):
        self.activation_ = tf.segment_max(input_, lengths)
        return self.activation_

class BagLayerSum(Activation):
    def __init__(self, *args, **kwargs):
        super(BagLayerSum, self).__init__()
        self.reduce_length = True

    def build(self, *args, **kwargs):
        assert len(args) == 3
        assert 'lengths' in kwargs
        self.name = args[0]
        self.input_shape = args[1]
        self.output_shape = self.input_shape
        input_ = args[2]
        lengths = kwargs['lengths']
        return self.activation(input_, lengths)

    def activation(self, input_, lengths):
        self.activation_ = tf.segment_sum(input_, lengths)
        return self.activation_

class LinearLayer(Layer):
    def __init__(self, *args, **kwargs):
        super(LinearLayer, self).__init__()

        assert (len(args) == 1) != ('units' in kwargs)
        if len(args) == 1:
            self.units = args[0]
        else:
            self.units = kwargs['units']
        if 'init_type' in kwargs:
            self.init_type = kwargs['init_type']
        else:
            self.init_type = 'he_normal'
        self.reduce_length = False

        
    def build(self, *args, **kwargs):
        assert len(args) == 3
        self.name = args[0]
        self.input_shape = args[1]
        self.output_shape = self.input_shape[:-1] + [self.units]
        input_ = args[2]

        fan_in = self.input_shape[-1]
        fan_out = self.output_shape[-1]

        with tf.variable_scope(self.name):
            if self.init_type == 'identity':
                initializer = get_inizializer('he_normal', fan_in, fan_out)
                self.weights = tf.get_variable('weights', [fan_in, fan_out], initializer=initializer)
                self.weights = tf.add(self.weights, np.eye(fan_in, fan_out))
            else:
                initializer = get_inizializer(self.init_type, fan_in, fan_out)
                self.weights = tf.get_variable('weights', [fan_in, fan_out], initializer=initializer)
            self.bias    = tf.get_variable('bias', [fan_out], initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            self.summary_weights = tf.summary.histogram(self.name + '/weights', self.weights)
            self.summary_bias = tf.summary.histogram(self.name + '/bias', self.bias)
        return self.activation(input_)
        
    def activation(self, input_):
        self.activation_ = tf.add(tf.matmul(input_, self.weights), self.bias)        
        return self.activation_
    
    def get_summaries(self): 
        return [self.summary_weights, self.summary_bias]
    
    def get_activation(self):
        return self.activation_

class SparseLinearLayer(LinearLayer):
    def activation(self, input_):
        self.activation_ = tf.add(tf.sparse_tensor_dense_matmul(input_, self.weights), self.bias)
        return self.activation_   

class Convolution2D(Layer):
    def __init__(self, *args, **kwargs):


        assert (len(args) == 1) != ('output_channels' in kwargs)
        if len(args) == 1:
            self.output_channels = args[0]
        else:
            self.output_channels = kwargs['output_channels']
        if 'init_type' in kwargs:
            self.init_type = kwargs['init_type']
        else:
            self.init_type = 'he_normal'
        if 'kernel_size' in kwargs:
            self.kernel_size = kwargs['kernel_size']
        else:
            self.kernel_size = [3,3]
        if 'strides' in kwargs:
            self.strides = [1] + kwargs['strides'] + [1]
        else:
            self.strides = [1,1,1,1]
        if 'padding' in kwargs:
            self.padding = kwargs['padding']
        else:
            self.padding = 'VALID'

        self.reduce_length = False
        self.activation_ = None
    
    def build(self, *args, **kwargs):
        assert len(args) == 3
        self.name = args[0]
        self.input_shape = args[1]
        input_ = args[2]


        if self.padding == 'VALID':
            out_height = np.ceil(float(self.input_shape[0] - self.kernel_size[0] + 1) / float(self.strides[1]))
            out_width  = np.ceil(float(self.input_shape[1] - self.kernel_size[1] + 1) / float(self.strides[2]))
        if self.padding == 'SAME':
            out_height = np.ceil(float(self.input_shape[0]) / float(self.strides[1]))
            out_width  = np.ceil(float(self.input_shape[1]) / float(self.strides[2]))
        self.output_shape = [out_height, out_width, self.output_channels]
        self.input_channels = self.input_shape[-1]

        is_uniform = self.init_type == 'he_uniform'
        with tf.variable_scope(self.name):
            self.weights = tf.get_variable('weights', self.kernel_size + [self.input_channels, self.output_channels], initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=is_uniform))
            self.bias    = tf.get_variable('bias', [self.output_channels], initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            self.summary_weights = tf.summary.histogram(self.name + '/weights', self.weights)
            self.summary_bias = tf.summary.histogram(self.name + '/bias', self.bias)
            return self.activation(input_)
    
    def activation(self, input_):
        # Compute convolution conv -> relu -> pooling -> dropout
        self.activation_ = conv2d(input_, self.weights, strides=self.strides, padding=self.padding) + self.bias
        return self.activation_

    def get_summaries(self):
        return [self.summary_weights, self.summary_bias]

    def get_activation(self):
        return self.activation_

class MaxPooling2D(Layer):
    def __init__(self, *args, **kwargs):

        if 'kernel_size' in kwargs:
            self.kernel_size = [1] + kwargs['kernel_size'] + [1]
        else:
            self.kernel_size = [1,2,2,1]
        if 'strides' in kwargs:
            self.strides = [1] + kwargs['strides'] + [1]
        else:
            self.strides = [1,2,2,1]
        if 'padding' in kwargs:
            self.padding = kwargs['padding']
        else:
            self.padding = 'VALID'

        self.reduce_length = False
        self.activation_ = None

    def build(self, *args, **kwargs):
        assert len(args) == 3
        self.name = args[0]
        self.input_shape = args[1]
        input_ = args[2]
        
        if self.padding == 'VALID':
            out_height = np.ceil(float(self.input_shape[0] - self.kernel_size[1] + 1) / float(self.strides[1]))
            out_width  = np.ceil(float(self.input_shape[1] - self.kernel_size[2] + 1) / float(self.strides[2]))
        if self.padding == 'SAME':
            out_height = np.ceil(float(self.input_shape[0]) / float(self.strides[1]))
            out_width  = np.ceil(float(self.input_shape[1]) / float(self.strides[2]))
        self.output_shape = [out_height, out_width, self.input_shape[-1]]
        return self.activation(input_)
    
    def activation(self, input_):
        self.activation_ = max_pool_2x2(input_, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding)
        return self.activation_

    def get_summaries(self):
        return []

    def get_activation(self):
        return self.activation_

class Flatten(Layer):
    def __init__(self, *args, **kwargs):
        self.reduce_length = False
        self.activation_ = None

    def build(self, *args, **kwargs):
        assert len(args) == 3
        self.name = args[0]
        self.input_shape = args[1]
        input_ = args[2]

        self.output_shape = [int(np.prod(self.input_shape))]
        return self.activation(input_)

    def activation(self, input_):
        self.activation_ = tf.reshape(input_, [-1] +  self.output_shape)
        return self.activation_

    def get_summaries(self):
        return []

    def get_activation(self):
        return self.activation_
    
class Model(object):
    def __init__(self):
        self.model_struct = []
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)
        
    def save_to_json(self, model_name):
        for layer in self.layers:
            self.model_struct.append(layer.get_json())
        json.dump(self.model_struct, open(model_name, 'w')) 

    def _create_layer(self, layer):
        module = __import__(__name__)
        l = getattr(module, layer['layer'])
        c_layer = l(**layer)
        return c_layer

    def load_from_json(self, model_name):
         model = json.load(open(model_name))
         for layer in model:
             l = self._create_layer(layer)
             self.add_layer(l)
  
    def _get_segment_indices(self, ids_):
        head = tf.reverse(tf.add(tf.reduce_max(ids_, axis=0), 1),[0])
        max_columns = tf.concat([tf.reverse(tf.slice(head, [0], [tf.size(head)-1]),[0]),[1]],0)
        multipliers = tf.cumprod(max_columns, reverse=True)
        y,idx = tf.unique(tf.reduce_sum(tf.multiply(ids_, multipliers),axis=1))
        return idx

    def _reduce_indices(self, ids_):
        ids_shape   = tf.shape(ids_)
        root_shape  = tf.gather(ids_shape, tf.range(0,tf.size(ids_shape)-1))
        last_column = tf.gather(ids_shape, [tf.size(ids_shape)-1])
        new_shape = tf.concat([last_column-1, root_shape], 0)
        reduced_ids = tf.reshape(tf.gather(tf.reshape(tf.transpose(ids_), [-1]), tf.range(0, tf.reduce_prod(new_shape))), new_shape)
        reduced_ids = tf.transpose(reduced_ids)
        return reduced_ids

    def compile_(self, shapes, loss, callbacks=None, optimizer='adam', learning_rate=1e-4, debug_folder='../debug'):
        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(43)
            placeholders = []
            if shapes['x']['sparse']:
                self.inputs = tf.sparse_placeholder(shapes['x']['type'], name='x')
            else:
                self.inputs = tf.placeholder(shapes['x']['type'], shape=[None] + shapes['x']['shape'], name='x')
    
            self.targets = tf.placeholder(shapes['y']['type'], name='targets')
            self.segment_ids = tf.placeholder(shapes['segment_ids']['type'], shape=[None] + shapes['segment_ids']['shape'], name='segment_ids')
            self.is_training = tf.placeholder(tf.bool, name='phase')
    
            is_training = self.is_training
            outputs = self.inputs
            output_shape = shapes['x']['shape']
    
            lengths = self.segment_ids
            segment_indices = lengths
    
            summaries = []
            layer_names = {}
            for l,layer in enumerate(self.layers):
                layer_name = layer.__class__.__name__ 
                if layer_name not in layer_names:
                    layer_names[layer_name] = 1
                layer_names[layer_name] += 1
                layer_name += '%d' % (layer_names[layer_name]-1)
    
                if layer.reduce_length:
                    segment_indices = self._get_segment_indices(lengths)
                    reduced_ids = self._reduce_indices(lengths)
                    lengths = tf.segment_max(reduced_ids, segment_indices)
                
                outputs = layer.build(layer_name, output_shape, outputs, lengths=segment_indices, is_training=self.is_training)
                output_shape = layer.output_shape
                summaries += layer.get_summaries()
    
            # Save model structure
            timestamp = int(time.time())
            self.debug_folder = '%s/%d' % (debug_folder, timestamp)
            os.makedirs(self.debug_folder + '/logs')
            self.save_to_json('%s/model.json' % self.debug_folder)
            self.debug_folder = self.debug_folder + '/logs'
    
            self.summaries = summaries
            self.outputs = outputs
    
            # Build loss callback
            self.loss = loss
            self.loss.build(logits=self.outputs, labels=self.targets)
            self.loss_function,_ = self.loss.get_function()

            # Build Callbacks
            self.callbacks = []
            if callbacks is not None:
                for callback in callbacks:
                    if isinstance(callback, Regularizer):
                         callback.build(layers=self.layers)
                         r,_ = callback.get_function()
                         self.loss_function += r
                    else:
                         callback.build(logits=self.outputs, labels=self.targets)
                         self.callbacks += [callback]
                    
            # Merge Summaries
            self.merged = []
            self.merged_callbacks = self.loss.get_summary()
    
            if len(self.summaries):
                self.merged = tf.summary.merge(self.summaries)
            else:
                self.merged = []
            if self.callbacks is not None:
                for callback in self.callbacks:
                    self.merged_callbacks += callback.get_summary()
            self.merged_callbacks = tf.summary.merge(self.merged_callbacks)
    
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if optimizer is None:        
                    self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss_function)    
                if optimizer == 'sgd':
                    self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss_function)
                if optimizer == 'adadelta':
                    self.train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(self.loss_function)
                if optimizer == 'adam':
                    self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_function)    
            self.saver = tf.train.Saver(max_to_keep=None)
#            config = tf.ConfigProto(
#                device_count = {'GPU': 0}
#            )      
            #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99)
            sess = tf.Session(graph=self.graph)#, config=config)#config=tf.ConfigProto(gpu_options=gpu_options))
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
    
            self.sess = sess

    def restore(self, path):
        self.saver = tf.train.import_meta_graph(path)
        self.saver.restore(self.sess, tf.train.latest_checkpoint('/'.join(path.split('/')[:-1])))

    def train_on_batch(self, xb, mb_dim, yb, ib):
        # Training placeholder
        sess = self.sess
        x = self.inputs
        y = self.targets
        segment_ids = self.segment_ids
        is_training = self.is_training
        self.train_step.run(session=sess, feed_dict={x: xb, segment_ids:ib, y:yb, is_training:True})

    def save_model(self, epoch):
        save_path = self.saver.save(self.sess, '%s/model%d.ckpt' % (self.debug_folder, epoch+1))

    def save_merged(self, name, epoch, verbose=False):
        writer = tf.summary.FileWriter('%s/%s' % (self.debug_folder, name))
        writer.add_summary(self.sess.run(self.merged, feed_dict={}), epoch)

        loss_value = self.loss(self.sess)
        callback_values = []

        feed_dict={self.loss.placeholder: loss_value}
        for callback in self.callbacks:
            value = callback(self.sess)
            feed_dict[callback.placeholder] = value
            callback_values.append([callback.name, value])

        res = self.sess.run(self.merged_callbacks, feed_dict)
        writer.add_summary(res, epoch)
        writer.close()
        if verbose:
           ret_dict = {'loss': loss_value}
           for c in callback_values:
               ret_dict[c[0]] = c[1]
           return ret_dict          
 
    def update_metrics(self, xb, mb_dim, yb, ib):
        sess = self.sess
        x = self.inputs
        y = self.targets
        segment_ids = self.segment_ids
        is_training = self.is_training

        feed_dict={x: xb, segment_ids:ib, y:yb, is_training:False}
        self.loss.update(sess, mb_dim, feed_dict)

        for callback in self.callbacks:
            callback.update(sess, feed_dict)
