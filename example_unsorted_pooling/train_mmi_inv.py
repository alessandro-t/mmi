import argparse
import h5py
import numpy as np
import tensorflow as tf

from sklearn.feature_extraction.image import PatchExtractor
from tensorflow.examples.tutorials.mnist import input_data

def extend_images(img, top_bag_size):
    offset_x = np.ceil(img.shape[1]*1./top_bag_size[0]).astype(np.int32) + img.shape[1]
    offset_y = np.ceil(img.shape[2]*1./top_bag_size[1]).astype(np.int32) + img.shape[2]
    ext_img = np.zeros((img.shape[0], offset_x, offset_y, img.shape[3]), dtype=img.dtype)
    ext_img[:,:img.shape[1],:img.shape[2],:] = img
    return ext_img

def create_subsampling_mask(patches_shape, img_shape, stride=(5,5)):
    points_x_full = np.arange(0,img_shape[0],1)
    points_x_full = points_x_full[points_x_full < img_shape[0] - patches_shape[0] +1 ]
    
    points_y_full = np.arange(0,img_shape[1],1)
    points_y_full = points_y_full[points_y_full < img_shape[1] - patches_shape[1] +1 ]
    
    mask = np.zeros((len(points_x_full),len(points_y_full)))

    points_x = np.arange(0,img_shape[0],stride[0])
    points_x = points_x[points_x < img_shape[0] - patches_shape[0] +1 ]
    points_y = np.arange(0,img_shape[1],stride[1])
    points_y = points_y[points_y < img_shape[1] - patches_shape[1] +1 ]
    
    for x in points_x:
        for y in points_y:
            mask[x,y] = 1
    return mask > 0

def get_patches(img, tb_size, sb_size, mask_tb=None, mask_sb=None):
    tb_extractor = PatchExtractor(tuple(tb_size))
    sb_extractor = PatchExtractor(tuple(sb_size))
    tb_images = tb_extractor.transform(img)
    tb_images = tb_images.reshape((img.shape[0], -1, tb_images.shape[-2], tb_images.shape[-1]))
    
    if mask_tb is not None:
        tb_images = tb_images[:, mask_tb.ravel(), : ,:]
    tb_images = np.rollaxis(tb_images, 1, 4)
    sb_images = sb_extractor.transform(tb_images)
    sb_images = sb_images.reshape((img.shape[0], -1, sb_images.shape[-3], sb_images.shape[-2], sb_images.shape[-1]))
    sb_images = np.rollaxis(sb_images, 4, 1)

    if mask_sb is not None:
        sb_images = sb_images[:, :, mask_sb.ravel(), : ,:]
    sb_images = sb_images.reshape(list(sb_images.shape[:-2]) + [np.prod(sb_images.shape[-2:])])
    return sb_images

            
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def bag_layer(x, W, b):
    size_x = len(x.get_shape())
    feat_dim = x.get_shape()[-1].value
    nested_dim = [d.value for d in x.get_shape()[1:-1]]
    
    w1 = W.get_shape()[0].value
    w2 = W.get_shape()[1].value
    
    xxx = tf.reshape(x, [-1, feat_dim])
    # ADD LAYER HERE
    # LINEAR LAYER + RELU
    xxx = tf.matmul(xxx, W) + b
    xxx = tf.reshape(xxx, [-1] + nested_dim + [w2])
    xxx = tf.reduce_max(xxx, axis=-2)
    xxx = tf.nn.relu(xxx)
#     nested_dim = nested_dim[:-1]
#     if len(nested_dim):
#         xxx = tf.reshape(xxx, [-1] + nested_dim + [w2])

    return xxx

# def predict(X):

def get_model(X_train, y_train, patch_shape):
    graph_model = tf.Graph()
    with graph_model.as_default():
        x = tf.placeholder(tf.float32, shape=[None] + list(patch_shape[1:]))
        y_ = tf.placeholder(tf.float32, shape=[None, y_train.shape[1]])

        W1 = weight_variable((25,100))
        b1 = bias_variable([100])

        f1 = bag_layer(x, W1, b1)

        W2 = weight_variable((100,200))
        b2 = bias_variable([200])

        f2 = bag_layer(f1, W2, b2)

        # DENSE LAYER
        W_fc1 = weight_variable([200, 1024])
        b_fc1 = bias_variable([1024])

        h_fc1 = tf.nn.relu(tf.matmul(f2, W_fc1) + b_fc1)

        # DROPOUT
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([1024, y_train.shape[1]])
        b_fc2 = bias_variable([y_train.shape[1]])

        output_ = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        return x, keep_prob, y_, output_, graph_model


def train(X_train, y_train, X_test, y_test, x, keep_prob, y_, output_, graph_model, tb_size, sb_size, mask_tb):
    with graph_model.as_default():
        sess = tf.Session()
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output_))
        train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
        correct_prediction = tf.equal(tf.argmax(output_,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        sess.run(tf.global_variables_initializer())

        indices = np.arange(len(X_train))
        mb_size = 10
        n_batches = np.ceil(1.*len(indices)/mb_size).astype(np.int32)

        for e in range(20):
            np.random.shuffle(indices)
            for b in range(n_batches):
                left_i  = b*mb_size
                right_i = min((b+1)*mb_size,len(indices))
                current_indices = indices[left_i:right_i]
                x_batch = get_patches(extend_images(X_train[current_indices], tb_size), tb_size, sb_size, mask_tb)
                y_batch = y_train[current_indices]
                train_step.run(session=sess,feed_dict={x:x_batch , y_: y_batch, keep_prob: 0.5})
                
            acc = 0
            for k in range(0,10000,10):
                test_x_batch = get_patches(extend_images(X_test[k:k+10], tb_size), tb_size, sb_size, mask_tb)
                test_y_batch = y_test[k:k+10]
                acc += correct_prediction.eval(session=sess, feed_dict={x: test_x_batch, y_: test_y_batch, keep_prob: 1.0}).sum()
            acc /= 10000.
            print("step %d, test accuracy %g"%(e+1, acc))

def generate_data(w, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True, validation_size=0)

    train = np.zeros((mnist.train.images.shape[0],w,w,1), dtype=np.float32)
    test  = np.zeros((mnist.test.images.shape[0], w,w,1), dtype=np.float32)

    for ll,xi in enumerate(mnist.train.images.reshape(-1,28,28,1)):
        i,j = np.random.randint(28,w-28,(2,))
        train[ll,i:i+28,j:j+28,:] = xi

    for ll,xi in enumerate(mnist.test.images.reshape(-1,28,28,1)):
        i,j = np.random.randint(28,w-28,(2,))
        test[ll,i:i+28,j:j+28,:] = xi

    X_train, y_train = train, mnist.train.labels
    X_test,  y_test  = test,  mnist.test.labels
    return X_train, y_train, X_test,  y_test

def load_data(path):
    h5file = h5py.File(path)
    X_train = h5file['train']['X'][:]
    y_train = h5file['train']['y'][:]
    X_test  = h5file['test']['X'][:]
    y_test  = h5file['test']['y'][:]
    return X_train, y_train, X_test,  y_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-w', '--width', type=int, help="Stride dimension", default=None)         
    parser.add_argument('-d', '--data', type=str, help="h5 file containing train and test", default=None)                                                                                            
           
    args = parser.parse_args()  

    if args.data is None and args.width is None:
        print('Please enter either width or data')
        parser.print_help()
    elif args.width is not None:
        X_train, y_train, X_test,  y_test = generate_data(args.width)
    elif args.data is not None:
        X_train, y_train, X_test,  y_test = load_data(args.data)

    tb_size   = [15, 15]
    tb_stride = [5, 5]
    sb_size   = [5, 5]

    X_sample = extend_images(X_train[0:1], tb_size)
    mask_tb = create_subsampling_mask(patches_shape=tb_size, img_shape=X_sample.shape[1:-1], stride=tb_stride)
    patch_sample = get_patches(X_sample, tb_size, sb_size, mask_tb)
    x, keep_prob, y_, output_, gm = get_model(X_train, y_train, patch_sample.shape)
    train(X_train, y_train, X_test, y_test, x, keep_prob, y_, output_, gm, tb_size, sb_size, mask_tb)






