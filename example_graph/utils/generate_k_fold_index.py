import argparse
import numpy as np
import h5py


def k_fold(y, train_sample_for_each_class=20):
    train_indices = []
    for i in range(max(y)+1):
        train_indices += [np.random.choice(np.where(y == i)[0], train_sample_for_each_class, replace=False)]
    train_indices = np.concatenate(train_indices)
    test_indices  = np.array(list(set(np.arange(len(y))) - set(train_indices)))
    test_indices  = np.random.choice(test_indices, 1000, replace=False)

    val_indices  = np.array(list(set(np.arange(len(y))) - set(train_indices) - set(test_indices)))
    val_indices  = np.random.choice(val_indices, 500, replace=False)

    return train_indices, test_indices, val_indices

def main(args):
    data = np.load(args.input)[()]
    if args.seed is not None:
        np.random.seed(args.seed)
    y = data['y']
    y = np.argmax(y, axis=1)
    splits = h5py.File('k_folds_split.h5', 'w')
    # train = splits.create_group('train')
    # test = splits.create_group('test')
    # val = splits.create_group('val')

    for k in range(args.k_folds):
        train_indices, test_indices, val_indices = k_fold(y)
        print(k)
        train_split = splits.create_dataset('train/split_%d' % k, train_indices.shape, h5py.h5t.NATIVE_INT32)
        test_split = splits.create_dataset('test/split_%d' % k, test_indices.shape, h5py.h5t.NATIVE_INT32)
        val_split = splits.create_dataset('val/split_%d' % k, val_indices.shape, h5py.h5t.NATIVE_INT32)
        train_split[:] = train_indices
        test_split[:] = test_indices
        val_split[:] = val_indices
    splits.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', type=str, help='numpy input file')
    parser.add_argument('-k', '--k-folds', type=int, help='Number of K-folds', default=10)         
    parser.add_argument('-s', '--seed', type=int, help='Seed for K-fold generation', default=None)         
    args = parser.parse_args()
    main(args)
    
