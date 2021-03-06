import h5py
import numpy as np
import tensorflow as tf

class Dataset(object):
    """Dataset is an abstract class that
    represents a multi^N instance problem."""
    def get_minibatches(self, mb_size, shuffle=True):
        raise NotImplementedError()

class BasicMMIMnist(Dataset):
    def __init__(self, path, which_set='train', seed=None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        data = h5py.File(path)
        self.X, self.y = data['%s_features' % which_set][:], data['y_%s' % which_set][:]
        
    def create_dataset(self, pos=100, neg=100, max_top_bags=7, max_sub_bags=7):
        pos_found = 0
        neg_found = 0
        top_bags = []
        top_bags_labels = []
        while len(top_bags)<pos+neg:
            n_top_bags = np.random.randint(2,max_top_bags+1)
            all_index = []
            for t in range(n_top_bags):
                is_positive = False
                sub_bag_labels = np.random.randint(0,10,np.random.randint(2,max_sub_bags+1))
                sub_bag_index = []
                for label in sub_bag_labels:
                    sub_bag_index += [np.random.choice(np.where(self.y == label)[0],1)]
                sub_bag_index = np.array(sub_bag_index).ravel()
                if (7 in set(self.y[sub_bag_index])) and (3 not in set(self.y[sub_bag_index])):
                    is_positive = True
                all_index += [sub_bag_index]
                
            if is_positive and pos_found < pos:
                top_bags += [all_index]
                top_bags_labels += [1]
                pos_found += 1
                
            if not is_positive and neg_found < neg:
                top_bags += [all_index]
                top_bags_labels += [0]
                pos_found += 1
        self.tb_index = np.array(top_bags)
        self.tb_labels = np.array(top_bags_labels)
        
    def get_minibatches(self, mb_size, shuffle=True):
        indices = np.arange(len(self.tb_index))
        if shuffle:
            np.random.shuffle(indices)
        n_batches = np.ceil(1.*len(indices)/mb_size).astype(np.int32)
        for b in range(n_batches):
            left_i  = b*mb_size
            right_i = min((b+1)*mb_size, len(indices))
            current_indices = indices[left_i:right_i]
            current_bags = self.tb_index[current_indices]
            segment_ids = []
            index = []
            for i,tb in enumerate(current_bags):
                for j,sb in enumerate(tb):
                    for k, ib in enumerate(sb):
                        segment_ids += [[i,j]]
                        index += [ib]
            index = np.array(index)
            segment_ids = np.array(segment_ids)
            yield self.X[index], right_i-left_i, self.tb_labels[current_indices], segment_ids
            
class MMIMNIST(Dataset):
    def __init__(self, X, y, mode='standard', seed=None):
        self.X = X[:,:,:,None]
        self.y = y
        self.mode = mode
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def _get_labels(self, random_samples, positive_rules, negative_rules):
        labels = []
        for sample in random_samples:
            contains_positive = np.array([np.in1d(rule,sample).sum() == len(rule) for rule in positive_rules]).sum() > 0
            contains_negative = np.array([np.in1d(rule,sample).sum() == len(rule) for rule in negative_rules]).sum() > 0
            if contains_positive:
                labels.append(1)
            else:
                if contains_negative:
                    labels.append(-1)
                else:
                    labels.append(0)
        return labels

    def _get_top_labels(self, labels):
        if (np.array(labels) == -1).sum() == 0 and (np.array(labels) == 1).sum() > 0:
            return 1
        return -1

    def _generate_random_sub_bag(self, n_sub_bags, max_lenght=12):            
        random_samples = []
        for x in range(n_sub_bags):
            bag_length = np.random.randint(2, max_lenght+1)
            random_samples.append(np.random.randint(0, 10, bag_length))
        return random_samples
    
    def _generate_random_top_bags(self, n_positive_top_bags, n_negative_top_bags, max_length_top_bag=5, max_length_sub_bag=6):
        positive_top_bags = []
        negative_top_bags = []

        positive_rules = [[1,3,5,7],[2,1,3],[3,2,7,9]]
        negative_rules = [[8,9],[4,5,6],[7,2,1]]

        while (n_positive_top_bags + n_negative_top_bags) > 0:
            n_sub_bags = np.random.randint(2, max_length_top_bag+1)
            random_samples = self._generate_random_sub_bag(n_sub_bags, max_length_sub_bag)
            labels = self._get_labels(random_samples, positive_rules, negative_rules)
            top_label = self._get_top_labels(labels)
            if top_label == 1 and n_positive_top_bags > 0:
                positive_top_bags.append(random_samples)
                n_positive_top_bags -= 1
            if top_label == -1 and n_negative_top_bags > 0:
                negative_top_bags.append(random_samples)
                n_negative_top_bags -= 1

        return positive_top_bags, negative_top_bags

    def create_dataset(self, n_positive_top_bags, n_negative_top_bags, max_length_top_bag=5, max_length_sub_bag=6):
        pos, neg = self._generate_random_top_bags(n_positive_top_bags,n_negative_top_bags)
        indices = []
        segmented_ids = []
        top_bag_counter = 0
        sub_bag_counter = 0
        counts = 0
        self.indices_hash = {}
        for top_bag in pos+neg:
            self.indices_hash[top_bag_counter] = []
            for sub_bag in top_bag:
                for n in sub_bag:
                    index = np.random.choice(np.where(self.y == n)[0])
                    indices.append(index)
                    segmented_ids.append([top_bag_counter, sub_bag_counter])
                    self.indices_hash[top_bag_counter].append(counts)
                    counts += 1
                sub_bag_counter += 1
            top_bag_counter += 1
            sub_bag_counter = 0
        self.X_indices = np.array(indices)
        self.segmented_ids = np.array(segmented_ids)
        self.dimensions = [n_positive_top_bags+n_negative_top_bags, 1+2]
        self.indices = np.arange(self.dimensions[0])
        self.labels =  np.array([1]*n_positive_top_bags + [0]*n_negative_top_bags)

    def get_minibatches(self, mb_size, shuffle=True):
        assert type(mb_size) == int
        assert mb_size > 0

        n_samples = self.dimensions[0]
        n_batches = np.ceil(1.*n_samples/mb_size).astype(np.int32)

        np.random.shuffle(self.indices)
        if shuffle:
            all_indices = self.indices
        else:
            all_indices = np.arange(n_samples)

        for b in range(n_batches):
            left_i  = b*mb_size
            right_i = min((b+1)*mb_size, n_samples)

            curr_indices = all_indices[left_i:right_i]
            indices_map = {x:i for i,x in enumerate(curr_indices)}

            curr_row_indices = np.hstack([self.indices_hash[ii] for ii in curr_indices])

            renamed_row_indices = np.vectorize(indices_map.get)(self.segmented_ids[curr_row_indices][:,0])

            X_batch = self.X[self.X_indices[curr_row_indices]]
            new_seg_ids = np.hstack((renamed_row_indices[:,None], self.segmented_ids[:,1:][curr_row_indices]))
            
            yield X_batch, right_i-left_i, self.labels[curr_indices].astype(np.float32), new_seg_ids
