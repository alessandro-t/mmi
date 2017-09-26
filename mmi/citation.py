import numpy as np
from mmi.dataset import Dataset


class CitationDataset(Dataset):
    def __init__(self, data, vocabulary, indices, include_node=True, type_='mmi'):
        self.X = np.array(data['X'])
        self.y = data['y']
        self.adj = data['adj']
        self.vocabulary = vocabulary
        self.indices = indices
        self.include_node = include_node
        self.type_ = type_
        if self.include_node:
            self.adj[np.arange(self.adj.shape[0]) , np.arange(self.adj.shape[0])] = 1 
        
    
    def get_minibatches(self, mb_size, seed=None, shuffle=True):
        all_indices = self.indices.copy()
        if shuffle:
            np.random.shuffle(all_indices)
        n_samples = len(all_indices)
        n_batches = np.ceil(1.*n_samples/mb_size).astype(np.int32)

        for b in range(n_batches):
            left_i  = b*mb_size
            right_i = min((b+1)*mb_size, n_samples)
            
            current_indices = all_indices[left_i:right_i]
            segment_ids = []
            x_batch = []
            y_batch = self.y[current_indices]
            for t,c in enumerate(current_indices):
                neighbor_indices = self.adj[c].nonzero()[1]
                neighbors = self.X[neighbor_indices]
                
                for s,neighbor in enumerate(neighbors):
                    filtered_neighbor = [self.vocabulary[x] for x in neighbor if x in self.vocabulary]
                    x_batch += filtered_neighbor
                    if self.type_ == 'mmi':
                        segment_ids += [[t,s]]*len(filtered_neighbor)
                    if self.type_ == 'mi':
                        segment_ids += [t]*len(filtered_neighbor)
                    
                
            yield np.vstack(x_batch),right_i-left_i,y_batch,np.vstack(segment_ids)
