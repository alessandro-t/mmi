from natsort import natsorted
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.corpus import stopwords
from scipy.sparse import coo_matrix

import argparse
import gensim
import numpy as np
import operator
import string


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input',  type=str, help='input  Cora folder')
    parser.add_argument('output', type=str, help='output file name')
    # parser.add_argument('-l', '--self-loops', action='store_true', default=False, help='include self loops within adj matrix')
    return parser

def preprocess(path):
    papers = open('%s/papers' % path).readlines()
    edges  = open('%s/cora.cites' % path).readlines()
    ids = set([id_.split('\t')[0].strip() for id_ in open('%s/cora.content' % path).readlines()])
    labels = {id_.split('\t')[0].strip():id_.split('\t')[-1].strip() for id_ in open('%s/cora.content' % path).readlines()}
    edges = np.array([x.split() for x in edges])
    id_to_index_map = {k:i for i,k in enumerate(sorted(list(ids)))}
    index_to_id_map = {i:k for i,k in enumerate(sorted(list(ids)))}

    labels = [labels[x] for x in sorted(list(ids))]
    label_names = sorted(list(set(labels)))
    label_name_to_id_map = {}

    for i,name in enumerate(label_names):
        one_hot = np.zeros(len(label_names))
        one_hot[i] = 1.0
        label_name_to_id_map[name] = one_hot

    y = np.vstack(list(map(lambda x: label_name_to_id_map[x], labels)))

    # Compute adj matrix
    x_ind = np.array(list(map(lambda x:id_to_index_map[x], edges[:,0])))[:,None]
    y_ind = np.array(list(map(lambda x:id_to_index_map[x], edges[:,1])))[:,None]
    adj_index = np.vstack((np.hstack((x_ind,y_ind)),np.hstack((y_ind,x_ind))))
    #adj_index = np.vstack(np.hstack((y_ind,x_ind)))

    adj = coo_matrix((np.ones(adj_index.shape[0]), (adj_index[:,0], adj_index[:,1])), shape=(len(ids), len(ids)))
    adj = adj.tocsr()

    # Extract corpus
    docs = {}
    for paper in papers:
        splitted_paper = paper.strip().split('\t')
        id_, site = splitted_paper[0], splitted_paper[1]
        if id_ in ids:
            docs[id_] = site.strip()

    corpus = {}
    vocabulary = {}
    doc_occurencies = {}
    sw = set(stopwords.words('english'))
    stem = SnowballStemmer('english')

    for id_ in ids:
        paper = open('%s/extractions/%s' % (path, docs[id_])).readlines()
        text = ''
        for p in paper:
            split_p = p.split(':')
            if len(split_p) >= 2:
                if sum([split_p[0].startswith(x) for x in ['Title', 'Abstract']]):#['Pubnum', 'Keyword', 'Degree','Note', 'Address', 'Author', 'Affiliation','URL', 'Refering-URL', 'Root-URL', 'Address', 'Phone', 'Email', 'Abstract-found', 'Intro-found', 'Reference']]):
                    text += ' ' + (':'.join(split_p[1:])).strip()
        corpus[id_] = gensim.utils.simple_preprocess(text)
    sorted_x = sorted(vocabulary.items(), key=operator.itemgetter(1), reverse=True)
    sorted_x = [x for x in sorted_x if len(x[0]) > 1 and not x[0].isdigit()]

    X = []
    new_ids = []
    for id_ in sorted(list(ids)):
        X += [corpus[id_]]
        new_ids += [id_]
    #np.save('ids', np.array(ids))
    return X, y, adj


def main(args):
    X, y, adj = preprocess(args.input)
    to_save = np.array({'X':X, 'y':y,'adj':adj})
    np.save('%s' % args.output, to_save)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)

