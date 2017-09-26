import argparse
import numpy as np

def get_parser():
	# to extract the dictionary from numpy array
	# glove = np.load('glove100.npy')[()] 
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input',  type=str, help='input  Glove file')
    parser.add_argument('output', type=str, help='output Numpy array')
    return parser

def main(args):        
    words = open(args.input, encoding='latin1').readlines()
    dictionary = {''.join(w.split()[0]):np.array(w.split()[-300:], dtype=np.float32) for w in words}
    np.save(args.output, dictionary)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
