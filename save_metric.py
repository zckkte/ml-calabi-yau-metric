from donaldson import donaldson
import numpy as np
import os as os
import argparse

dirname = 'data'
fname = 'metric_'

def parser_config():
    parser = argparse.ArgumentParser(description='Computes balanced metric using donaldson')
    parser.add_argument('-k', type=int, required=True, help='order of fermat quintic sections')
    return parser

if __name__ == '__main__':

    args = parser_config().parse_args()

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for i in range(1,args.k+1):
        print('Starting {}.'.format(i))
        metric = donaldson(i)
        np.save(os.path.join(dirname, fname+str(i)+'.npy'), metric)

    print('done')