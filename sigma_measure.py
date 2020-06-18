import argparse
from donaldson import donaldson
import fermat_quintic
from metric_measures import sigma
import datetime
import matplotlib.pyplot as plt 

def main(k_max, n_t, output_file):
    sigma_acc = [] 
    for k in range(1, k_max):
        sigma_val = sigma(k, n_t, donaldson(k))
        print('sigma(k=%d, n=%d)=%f' % (k, n_t, sigma_val))
        sigma_acc.append(sigma_val)

    plt.plot(list(range(1, k_max)), sigma_acc, color='red', linestyle='dashed', 
        marker='o', markerfacecolor='red') 
    plt.xlabel('k') 
    plt.ylabel(r'$\sigma_k$') 
    plt.title(r'$\sigma$-measure vs. k') 
    plt.savefig(output_file)

def parser_config():
    parser = argparse.ArgumentParser(description='Compute $\sigma_k$ measure for a given k and balanced metric')
    parser.add_argument('-k', type=int, required=True, help='order of fermat quintic sections')
    parser.add_argument('-N', type=int, required=True, help='number of sample points')
    parser.add_argument('-o', '--output', type=str, default='sigma_%s.png' % datetime.datetime.now().isoformat(), 
        help='output file name')
    return parser

if __name__ == '__main__':
    args = parser_config().parse_args()
    main(args.k + 1, args.N, args.output)