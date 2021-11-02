from scipy import stats
import numpy as np
import random
import argparse


from lib.nas_201_api import NASBench201API as API
api = API('a.pth')
from lib.nas_201_api import ResultsCount
import torch
x = []
y = []
y1 = []
scores = {}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some input flags.')
    parser.add_argument('--max_network', default=1000, type=int,
                        help='maximum index of network candidates')
    parser.add_argument('--min_network', default=0, type=int,
                        help='minimum index of network candidates')
    args = parser.parse_args()
    max_network = args.max_network
    min_network = args.min_network

    for i in range(min_network, max_network):
        results = api.query_by_index(i, 'ImageNet16-120')
        info = api.get_more_info(i, 'ImageNet16-120', None, True)
        acc = info['valid-accuracy']
        y.append(acc)
        if 777 in results:
            results = results[777]
        elif 888 in results:
            results = results[888]
        elif 999 in results:
            results = results[999]
        else:
            results = results
        acc_test = results.get_eval('ori-test')['accuracy']
        y1.append(results.get_train()['loss'])
        xdata  = torch.load('output/NAS-BENCH-201-4/simplifies/architectures/' + str(i).zfill(6) + '-FULL.pth')

        odata  = xdata['full']['all_results'][('ImageNet16-120', 777)]
        result1 = ResultsCount.create_from_state_dict( odata )
        scores[float(result1.get_train()['loss'])] = (acc,acc_test)#(results[777], acc1)

        x.append(float(result1.get_train()['loss']))

    new_scores = sorted(scores.items(), key=lambda x:x[0], reverse=True)
    print("Results of iop-k networks: lists of (validation accuracy, test accuracy)")
    print([item[1]  for item in new_scores[:10]])

    print("speaerman scores between MGM and valid accuracy")
    c,p = stats.spearmanr(np.array(x), np.array(y))
    print(c,p)

    print("spearman scores between MGM and training loss")
    c,p = stats.spearmanr(np.array(x), np.array(y1))
    print(c,p)
