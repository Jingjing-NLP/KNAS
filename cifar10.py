from scipy import stats
import numpy as np
import random
import argparse


from lib.nas_201_api import NASBench201API as API
api = API('a.pth') # networks associated with training/evaluation info
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
        results = api.query_by_index(i, 'cifar10') # get i-th network info
        info = api.get_more_info(i, 'cifar10-valid', None, True) 
        acc = info['valid-accuracy'] #get validation accuracy of the i-th network
        y.append(acc)
        acc_test = results[777].get_eval('ori-test')['accuracy'] #get test accuracy of the i-th network
        y1.append(results[777].get_train()['loss'])
        xdata  = torch.load('output/NAS-BENCH-201-4/simplifies/architectures/' + str(i).zfill(6) + '-FULL.pth')

        odata  = xdata['full']['all_results'][('cifar10', 777)]
        result1 = ResultsCount.create_from_state_dict( odata )
        scores[float(result1.get_train()['loss'])] = (acc,acc_test)#(acc1, acc)#we use loss to store MGM score

        x.append(float(result1.get_train()['loss']))

    new_scores = sorted(scores.items(), key=lambda x:x[0], reverse=True) # rank networks based on MGM scores
    print("Results of iop-k networks: lists of (validation accuracy, test accuracy)")
    print(max([item[1] for item in new_scores[:10]]))


    c,p = stats.spearmanr(np.array(x), np.array(y))
    print("spearman score between MGM and valid accuracy")
    print(c,p)

    c,p = stats.spearmanr(np.array(x), np.array(y1))
    print("spearman score between MGM and train loss")
    print(c,p)
