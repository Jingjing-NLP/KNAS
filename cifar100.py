from scipy import stats
import numpy as np
import random
import argparse

from lib.nas_201_api import NASBench201API as API
api = API('a.pth') #networks associated with training/evaluation info
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
    parser.add_argument('--topk', default=10, type=int)
    
    args = parser.parse_args()
    max_network = args.max_network
    min_network = args.min_network
    topk = args.topk

    for j in range(9):
      x = []
      y = []
      y1 = []
      scores = {}
      print("Experiment: ", j)
      for j in range(300):
        i = random.randint(min_network, max_network)
        results = api.query_by_index(i, 'cifar100')
        if 777 in results:
            results = results[777]
        elif 888 in results:
            results = results[888]
        elif 999 in results:
            results = results[999]
        else:
            results = results
        #print(results)
        train_loss = results.get_train(20)['loss']#results.get_eval('x-valid')['accuracy'] # valid accuracy
        acc_test = results.get_eval('x-test')['accuracy'] #test accuracy
        y1.append(results.get_train(20)['loss']) # training loss
        y.append(float(train_loss))
        xdata  = torch.load('output/NAS-BENCH-201-4/simplifies/architectures/' + str(i).zfill(6) + '-FULL.pth')

        odata  = xdata['full']['all_results'][('cifar100', 777)]
        result1 = ResultsCount.create_from_state_dict( odata )
        #since NAS-bench-201 does not provide validation accuracy at each step, we use the training losss to select networks
        scores[float(result1.get_train()['loss'])] = (train_loss,acc_test) # we use loss to store MGM score to avoid changing read interface of NAS-bench-201.

        x.append(float(result1.get_train()['loss']))
      
      new_scores = sorted(scores.items(), key=lambda x:x[0], reverse=True)
      #print("Results of best networks: lists of (validation accuracy, test accuracy)")
      topk_value = [item[1] for item in new_scores[:topk]]
      #print(topk_value)
      top_acc = sorted(topk_value, key=lambda x: x[0], reverse=False)
      print("The results of the best network:")
      print(top_acc[0][1])
      #print("results of top-k networks: lists of (validation accuracy, test accuracy)")
      #print([item[1] for item in new_scores[:10]])
      #c,p = stats.spearmanr(np.array(x), np.array(y))
      #print("spearman score between MGM and valid accuracy")
      #print(c,p)

      #c,p = stats.spearmanr(np.array(x), np.array(y1))
      #print("spearman score between MGM and train loss")
      #print(c,p)


