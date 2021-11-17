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
    parser.add_argument("--topk", default=10, type=int)
    args = parser.parse_args()
    max_network = args.max_network
    min_network = args.min_network
    topk=args.topk
    
    for j in range(9):
      x = []
      y = []
      y1 = []
      scores = {}
      print("experiment: ", j)
      for i in range(300):
        #print("Experiment ", str(i))
        i = random.randint(min_network, max_network)
        results = api.query_by_index(i, 'cifar10') # get i-th network info
        results_valid = api.query_by_index(i, 'cifar10-valid')#get_more_info(i, 'cifar10-valid', None, True)
        #result.get_train()
        if 777 in results_valid:
            results_valid = results_valid[777]
        elif 888 in results_valid:
            results_valid = results_valid[888]
        elif 999 in results_valid:
            results_valid = results_valid[999]
        else:
            results_valid = results_valid
            
        #print(results[777].get_train())
        acc = results_valid.get_eval('x-valid', 20)['accuracy']#info['valid-accuracy'] #get validation accuracy of the i-th network
        y.append(acc)
        if 777 in results:
            results = results[777]
        elif 888 in results:
            results = results[888]
        elif 999 in results:
            results = results[999]
        else:
            results = results  
        #print(results.get_train(20))
        #print(results)
        #acc = results.get_eval('x-valid', 20)['accuracy']
        acc_test = results.get_eval('ori-test')['accuracy'] #get test accuracy of the i-th network
        y1.append(results.get_train()['loss'])
        xdata  = torch.load('output/NAS-BENCH-201-4/simplifies/architectures/' + str(i).zfill(6) + '-FULL.pth')

        odata  = xdata['full']['all_results'][('cifar10', 777)]
        result1 = ResultsCount.create_from_state_dict( odata )
        scores[float(result1.get_train()['loss'])] = (acc,acc_test)#(acc1, acc)#we use loss to store MGM score

        x.append(float(result1.get_train()['loss']))

      new_scores = sorted(scores.items(), key=lambda x:x[0], reverse=True) # rank networks based on MGM scores
      #print("Results of best networks: lists of (validation accuracy, test accuracy)")
      topk_value = [item[1] for item in new_scores[:topk]]
      #print(topk_value)
      top_acc = sorted(topk_value, key=lambda x: x[0], reverse=True)
      print("The results of the best network:")
      print(top_acc[0][1])


    #c,p = stats.spearmanr(np.array(x), np.array(y))
    #print("spearman score between MGM and valid accuracy")
    #print(c,p)

    #c,p = stats.spearmanr(np.array(x), np.array(y1))
    #print("spearman score between MGM and train loss")
    #print(c,p)
