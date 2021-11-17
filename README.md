# KNAS
Codes for paper "KNAS: Green Neural Architecture Search"


KNAS is a green (energy-efficient) Neural Architecture Search (NAS) approach. It contains two steps: coarse-grained selection and fine-grained selection. 
The first step selects k networks candidates without any training and then fine-grained step selects the best one from the selected candidates via training on downstream tasks. 
KNAS is very simple and only requires gradient vectors to get MGM scores. Please refer to function "procedure" in file exps/NAS-Bench-201/functions.py for MGM implementation.
 
# Requirements and Installation

The required environments:
* python 3
* scipy
* numpy

The required data:
* [NAS-Bench-201](https://drive.google.com/drive/folders/1ihjh90KpcmAr5-d_WUQlXmQNpOY8fZbT?usp=sharing)


**To use KNAS** and develop locally:

* The first step is to initialize the output directory. You will see a directory called "output" after running this step.

```
bash scripts-search/NAS-Bench-201/meta-gen.sh NAS-BENCH-201 4
```

* The second step is to compute MGM scores for network candidates. The second and the third parameters represent the index range of network candidates (e.g., [0,5000)). The last parameter means random seeds. You can find the details of MGM at function procedure in file exps/NAS-Bench-201/functions.py.  

```
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/NAS-Bench-201/train-models.sh 0     0   5000 -1 '777 888 999'
```

* The third step is to extract MGM info and save it to the directory: outout/NAS-Bench-201/output/NAS-BENCH-201-4/simplifies/ . 

```
CUDA_VISIBLE_DEVICES=0 python3 exps/NAS-Bench-201/statistics.py --mode cal --target_dir 000000-005000-C16-N5
```

* The last step is to select networks. Since benchmark [NAS-bench-201](https://github.com/D-X-Y/NAS-Bench-201) provides all test results, we directly use validation accuracy to select the best network.  

```
python3 cifar10.py --min_network 0 --max_network 5000 --topk 40 

```


# Citation

Please cite as:

``` bibtex
@inproceedings{knas,
  title = {KNAS: Green Neural Architecture Search},
  author= {Jingjing Xu and
               Liang Zhao and
               Junyang Lin and
               Rundong Gao and
               Xu Sun and
               Hongxia Yang},
  booktitle = {Proceedings of ICML 2021},
  year = {2021},
}
```



