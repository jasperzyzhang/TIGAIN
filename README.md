# Code For TIGAIN
 ## Tightly Integrative Generative Adversarial Imputation Networks(TIGAIN)

Author: Jasper Zhang

This Codebase is Implemented based on the Code provided by Paper: Jinsung Yoon, James Jordon, Mihaela van der Schaar, "GAIN: Missing Data Imputation using Generative Adversarial Nets," International Conference on Machine Learning (ICML), 2018.

### Command inputs:

- data_name: GE
- miss_rate: probability of missing components
- batch_size: batch size
- hint_rate: hint rate
- alpha: hyperparameter
- iterations: iterations
- col_num: Number of columns selected from data
- result_name: name of result file
- imp_type: imputation method name, 'mice', 'gain' or 'TIgain'
- info_name: extra info provided for TIgain. 'ME': DNA Methylation, 'CN': Copy Number Variation, or 'CON': Concatenation of ME and CN.

### Example command

Run TIGAIN experiments on Gene Expression Data(GE), randomly select 2000 columns out of 17k columns, using Copy Numver Variation(CN) as additional infomation for data imputation,saved data in result folder with name 'new_result.csv'.

```
$ python3 main_TIGAIN.py --data_name GE --miss_rate 0.2 --batch_size 32 --hint_rate 0.9 --alpha 100 --iterations 1000 --col_num 2000 --result_name 'new_result' --imp_type 'TIgain' --info_name "CN"
```

