# Robustness-Pytorch

Implementation of "Improving robustness of medical image diagnosis with denoising convolutional neural networks" in MICCAI2019 [paper]( https://link.springer.com/chapter/10.1007/978-3-030-32226-7_94)

## Requirements

- python, pytorch, numpy, scikit-learn, tensorboardX

## File Tree

```
├── README.md                
├── main.py                 
├── sk_g_test.sh            
└── src                     
    ├── __init__.py
    ├── action.py           # operations used by the model
    ├── agent.py            
    ├── attack.py           
    ├── config.py           
    ├── dataset.py          
    ├── net.py              
    └── tool.py             
```


## Detail

1. `dataset.py`   
You can define all your datasets here, and use `get_dataloader` function to get its PyTorch dataloader.
2. `net.py`   
This module includes all the models your project needed. And to reuse some sub-models, you can define them in a separate `net_parts.py`.
3. `action.py`   
The Action module occupies a very important position as you can code any action you want to record your network results, such as model graph,accuracy, confusion matrix, auc and loss etc..
4. `agent.py`   
It is the spokesman of the above three modules, because model train/eval with dataset and using action to record the performance. So, in this part, you can assign dataloader, model and action by get_dataloader, get_action and get_net function respectively.
5. `config.py`   
This part mainly used to config some directories which can be dirs to store results or dirs referring to datasets. It contains the environment which remains unchanged. It can be useful when you debug locally and deploy remotely.
6. `tool.py`   
It is a utils combination mainly used to generate parser. But you can put any mess code here.  
7. `main.py`   
It is the **core** of the framework as it provides `args` to other modules. We use `args` to control almost all parameters.

## Basic Usage
`main.py`  
```python main.py test --n_dl 16 --dsid skin4 --mid res50 --midtf skin4_res50_g --batch_attack 64 --testidtf skin4_res18_b_eval_FGSM_e4```


----

## Cite

`@inproceedings{xue2019robust,
  title={Improving robustness of medical image diagnosis with denoising convolutional neural networks},
  author={Xue, Fei-Fei and Peng, Jin and Wang, Ruixuan and Zhang, Qiong and Zheng, Wei-Shi},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={846--854},
  year={2019},
  organization={Springer}
}`

