# Kaggle Sperm Whale Identification Challenge  


### docithurtswhaleIP code



#### Dependencies
- python==3.6
- torch==0.4.1
- torchvision==0.2.1

other dependencies in requierements.txt

## Solution  
Forked and Adapted from:
https://github.com/earhian/Humpback-Whale-Identification-1st-
https://www.kaggle.com/c/humpback-whale-identification/discussion/82366  

### prepare data  
## competition dataset  
trainset    -> ./WC_input/train  
testset    -> ./WC_input/test  

## mask
cd input  
Download training masks 
https://storage.googleapis.com/kaggle-forum-message-attachments/459392/11072/masks.zip
unzip mask.zip  

## playground data  
download humpback whale kaggle data, then put them into input  
https://www.kaggle.com/c/whale-categorization-playground/data

download sperm whale data, flatten the data in single directories train, test, and data (all) then put them into WC_input
https://gdsc.ce.capgemini.com/resources/


### Train Hyperparams  
line 345 in train.py  
step 1.   
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;           freeze = False  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;           model_name = 'seresnext101'  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;           min_num_class = 10  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;           checkPoint_start = 0  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;            lr = 3e-4  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;            #until train map5 >= 0.98  

step 2.   
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;            freeze = True  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;            model_name = 'seresnext101'  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;            min_num_class = 0  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;            checkPoint_start = 0  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;            lr = 3e-4  


step 2.   
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;             freeze = True  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;             model_name = 'seresnext101'  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;             min_num_class = 0  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;             checkPoint_start = 0  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;             lr = 3e-4  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;             test_train=True
 


### Test  
line 130 in test.py  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;       checkPoint_start = best checkPoint of step 2  

