CUDA_VISIBLE_DEVICES=0 python main.py --mode=train --model=resnet101 --image_h=256 --image_w=512 --fold_index=0 --batch_size=12


# Kaggle Humpback Whale Identification Challenge  1st place code
Add SSH User: eval $(ssh-agent -s) ssh-add ~/.ssh/backup/id_rsa

cat id_rsa.pub | ssh pt-support@whaleip.westeurope.cloudapp.azure.com 'cat >> ~/.ssh/authorized_keys'

ssh pt-support@whaleip.westeurope.cloudapp.azure.com -L 9999:127.0.0.1:9999

jupyter notebook --port 9999 --NotebookApp.token='H3HjxVXi9zIFhxZ' --no-browser

Install custom version of fastai conc git clone https://github.com/fastai/fastai.git cd fastai tools/run-after-git-clone

conda create -n fastai python=3.6 git checkout 1.0.36.post1 pip install -e "[dev]"

conda env create -f environment.yml conda install -c anaconda ipykernel python -m ipykernel install --user --name=firstEnv

Go Back to tmux session> tmux attach -t 0

for f in *.tif; do echo "Converting $f"; convert "$f" "$(basename "$f" .tif).jpg"; done

TMUX

See existing sessions tmux ls

Go back to session
tmux attach -t 0

Start session tmux

Split screen horizontally (1) Strg + b (2) %

Split screen horizontally (1) Strg + b (2) "

Go to next terminal (1) Strg + b (2) arrow keys

checking the usage of GPU watch -n 1 nvidia-smi



#### Recent Updates
[2019.3.1 16:44] uploading mask file
#### Dependencies
- python==3.6
- torch==0.4.1
- torchvision==0.2.1


## Solution  
https://www.kaggle.com/c/humpback-whale-identification/discussion/82366  

### prepare data  
## competition dataset  
trainset    -> ./input/train  
testset    -> ./input/test  

## mask
cd input  
unzip mask.zip  
download model_50A_slim_ensemble.csv(https://drive.google.com/file/d/1hfOu3_JR0vWJkNlRhKwhqJDaF3ID2vRs/view?usp=sharing)  into ./input

## playground data  
download playground data, then put them into input/train  
https://www.kaggle.com/c/whale-categorization-playground/data



### Train  
line 301 in train.py  
step 1.   
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;            freeze = False  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;               model_name = 'senet154'  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;               min_num_class = 10  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;             checkPoint_start = 0  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;             lr = 3e-4  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;             #until train map5 >= 0.98  

step 2.  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;             freeze = True  
 &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;            model_name = 'senet154'  
  &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;           min_num_class = 0  
    &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;         checkPoint_start = best checkPoint of step 1  
     &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;        lr = 3e-4  

step 3.  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;             freeze = True  
 &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;       model_name = 'senet154'  
   &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;     min_num_class = 0  
   &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;     checkPoint_start = best checkPoint of step 2  
   &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;     lr = 3e-5  

### Test  
line 99 in test.py  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;       checkPoint_start = best checkPoint of step 3  

