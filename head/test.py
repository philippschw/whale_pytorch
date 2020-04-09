import pandas as pd
import numpy as np
import os
import torch
from tqdm import tqdm
from models import *
from reader import *
import os
import shutil
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
num_TTA = 2

import ipdb

def test(checkPoint_start=0, margin=1):
    sample_submission = pd.read_csv('../WC_input/sample_submission.csv', header=None)
    test_imgs = sample_submission.iloc[:, 0].tolist()
    batch_size = 1200
    dst_test = WhaleTestDataset()
    dataloader_test = DataLoader(dst_test, batch_size=batch_size, num_workers=18)
    model = HeadWhaleModel()
    if torch.cuda.is_available():
        model = model.cuda()
    resultDir = './WC_result/{}'.format('HeadWhaleModel')
    checkPoint = os.path.join(resultDir, 'checkpoint')

    npy_dir = resultDir + '/out_{}'.format(checkPoint_start)
    os.makedirs(npy_dir, exist_ok=True)
    if not checkPoint_start == 0:
        model.load_pretrain(os.path.join(checkPoint, '%08d_model.pth' % (checkPoint_start)),skip=[])
        ckp = torch.load(os.path.join(checkPoint, '%08d_optimizer.pth' % (checkPoint_start)))
    allnames = []
    allresults = []
    with torch.no_grad():
        model.eval()
        for data in tqdm(dataloader_test):
            images, names = data
            if torch.cuda.is_available():
                images = images.cuda().float()
            output1, output2 = model(images)
            results = threashold_contrastive_loss(output1, output2, margin)
            allresults.append(results)
            for name in names:
                allnames.append(name)
        allresults = torch.cat(allresults)
        
        allresults_np = allresults.cpu().numpy()
#         ipdb.set_trace()
        allresults_df = pd.DataFrame(allresults_np, index=allnames)
        
        allresults_df = allresults_df.reset_index()

        allresults_df.columns = ['names', 'target']

        allresults_df[['name1', 'name2']] = allresults_df['names'].str.split('|',1, expand=True)

        df = allresults_df.sort_values('target',ascending = False).groupby('name1').head(20)
        df = df[['name1', 'name2']]
        
        dic = {}
        for ref in df.name1.unique():
            dic[ref] = df[df.name1==ref].name2.tolist()
        df = pd.DataFrame(dic).T
        df.to_csv(os.path.join(npy_dir, 'submission_header_model.csv'), header=False)

if __name__ == '__main__':
    checkPoint_start = 10000
    margin = 1
    test(checkPoint_start)

