import pandas as pd
import numpy as np
import os
import torch
from tqdm import tqdm
from models import *
from dataSet.reader import *
from dataSet.transform import *
import os
import shutil
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
num_TTA = 2

import ipdb


def get_df_top20(dist_mat, test_imgs, allnames):
    dist_mat = dist_mat.cpu().numpy()
    dist_mat_sorted = dist_mat.argsort()
    df_top20 = pd.DataFrame(dist_mat_sorted[:, :21])
    df_top20 = df_top20.applymap(lambda x: allnames[x])
    df_top20 = df_top20[df_top20[0].isin(test_imgs)]
    return df_top20

def test(checkPoint_start=0):
    names_test = os.listdir(f'./WC_input/{mode}')
    sample_submission = pd.read_csv('./WC_input/sample_submission.csv', header=None)
    test_imgs = sample_submission.iloc[:, 0].tolist()
    batch_size = 80
    dst_test = WhaleTestDataset()
    dataloader_test = DataLoader(dst_test, batch_size=batch_size, num_workers=0)
    model = HeadWhaleModel()
    if torch.cuda.is_available():
        model = model.cuda()
    resultDir = './WC_result/{}_{}'.format(model_name, fold_index)
    checkPoint = os.path.join(resultDir, 'checkpoint')

    npy_dir = resultDir + '/out_{}'.format(checkPoint_start)
    os.makedirs(npy_dir, exist_ok=True)
    if not checkPoint_start == 0:
        model.load_pretrain(os.path.join(checkPoint, '%08d_model.pth' % (checkPoint_start)),skip=[])
        ckp = torch.load(os.path.join(checkPoint, '%08d_optimizer.pth' % (checkPoint_start)))
        best_t = ckp['best_t']
        print('best_t:', best_t)
    allnames = []
    allresults = []
    with torch.no_grad():
        model.eval()
        for data in tqdm(dataloader_test):
            images, names = data
            if torch.cuda.is_available():
                images = images.cuda()
            results = model(images)
            allresults.append(results)
            for name in names:
                allnames.append(name)
        allresults = torch.cat(allresults)
        
        allresults_np = allresults.cpu().numpy()
        ipdb.set_trace()
        allresults_df = pd.DataFrame(allresults_np, index=allnames)
        
        # .to_csv(os.path.join(npy_dir, 'encoding_org_img.csv'), header=False)
        

        dist_global_org = dist_global[::2, :]
        dist_global_flp = dist_global[1::2, :]
        dist_global_min = np.minimum(dist_global[::2, :], dist_global[1::2, :])

        get_df_top20(dist_global_org, test_imgs, allnames).to_csv(os.path.join(npy_dir, f'submission_{model_name}_sub_fold{fold_index}_org.csv'),
                                                                  header=False, index=False)
        get_df_top20(dist_global_flp, test_imgs, allnames).to_csv(os.path.join(npy_dir, f'submission_{model_name}_sub_fold{fold_index}_flp.csv'),
                                                                  header=False, index=False)
        get_df_top20(dist_global_min, test_imgs, allnames).to_csv(os.path.join(npy_dir, f'submission_{model_name}_sub_fold{fold_index}_min.csv'),
                                                                  header=False, index=False)


if __name__ == '__main__':
    checkPoint_start = 10000
    test(checkPoint_start)

