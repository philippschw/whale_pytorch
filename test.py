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
import ipdb
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
num_TTA = 2

import ipdb

def train_collate(batch):
#     ipdb.set_trace()
    batch_size = len(batch)
    images = []
    labels = []
    names = []
    for b in range(batch_size):
        if batch[b][0] is None:
            continue
        else:
            images.extend(batch[b][0])
            labels.append(batch[b][1])
            names.append(batch[b][2])
    images = torch.stack(images, 0)
    labels = torch.from_numpy(np.array(labels))
    return images, labels, names


def transform(image, mask):
    raw_iamge = cv2.resize(image, (512, 256))
    raw_mask = cv2.resize(mask, (512, 256))
    raw_mask = raw_mask[:, :, None]
    raw_iamge = np.concatenate([raw_iamge, raw_mask], 2)
    images = []

    image = raw_iamge.copy()
    image = np.transpose(image, (2, 0, 1))
    image = image.copy().astype(np.float)
    image = torch.from_numpy(image).div(255).float()
    images.append(image)

    image = raw_iamge.copy()
    image = np.fliplr(image)
    image = np.transpose(image, (2, 0, 1))
    image = image.copy().astype(np.float)
    image = torch.from_numpy(image).div(255).float()
    images.append(image)

    return images

def get_df_top20(dist_mat, test_imgs, allnames):
    dist_mat = dist_mat.cpu().numpy()
    dist_mat_sorted = dist_mat.argsort()
    df_top20 = pd.DataFrame(dist_mat_sorted[:, :21])
    df_top20 = df_top20.applymap(lambda x: allnames[x])
    df_top20 = df_top20[df_top20[0].isin(test_imgs)]
    return df_top20

def test(checkPoint_start=0, fold_index=1, model_name='senet154'):
    mode = 'data'
    names_test = os.listdir(f'./WC_input/{mode}')
    sample_submission = pd.read_csv('./WC_input/sample_submission.csv', header=None)
    test_imgs = sample_submission.iloc[:, 0].tolist()
    batch_size = 80
    dst_test = WhaleTestDataset(names_test, mode=mode, transform=transform)
    dataloader_test = DataLoader(dst_test, batch_size=batch_size, num_workers=0, collate_fn=train_collate)
    label_id = dst_test.labels_dict
    id_label = {v:k for k, v in label_id.items()}
    # id_label[2233] = '-1'
    model = model_whale(num_classes=2233 * 2, inchannels=4, model_name=model_name).cuda()
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
    global_feats = []
    local_feats  = []
    with torch.no_grad():
        model.eval()
        for data in tqdm(dataloader_test):
#             ipdb.set_trace()
            images, labels, names = data
            images = images.cuda()
            labels = labels.cuda()
            global_feat, local_feat, _ = model(images, labels)
            global_feats.append(global_feat)
            local_feats.append(local_feat)
            for name in names:
                allnames.append(name)
        all_global_feat = torch.cat(global_feats)
        all_local_feat = torch.cat(local_feats)
        
        all_global_feat_np = all_global_feat.cpu().numpy()
        all_local_feat_np = all_local_feat.cpu().numpy().reshape(-1, 4096)
        
        pd.DataFrame(all_global_feat_np[::2, :], index=allnames).to_csv(os.path.join(npy_dir, 'encoding_org_img.csv'), header=False)
        pd.DataFrame(all_global_feat_np[1::2, :], index=allnames).to_csv(os.path.join(npy_dir, 'encoding_flp_img.csv'), header=False)
#         ipdb.set_trace()
        pd.DataFrame(all_local_feat_np[::2, :], index=allnames).to_csv(os.path.join(npy_dir, 'local_encoding_org_img.csv'), header=False)
        pd.DataFrame(all_local_feat_np[1::2, :], index=allnames).to_csv(os.path.join(npy_dir, 'local_encoding_flp_img.csv'), header=False)
#         ipdb.set_trace()
        dist_local = euclidean_dist(all_local_feat.reshape(-1, 4096), all_local_feat.reshape(-1, 4096))
        dist_global = euclidean_dist(all_global_feat, all_global_feat)

        dist_global_org = dist_global[::2, ::2]
        dist_global_flp = dist_global[1::2, 1::2,]
        dist_global_min = np.minimum(dist_global[::2, ::2], dist_global[1::2, 1::2])

        dist_local_org = dist_local[::2, ::2]
        dist_local_flp = dist_local[1::2, 1::2]
        dist_local_min = np.minimum(dist_local[::2, ::2], dist_local[1::2, 1::2])
#         ipdb.set_trace()
        get_df_top20(dist_global_org, test_imgs, allnames).to_csv(os.path.join(npy_dir, f'submission_global_{model_name}_sub_fold{fold_index}_org.csv'),
                                                                  header=False, index=False)
        get_df_top20(dist_global_flp, test_imgs, allnames).to_csv(os.path.join(npy_dir, f'submission_global_{model_name}_sub_fold{fold_index}_flp.csv'),
                                                                  header=False, index=False)
        get_df_top20(dist_global_min, test_imgs, allnames).to_csv(os.path.join(npy_dir, f'submission_global_{model_name}_sub_fold{fold_index}_min.csv'),
                                                                  header=False, index=False)

        get_df_top20(dist_local_org, test_imgs, allnames).to_csv(os.path.join(npy_dir, f'submission_local_{model_name}_sub_fold{fold_index}_org.csv'),
                                                                  header=False, index=False)
        get_df_top20(dist_local_flp, test_imgs, allnames).to_csv(os.path.join(npy_dir, f'submission_local_{model_name}_sub_fold{fold_index}_flp.csv'),
                                                                  header=False, index=False)
        get_df_top20(dist_local_min, test_imgs, allnames).to_csv(os.path.join(npy_dir, f'submission_local_{model_name}_sub_fold{fold_index}_min.csv'),
                                                                  header=False, index=False)
        
if __name__ == '__main__':
    checkPoint_start = 8600
    fold_index = 1
    model_name = 'se_resnet50'
    test(checkPoint_start, fold_index, model_name)

