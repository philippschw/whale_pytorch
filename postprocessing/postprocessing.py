import pandas as pd
from glob import glob
from pathlib import Path

from matplotlib import pyplot as plt
import cv2
import re
import ipdb


fp = './WC_input/'
rp = './WC_result/'


train = pd.read_csv(fp+'train.csv') 
image_to_id = dict(zip(train.Image, train.Id))

train = train.groupby('Id')['Image'].agg({lambda x: set(x)})
id_to_images = dict(zip(train.index, train.iloc[:, 0]))


def inject_train_class_img(row):
    ref = row[0]
    s = row[1:].copy()
    start = 0
    result = []
    for i, img in enumerate(row[1:]):
        try:
            imgs_same_class = list(id_to_images[image_to_id[img]])
            imgs_same_class = [e for e in imgs_same_class if e not in result]
            result.extend(imgs_same_class)                
        except:
            result.extend([img])

    return ([ref] + result)[:21]
    
def inject_matched_img(row, dic_pairs):
    ref = row[0]
    s = row[1:].copy()
    if ref in dic_pairs.keys():
        imgs_same_class = dic_pairs[ref]
        s_list = s.tolist()
        if imgs_same_class in s_list:
            s_list.remove(imgs_same_class)
        result = [row[0]] + [imgs_same_class] + s_list
        result = pd.Series(result)
    else:
        result = row
    return result[:21]

class Post_Pipeline(object):
    def __init__(self, model_name, fold, checkpoint, algo, df=False):
        
        self.model_name = model_name
        self.fold = fold
        self.checkpoint = checkpoint
        self.algo = algo
        if isinstance(df, pd.DataFrame):
            self.pre_df = df.set_index(0)
        else:
            self.pre_df = pd.read_csv(rp + f'{self.model_name}_{self.fold}/out_{self.checkpoint}/{self.model_name}_sub_fold{self.fold}_{self.algo}.csv', header=None).set_index(0)

        
    def save_result(self):
        self.result.to_csv(rp + f'{self.model_name}_{self.fold}/out_{self.checkpoint}/postprocessingII_{self.model_name}_sub_fold{self.fold}_{self.algo}.csv',
                      header=None, index=True)
       
    def infer_missing_from_train_set(self, df=False):
        if isinstance(df, pd.DataFrame):
            ma = df.copy()
        else:
            ma = self.pre_df.copy()
            
        if ma.shape[1] == 21:
            ma = ma.set_index(0)
        elif ma.shape[1] == 20:
            pass
        else:
            print ('NON Valid DataFrame')
#         ipdb.set_trace()
        ma = ma.reset_index()
        ma.columns = list(range(ma.shape[1]))
        
        ma['new'] = ma.apply(lambda x: inject_train_class_img(x), axis=1)
        return ma['new'].apply(pd.Series).set_index(0)
        
    def match_images_on_fname(self, df=False):
        if isinstance(df, pd.DataFrame):
            ma = df.copy()
        else:
            ma = self.pre_df.copy()
            
        if ma.shape[1] == 21:
            ma = ma.set_index(0)
        elif ma.shape[1] == 20:
            pass
        else:
            print ('NON Valid DataFrame')
            
        test_imgs = glob(fp + 'test/**.jpg')
        test_imgs = [Path(img).name for img in test_imgs]
        test_imgs  = pd.DataFrame(test_imgs)
        test_imgs['Id'] =  test_imgs[0].str.rsplit('-', 1).map(lambda x: x[1].split('.')[0])
        test_imgs['date'] = test_imgs[0].str.rsplit('-', 1).map(lambda x: x[0].rsplit('-')[2])
        test_imgs['img_id'] = test_imgs['date'] + test_imgs['Id']
        test_imgs['img_id'] = test_imgs['img_id'].map(lambda x: re.sub('\D', '', x))
        test_imgs[test_imgs['img_id'].map(lambda x: 'A' in  x)]
        test_imgs['img_id'] = test_imgs['img_id'].astype(int)
        test_imgs_minusone = test_imgs['img_id']-1
        test_imgs_plusone =  test_imgs['img_id']+1
        imgs_id_matched = test_imgs.merge(test_imgs_minusone, how='inner', on='img_id')
        imgs_id_matched['img_id_org'] = imgs_id_matched['img_id'] + 1
        self.imgs_id_matched = imgs_id_matched.merge(test_imgs, left_on='img_id_org', right_on='img_id')

        a = self.imgs_id_matched[['0_x', '0_y']]

        b = self.imgs_id_matched[['0_y', '0_x']]

        a.columns = ['x', 'y']
        b.columns = ['x', 'y']

        pairs = a.drop(3).append(b.drop(3)).reset_index(drop=True)

        dic_pairs = dict(zip(pairs.x, pairs.y))
        
        collector_df = []
        ma = ma.reset_index()
        ma.columns = list(range(ma.shape[1]))
        for i in range(ma.shape[0]):
            row = ma.iloc[i]
            collector_df.append(inject_matched_img(row, dic_pairs))

        result = pd.concat(collector_df, axis=1)
        result = result.T
        return result.set_index(0)

    def view_matched_images(self):
        f, axarr = plt.subplots(self.imgs_id_matched.shape[0], 2)
        f.set_figwidth(10)
        f.set_figheight(50)

        for i in range(self.imgs_id_matched.shape[0]):

            # fetch the url as a file type object, then read the image
            f1 = self.imgs_id_matched.loc[i, '0_x']
            f2 = self.imgs_id_matched.loc[i, '0_y']

            a =  plt.imread(fp + 'test/'+f1)
            b =  plt.imread(fp + 'test/'+f2)
            a = cv2.resize(a, (384, 192))
            b = cv2.resize(b, (384, 192))

            axarr[i,0].imshow(a)
            axarr[i,1].imshow(b)

            axarr[i,0].axis('off')
            axarr[i,1].axis('off')
            axarr[i,0].set_title(f1)
            axarr[i,1].set_title(f2)
        plt.tight_layout()
        plt.show()
        
        
    def show_top20_imgs(self, img_name, df=False):
        if isinstance(df, pd.DataFrame):
            ma = df.copy()
        else:
            ma = self.pre_df.copy()
            
        if ma.shape[1] == 21:
            ma = ma.set_index(0)
        elif ma.shape[1] == 20:
            pass
        else:
            print ('NON Valid DataFrame')
            
        img_ref_name = img_name
        print (img_ref_name)
        f, axarr = plt.subplots(5, 5)
        f.set_figwidth(20)
        f.set_figheight(15)

        for i in range(0, 20):
            img_name = ma.loc[img_ref_name, i+1]
            print (i+1)
            print (img_name)
            img =  plt.imread(fp +'data/'+img_name)

            axarr[int(i/5)+1, i%5].imshow(img)
            axarr[int(i/5)+1, i%5].axis('off')
            if img_name in image_to_id.keys():
                axarr[int(i/5)+1, i%5].set_title(f'{img_name}-{image_to_id[img_name]}')
            else:
                axarr[int(i/5)+1, i%5].set_title(f'{img_name}-')
        img = plt.imread(fp + 'data/'+img_ref_name)
        axarr[0, 0].imshow(img)
        axarr[0, 0].axis('off')
        axarr[0, 0].set_title(img_ref_name)
        axarr[0, 1].axis('off')
        axarr[0, 2].axis('off')
        axarr[0, 3].axis('off')
        axarr[0, 4].axis('off')

        plt.show()