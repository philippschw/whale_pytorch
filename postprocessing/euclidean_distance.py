import faiss

import pandas as pd
import numpy as np
from time import time
from tqdm import tqdm
tqdm.pandas()

fp = './WC_input/'
rp = './WC_result/'

sample_submission = pd.read_csv(rp+'sample_submission.csv', header=None)
sample_submission = sample_submission.set_index(0)

train = pd.read_csv(fp+'train.csv') 
image_to_id = dict(zip(train.Image, train.Id))

class ExactIndex():
    def __init__(self, vectors, labels):
        self.dimension = vectors.shape[1]
        self.vectors = np.ascontiguousarray(vectors.astype('float32'))
        self.labels = labels    
   
    def build(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.vectors)
        
    def add_img(self, vectors, allnames):
        self.labels = np.append(self.labels, np.array(allnames))
        self.index.add(np.ascontiguousarray(vectors.astype('float32')))
    
    def query(self, vectors, k=10):
        distances, indices = self.index.search(vectors, k) 
        return distances, [self.labels[i] for i in indices[0]]
    
class L2(object):
    def __init__(self, model_name, fold, checkpoint):
        self.model_name = model_name
        self.fold = fold
        self.checkpoint = checkpoint
        self.load_encoding()
        self.enc2index()
        
    def load_encoding(self):
        enc = pd.read_csv(rp+f'{self.model_name}_{self.fold}/out_{self.checkpoint}/encoding_org_img.csv', header=None)
        enc = enc.set_index(0)
        enc['embeddings'] = enc.values.tolist()
        enc = enc.reset_index()
        enc = enc.iloc[:, [0, 2050-1]]
        enc.columns = ['Image', 'embeddings']
        enc['Id'] = enc.Image.map(image_to_id)
        enc = enc[enc.Id!='-1']    
        self.enc = enc.reset_index(drop=True)    

    def enc2index(self):
        self.index = ExactIndex(self.enc["embeddings"].apply(pd.Series).values, self.enc["Image"].values)
        self.index.build()
        
    def querytopnclosest(self, n):
        self.n = n
        self.dat = np.ascontiguousarray(self.enc["embeddings"].apply(pd.Series).values).astype('float32')
        
        test = self.enc[self.enc.Image.isin(sample_submission.index)]
        test = test.reset_index() 
        test.columns = ['id', 'Image' , 'embeddings', 'class']
        
        test['top20dist'] = np.nan
        test['top20imgs'] = np.nan
        
        test['top20imgs'] = test['id'].map(lambda x: self.index.query(np.expand_dims(self.dat[x], 1).reshape(1, -1), n+1))
    
        test['top20dist'] = test['top20imgs'].map(lambda x: x[0][0][1:])
        test['top20imgs'] = test['top20imgs'].map(lambda x: x[1][1:])
        
        self.img_dist_class = pd.concat([test['top20imgs'].apply(pd.Series), 
                             test['top20dist'].apply(pd.Series),
                             test['top20imgs'].apply(pd.Series).applymap(lambda x: image_to_id[x] if x in image_to_id.keys() else np.nan)
                            ], axis=1, keys=["img", "dist", "class"])
        
        self.test = pd.DataFrame(test.Image).join(test['top20imgs'].apply(pd.Series))
        self.test.columns = list(range(n+1))
        
        
        