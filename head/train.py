import datetime
import os

from timeit import default_timer as timer
from reader import *

from models import *
import torch
import time
from torch.nn.parallel.data_parallel import data_parallel
import ipdb

from torch.utils.tensorboard import SummaryWriter

def time_to_str(t):
    t  = int(t)
    hr = t//60
    min = t%60
    return '%2d hr %02d min'%(hr,min)

def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def eval(model, dataLoader_valid):
    with torch.no_grad():
        model.eval()
        model.mode = 'valid'
        valid_loss, index_valid = 0, 0
        all_results = []
        all_labels = []
        for valid_data in dataLoader_valid:
            image1, image2, labels  = valid_data
#             ipdb.set_trace()
            if torch.cuda.is_available():
                image1, image2 = image1.cuda().float(), image2.cuda().float()
                labels = labels.cuda().float()
            results = model(image1, image2)
#             output1, output2 = data_parallel(model, image1, image2)
            model.getLoss(results, labels)
            all_results.append(results)
            all_labels.append(labels)
            b = len(labels)
            valid_loss += model.loss.data.cpu().numpy() * b
            index_valid += b
        all_results = torch.cat(all_results, 0)
        all_labels = torch.cat(all_labels, 0)
        valid_loss /= index_valid
        return valid_loss

def train(checkPoint_start=0, lr=3e-4, batch_size=36):
    model = HeadWhaleModel()
    print(model)
    if torch.cuda.is_available():
        model = model.cuda()
    i = 0
    iter_smooth = 50
    iter_valid = 200
    iter_save = 200
    epoch = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,  betas=(0.9, 0.99), weight_decay=0.002)
    resultDir = './WC_result/{}'.format('HeadWhaleModel8')
    writer = SummaryWriter(resultDir)
    ImageDir = resultDir + '/image'
    checkPoint = os.path.join(resultDir, 'checkpoint')
    os.makedirs(checkPoint, exist_ok=True)
    os.makedirs(ImageDir, exist_ok=True)

    dst_train = WhaleDataset()
    dataloader_train = DataLoader(dst_train, shuffle=True, drop_last=True, batch_size=batch_size, num_workers=12)
    
    num_data = len(dst_train.com_train)
    print(dst_train.__len__())
    dst_valid = WhaleValidDataset()
    dataloader_valid = DataLoader(dst_valid, shuffle=False, drop_last=True, batch_size=batch_size, num_workers=10)
    train_loss = 0.0
    valid_loss = 0.0

    batch_loss = 0.0
    train_loss_sum = 0

    sum = 0
    skips = []
    if not checkPoint_start == 0:
        model.load_pretrain(os.path.join(checkPoint, '%08d_model.pth' % (checkPoint_start)),skip=skips)
        ckp = torch.load(os.path.join(checkPoint, '%08d_optimizer.pth' % (checkPoint_start)))
        optimizer.load_state_dict(ckp['optimizer'])
        adjust_learning_rate(optimizer, lr)
        i = checkPoint_start
        epoch = ckp['epoch']
    start = timer()
    epoch = 0
    i = 0
    start_epoch = epoch
    while i < 10000000:
        for data in dataloader_train:
            epoch = start_epoch + (i - checkPoint_start) * 4 * batch_size/num_data
            if i % iter_valid == 0:
                valid_loss = eval(model, dataloader_valid)
                writer.add_scalars('loss', {'train_loss': train_loss,
                                           'val_loss': valid_loss
                                          },
                                epoch * len(dataloader_train) + i)
                
            if i % iter_save == 0 and not i == checkPoint_start:
                torch.save(model.state_dict(), resultDir + '/checkpoint/%08d_model.pth' % (i))
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'iter': i,
                    'epoch': epoch,
                }, resultDir + '/checkpoint/%08d_optimizer.pth' % (i))


            model.train()

            model.mode = 'train'
            image1, image2, labels  = data
#             ipdb.set_trace()
            if torch.cuda.is_available():
                image1, image2 = image1.cuda().float(), image2.cuda().float()
                labels = labels.cuda().float()
            output = model(image1, image2)
            model.getLoss(output, labels)
            batch_loss = model.loss

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
            optimizer.step()
            batch_loss = batch_loss.data.cpu().numpy()
            sum += 1
            train_loss_sum += batch_loss
            if (i + 1) % iter_smooth == 0:
                train_loss = train_loss_sum/sum
            i += 1

        pass


if __name__ == '__main__':
    if 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        checkPoint_start = 0
        lr = 0.0001 # 3e-4
        batch_size = 32
        train(checkPoint_start, lr, batch_size)