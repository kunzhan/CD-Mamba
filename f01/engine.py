import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs

def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    logger, 
                    config, 
                    scaler=None):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
        if config.amp:
            with autocast():
                out = model(images)
                loss = criterion(out, targets)      
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step() 


def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config):
    # switch to evaluate mode
    model.eval()

    TN = 0
    FP = 0
    FN = 0
    TP = 0

    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out = model(img)
            msk= msk.squeeze(1).cpu().detach().numpy()
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds = np.array(out).reshape(-1)
            gts = np.array(msk).reshape(-1)

            y_pre = np.where(preds>=0.5, 1, 0)
            y_true = np.where(gts>=0.5, 1, 0)
            tp = (y_pre * y_true).sum()
            tn = ((1 - y_pre) * (1 - y_true)).sum() 
            fp = (y_pre * (1 - y_true)).sum() 
            fn = ((1 - y_pre) * y_true).sum()

            TN += tn
            FP += fp
            FN += fn
            TP += tp


    accuracy = float(TN + TP) / float(TN+ TP + FP + FN)
    
    f1_or_dsc = float(2 * TP) /  (2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

    log_info = f'val epoch: {epoch},  miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}'
    print(log_info)
    logger.info(log_info)
    
    return miou


def v_one_epoch(test_loader,
                    model,
                    epoch, 
                    logger,
                    ):
    # switch to evaluate mode
    model.eval()

    TN = 0
    FP = 0
    FN = 0
    TP = 0

    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out = model(img)
            msk= msk.squeeze(1).cpu().detach().numpy()
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds = np.array(out).reshape(-1)
            gts = np.array(msk).reshape(-1)

            y_pre = np.where(preds>=0.5, 1, 0)
            y_true = np.where(gts>=0.5, 1, 0)
            tp = (y_pre * y_true).sum()
            tn = ((1 - y_pre) * (1 - y_true)).sum() 
            fp = (y_pre * (1 - y_true)).sum() 
            fn = ((1 - y_pre) * y_true).sum()

            TN += tn
            FP += fp
            FN += fn
            TP += tp


    accuracy = float(TN + TP) / float(TN+ TP + FP + FN)
    
    f1_or_dsc = float(2 * TP) /  (2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

    log_info = f'val epoch: {epoch},  miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}'
    print(log_info)
    logger.info(log_info)
    
    return accuracy


