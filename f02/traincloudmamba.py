import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from models.cloud import cdMamba
from engine import *
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from dataprepare.tifsemi import SemiDataset
from utils import *
from configs.config_setting import setting_config
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import warnings
warnings.filterwarnings("ignore")


def main(config):
    net = 'cloudmamba'
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir +net+  '/')
    log_dir = os.path.join(config.work_dir + net+  '/', 'log')
    checkpoint_dir = os.path.join(config.work_dir +  '/'+ net+  '/', 'checkpoints')
    resume_model = os.path.join(checkpoint_dir+  '/', 'latest.pth')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    global logger
    logger = get_logger('train', log_dir)

    log_config_info(config, logger)





    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]# [0, 1, 2, 3]
    torch.cuda.empty_cache()





    print('#----------Preparing dataset----------#')
    trainset_l = SemiDataset('/data/boime', 'train_l', '/data/boime/train/patch_in_456.txt')
    valset = SemiDataset( '/data/boime', 'val', '/data/boime/test/patch_in_456.txt')

    
    train_loader = DataLoader(trainset_l, batch_size=config.batch_size,
                               pin_memory=False, num_workers=0, drop_last=True, sampler=None)
   
    
    val_loader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False)
    test_loader = val_loader





    print('#----------Prepareing Models----------#')

    model = cdMamba(num_classes=1,input_channels=4).cuda()

    model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])







    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()





    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1
    best_miou = 0




    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch = checkpoint['min_loss'], checkpoint['min_epoch']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}'
        logger.info(log_info)




    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            logger,
            config,
            scaler=scaler
        )

        miou = val_one_epoch(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                config
            )


        if miou > best_miou:
            if best_miou != 0:
                os.remove(os.path.join(checkpoint_dir, '%.5f.pth'%(best_miou)))
            torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, '%.5f.pth'%(miou)))

            best_miou = miou

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir,  'latest.pth') )

if __name__ == '__main__':
    config = setting_config
    main(config)