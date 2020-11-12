import numpy as np 
import os 

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
import pandas as pd
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader 
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision.models as models
from torchvision import transforms
from tensorboardX import SummaryWriter

from emotic import Emotic 
from emotic_dataset import Emotic_PreDataset, Emotic_CSVDataset
from loss import DiscreteLoss, ContinuousLoss_SL1, ContinuousLoss_L2
from prepare_models import prep_models
from test import test_data
from utils import get_weight_of_samples


def append_log_to_file(file_path, line):
    with open(file_path, 'a') as opened_file:
        opened_file.write(line+'\n')
        opened_file.close()

def train_data(opt, scheduler, models, device, train_loader, val_loader, 
                disc_loss, cont_loss, train_writer, val_writer, train_log_path, 
                val_log_path, model_path, args):
    '''
    Training emotic model on train data using train loader.
    :param opt: Optimizer object.
    :param scheduler: Learning rate scheduler object.
    :param models: List containing model_context, model_body and emotic_model (fusion model) in that order. 
    :param device: Torch device. Used to send tensors to GPU if available. 
    :param train_loader: Dataloader iterating over train dataset. 
    :param val_loader: Dataloader iterating over validation dataset. 
    :param disc_loss: Discrete loss criterion. Loss measure between discrete emotion categories predictions and the target emotion categories. 
    :param cont_loss: Continuous loss criterion. Loss measure between continuous VAD emotion predictions and the target VAD values.
    :param train_writer: SummaryWriter object to save train logs. 
    :param val_writer: SummaryWriter object to save validation logs. 
    :param model_path: Directory path to save the models after training. 
    :param args: Runtime arguments.
    '''
    
    model_context, model_body, emotic_model = models

    emotic_model.to(device)
    model_context.to(device)
    model_body.to(device)

    print ('starting training')
    log_train_path = os.path.join(train_log_path, 'train.txt')
    log_val_path = os.path.join(val_log_path, 'val.txt')

    for e in range(args.epochs):

        running_loss = 0.0 
        running_cat_loss = 0.0 
        running_cont_loss = 0.0
        
        emotic_model.train()
        model_context.train()
        model_body.train()
        
        #train models for one epoch 
        n_train_batches = len(train_loader)
        for idx, (images_context, images_body, labels_cat, labels_cont) in \
                enumerate(train_loader):
            images_context = images_context.to(device)
            images_body = images_body.to(device)
            labels_cat = labels_cat.to(device)
            labels_cont = labels_cont.to(device)

            opt.zero_grad()

            pred_context = model_context(images_context)
            pred_body = model_body(images_body)

            pred_cat, pred_cont = emotic_model(pred_context, pred_body)
            cat_loss_batch = disc_loss(pred_cat, labels_cat)
            cont_loss_batch = cont_loss(pred_cont * 10, labels_cont * 10)

            loss = (args.cat_loss_weight * cat_loss_batch) + \
                        (args.cont_loss_weight * cont_loss_batch)
            
            running_loss += loss.item()
            running_cat_loss += cat_loss_batch.item()
            running_cont_loss += cont_loss_batch.item()
            
            loss.backward()
            opt.step()
            if idx % 20 == 0:
                log_train_bch_mess = 'Train on epoch {}: {}/{}, loss: {:.4f}, cat loss: {:.4f}, cont_loss {:.4f}'.format(e+1, 
                        idx, n_train_batches, running_loss/(idx+1), 
                        running_cat_loss/(idx+1), running_cont_loss/(idx+1))
                print(log_train_bch_mess)
                append_log_to_file(log_train_path, log_train_bch_mess)

        if e % 1 == 0:
            log_train_ep_mess = 'epoch = %d, loss = %.4f, cat loss = %.4f, cont_loss = %.4f' %(e+1, 
                                    running_loss/n_train_batches, 
                                    running_cat_loss/n_train_batches, 
                                    running_cont_loss/n_train_batches)
            print(log_train_ep_mess)
            append_log_to_file(log_train_path, log_train_ep_mess)
            save_checkpoint(emotic_model, model_context, model_body, opt, e, 
                                args, model_path)

        train_writer.add_scalar('losses/total_loss', running_loss, e)
        train_writer.add_scalar('losses/categorical_loss', running_cat_loss, e)
        train_writer.add_scalar('losses/continuous_loss', running_cont_loss, e)
        
        running_loss = 0.0 
        running_cat_loss = 0.0 
        running_cont_loss = 0.0 
        
        emotic_model.eval()
        model_context.eval()
        model_body.eval()
        
        n_val_batches = len(val_loader)

        emotic_model.to(device)
        model_context.to(device)
        model_body.to(device)

        with torch.no_grad():
            #validation for one epoch
            for idx, (images_context, images_body, labels_cat, labels_cont) in \
                enumerate(val_loader):
                images_context = images_context.to(device)
                images_body = images_body.to(device)
                labels_cat = labels_cat.to(device)
                labels_cont = labels_cont.to(device)

                pred_context = model_context(images_context)
                pred_body = model_body(images_body)

                pred_cat, pred_cont = emotic_model(pred_context, pred_body)
                cat_loss_batch = disc_loss(pred_cat, labels_cat)
                cont_loss_batch = cont_loss(pred_cont * 10, labels_cont * 10)
                loss = (args.cat_loss_weight * cat_loss_batch) + \
                            (args.cont_loss_weight * cont_loss_batch)
                
                running_loss += loss.item()
                running_cat_loss += cat_loss_batch.item()
                running_cont_loss += cont_loss_batch.item()
                
                if idx % 20 == 0:
                  log_val_bch_mess = 'Validate after epoch {}: {}/{}, loss: {:.4f}, cat loss: {:.4f}, cont_loss {:.4f}'.format(e+1, 
                        idx, n_val_batches, loss/(idx+1), cat_loss_batch/(idx+1), 
                        cont_loss_batch/(idx+1))
                  
                  print(log_val_bch_mess)
                  append_log_to_file(log_val_path, log_val_bch_mess)
        
        if e % 1 == 0:
            log_val_ep_mess = 'epoch = %d, validation loss = %.4f, cat loss = %.4f, cont loss = %.4f ' %(e+1, 
                                running_loss/n_val_batches, 
                                running_cat_loss/n_val_batches, 
                                running_cont_loss/n_val_batches)
                                
            print(log_val_ep_mess)
            append_log_to_file(log_val_path, log_val_ep_mess)

        # step learning rate scheduler
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(running_loss/n_val_batches)

        if isinstance(scheduler, StepLR):
            scheduler.step()
            
        val_writer.add_scalar('losses/total_loss', running_loss, e)
        val_writer.add_scalar('losses/categorical_loss', running_cat_loss, e)
        val_writer.add_scalar('losses/continuous_loss', running_cont_loss, e)
        
    print ('completed training')


def save_checkpoint(emotic_model, model_context, model_body, optimizer, epoch, 
                        args, model_path):
    emotic_model.to("cpu")
    model_context.to("cpu")
    model_body.to("cpu")
    cp_path = os.path.join(model_path, 'checkpoint_ep{}.pth'.format(epoch+1))
    state = {
        'archs': 'mlp {} {}'.format(type(model_context), type(model_body)),
        'epoch': epoch + 1,
        'state_dicts':{
            'emotic_model': emotic_model.state_dict(),
            'context_model': model_context.state_dict(),
            'body_model': model_body.state_dict()
        },
        'optimizer': optimizer.state_dict(),
        'args': args.__dict__
    }
    torch.save(state, cp_path)
    print ('saved models after epoch {}'.format(epoch + 1))


def train_emotic(result_path, model_path, train_log_path, val_log_path, ind2cat, 
                    ind2vad, context_norm, body_norm, args):
    ''' Prepare dataset, dataloders, models. 
    :param result_path: Directory path to save the results (val_predidictions mat object, val_thresholds npy object).
    :param model_path: Directory path to load pretrained base models and save the models after training. 
    :param train_log_path: Directory path to save the training logs. 
    :param val_log_path: Directoty path to save the validation logs. 
    :param ind2cat: Dictionary converting integer index to categorical emotion. 
    :param ind2vad: Dictionary converting integer index to continuous emotion dimension (Valence, Arousal and Dominance).
    :param context_norm: List containing mean and std values for context images. 
    :param body_norm: List containing mean and std values for body images. 
    :param args: Runtime arguments. 
    '''
    # Load preprocessed data from npy files
    # train_context = np.load(os.path.join(args.data_path, 'train_context_arr.npy'))
    # train_body = np.load(os.path.join(args.data_path, 'train_body_arr.npy'))
    # train_cat = np.load(os.path.join(args.data_path, 'train_cat_arr.npy'))
    # train_cont = np.load(os.path.join(args.data_path, 'train_cont_arr.npy'))

    # val_context = np.load(os.path.join(args.data_path, 'val_context_arr.npy'))
    # val_body = np.load(os.path.join(args.data_path, 'val_body_arr.npy'))
    # val_cat = np.load(os.path.join(args.data_path, 'val_cat_arr.npy'))
    # val_cont = np.load(os.path.join(args.data_path, 'val_cont_arr.npy'))

    # print ('train ', 'context ', train_context.shape, 'body', train_body.shape, 'cat ', train_cat.shape, 'cont', train_cont.shape)
    # print ('val ', 'context ', val_context.shape, 'body', val_body.shape, 'cat ', val_cat.shape, 'cont', val_cont.shape)

    # Initialize Dataset and DataLoader 
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), 
                        transforms.ColorJitter(brightness=0.4, contrast=0.4, 
                            saturation=0.4), transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])

    # train_dataset = Emotic_PreDataset(train_context, train_body, train_cat, train_cont, train_transform, context_norm, body_norm)
    # val_dataset = Emotic_PreDataset(val_context, val_body, val_cat, val_cont, test_transform, context_norm, body_norm)

    cat2ind = {}
    for k, v in ind2cat.items():
      cat2ind[v] = k

    train_df = pd.read_csv(os.path.join(args.data_path, 'train.csv'))
    val_df = pd.read_csv(os.path.join(args.data_path, 'val.csv'))

    train_dataset = Emotic_CSVDataset(train_df, cat2ind, train_transform, 
                      context_norm, body_norm, args.data_path)

    sample_weights = get_weight_of_samples(train_df, args.weight_classes_file, 
                        cat2ind)

    balanced_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    val_dataset = Emotic_CSVDataset(val_df, cat2ind, test_transform, 
                      context_norm, body_norm, args.data_path)
    
    train_loader = DataLoader(train_dataset, args.batch_size, 
                                sampler=balanced_sampler)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)

    print ('train loader ', len(train_loader), 'val loader ', len(val_loader))

    # Prepare models 
    model_context, model_body = prep_models(context_model=args.context_model, 
                                    body_model=args.body_model, 
                                    model_dir=model_path)


    emotic_model = Emotic(list(model_context.children())[-1].in_features, 
                            list(model_body.children())[-1].in_features)
    model_context = nn.Sequential(*(list(model_context.children())[:-1]))
    model_body = nn.Sequential(*(list(model_body.children())[:-1]))

    opt = optim.Adam((list(emotic_model.parameters()) + \
                        list(model_context.parameters()) + \
                        list(model_body.parameters())), 
                        lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.resume_from_ep > 0:
        cp_path = os.path.join(model_path, 'checkpoint_ep{}.pth'.format(args.resume_from_ep))
        cp = torch.load(cp_path)
        state_dicts = cp['state_dicts']
        emotic_model.load_state_dict(state_dicts['emotic_model'])
        model_context.load_state_dict(state_dicts['context_model'])
        model_body.load_state_dict(state_dicts['body_model'])
        opt.load_state_dict(cp['optimizer'])

    for param in emotic_model.parameters():
        param.requires_grad = True
    for param in model_context.parameters():
        param.requires_grad = True
    for param in model_body.parameters():
        param.requires_grad = True
    
    device = torch.device("cuda:%s" %(str(args.gpu)) if torch.cuda.is_available() \
                            else "cpu")

    # scheduler = StepLR(opt, step_size=7, gamma=0.1)
    scheduler_args = {
            "mode": "min",
            "threshold": 0.02,
            "factor": 0.5,
            "patience": 1,
            "verbose": True,
            "min_lr": 1e-05,
            "threshold_mode": "rel"
        }
    scheduler = ReduceLROnPlateau(opt, **scheduler_args)
    
    disc_loss = DiscreteLoss(args.discrete_loss_weight_type, device)
    if args.continuous_loss_type == 'Smooth L1':
        cont_loss = ContinuousLoss_SL1()
    else:
        cont_loss = ContinuousLoss_L2()

    train_writer = SummaryWriter(train_log_path)
    val_writer = SummaryWriter(val_log_path)

    # training
    train_data(opt, scheduler, [model_context, model_body, emotic_model], 
                device, train_loader, val_loader, disc_loss, cont_loss, 
                train_writer, val_writer, train_log_path, val_log_path, 
                model_path, args)
    # validation
    test_data([model_context, model_body, emotic_model], device, val_loader, 
                ind2cat, ind2vad, len(val_dataset), result_dir=result_path, 
                test_type='val')
