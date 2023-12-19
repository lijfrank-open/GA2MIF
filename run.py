import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import pickle as pk
import datetime
from parsers import args
import torch.nn as nn
import torch.optim as optim
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import time
from utils import AutomaticWeightedLoss, Focal_Loss, Poly1_Cross_Entropy, Poly1_Focal_Loss
from model import GNNModel
from sklearn.metrics import confusion_matrix, classification_report
from trainer import train_or_eval_graph_model, seed_everything
from dataloader import IEMOCAPDataset, MELDDataset
from torch.utils.data import DataLoader

save_path = 'save_model/best_model_{}/model_{}_{}{}{}_{}{}{}_{}_{}.pth'.format(
    args.Dataset, 
    args.base_model,
    args.base_nlayers, args.unimodal_nlayers, args.crossmodal_nlayers, 
    args.base_size, args.hidesize, args.list_mlp, 
    args.list_nheads,
    args.agg_type)
MELD_path = '/home/lijfrank/code/dataset/MELD_features/MELD_features_raw1.pkl'
IEMOCAP_path = '/home/lijfrank/code/dataset/IEMOCAP_features/IEMOCAP_features.pkl'

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12353'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

def cleanup():
    dist.destroy_process_group()

def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return DistributedSampler(idx[split:]), DistributedSampler(idx[:split])

def get_MELD_loaders(path, batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset(path)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset(path, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(path, batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset(path)

    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(path, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

def main(rank, world_size):
    print(f"Running main(**args) on rank {rank}.")
    setup(rank, world_size)

    today = datetime.datetime.now()
    name_ =args.modals+'_'+args.Dataset

    if args.ratio_speaker > 0:
        name_ = name_+'_speaker'
    if args.ratio_modal > 0:
        name_ = name_+'_modal'

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    cuda       = args.cuda
    n_epochs   = args.epochs
    batch_size = args.batch_size
    modals = args.modals
    feat2dim = {'IS10':1582,'3DCNN':512,'textCNN':100,'bert':768,'denseface':342,'MELD_text':600,'MELD_audio':300}
    D_audio = feat2dim['IS10'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = feat2dim['textCNN'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_text']

    if args.multi_modal:
        D_m = D_text
    else:
        if modals == 'a':
            D_m = D_audio
        elif modals == 'v':
            D_m = D_visual
        elif modals == 'l':
            D_m = D_text
        else:
            raise NotImplementedError
    n_speakers = 9 if args.Dataset=='MELD' else 2
    n_classes  = 7 if args.Dataset=='MELD' else 6 if args.Dataset=='IEMOCAP' else 1
    seed_everything()

    model = GNNModel(args, D_m_a=D_audio, D_m_v=D_visual, D_m=D_m, num_speakers=n_speakers, n_classes=n_classes)

    model = model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank,find_unused_parameters=True)

    if args.Dataset == 'MELD':
        loss_function = nn.CrossEntropyLoss()
    else:
        weights_iemocap = torch.FloatTensor([1.0/0.087179, 1.0/0.145836, 1.0/0.229786, 1.0/0.148392, 1.0/0.140051, 1.0/0.248756])
        loss_function = nn.CrossEntropyLoss(weight=weights_iemocap.cuda() if cuda else weights_iemocap)

    optimizer = optim.AdamW(model.parameters() , lr=args.lr, weight_decay=args.l2, amsgrad=True)

    if args.Dataset == 'MELD':
        train_loader, valid_loader, test_loader = get_MELD_loaders(path=MELD_path, valid=0.1,
                                                                    batch_size=batch_size,
                                                                    num_workers=0,
                                                                    pin_memory=True)
    elif args.Dataset == 'IEMOCAP':
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(path=IEMOCAP_path, valid=0.1,
                                                                      batch_size=batch_size,
                                                                      num_workers=0,
                                                                      pin_memory=True)
    else:
        print("There is no such dataset")

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []

    for e in range(n_epochs):
        if args.Dataset == 'MELD':
            trainset = MELDDataset(MELD_path)
        elif args.Dataset == 'IEMOCAP':
            trainset = IEMOCAPDataset(IEMOCAP_path)
        train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid=0.1)
        train_sampler.set_epoch(e)
        valid_sampler.set_epoch(e)

        start_time = time.time()
        train_loss, train_acc, _, _, train_fscore, _, _, _, _, _ , train_loss_a, train_acc_a, train_fscore_a, train_loss_v, train_acc_v, train_fscore_v, train_loss_l, train_acc_l, train_fscore_l = train_or_eval_graph_model(model, loss_function, train_loader, e, cuda, args.modals, optimizer, True, dataset=args.Dataset)
        valid_loss, valid_acc, _, _, valid_fscore, _, _, _, _, _, valid_loss_a, valid_acc_a, valid_fscore_a, valid_loss_v, valid_acc_v, valid_fscore_v, valid_loss_l, valid_acc_l, valid_fscore_l = train_or_eval_graph_model(model, loss_function, valid_loader, e, cuda, args.modals, dataset=args.Dataset)
        print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, tra_val time: {} sec'.\
            format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, round(time.time()-start_time, 2)))

        if rank == 0:
            test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _ , test_loss_a, test_acc_a, test_fscore_a, test_loss_v, test_acc_v, test_fscore_v, test_loss_l, test_acc_l, test_fscore_l = train_or_eval_graph_model(model, loss_function, test_loader, e, cuda, args.modals, dataset=args.Dataset)
            all_fscore.append(test_fscore)
            all_acc.append(test_acc)
  
            print('epoch: {}, test_loss: {}, test_acc: {}, test_fscore: {}, total time: {} sec, {}'.\
                    format(e+1, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2), time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
            print('epoch: {}, test_loss_a: {}, test_acc_a: {}, test_fscore_a: {}'.\
                    format(e+1, test_loss_a, test_acc_a, test_fscore_a))
            print('epoch: {}, test_loss_v: {}, test_acc_v: {}, test_fscore_v: {}'.\
                    format(e+1, test_loss_v, test_acc_v, test_fscore_v))
            print('epoch: {}, test_loss_l: {}, test_acc_l: {}, test_fscore_l: {}'.\
                    format(e+1, test_loss_l, test_acc_l, test_fscore_l))
            print('-'*150)

            if best_fscore == None or best_fscore < test_fscore: 
                best_fscore = test_fscore
                best_label, best_pred = test_label, test_pred

            if (e+1)%10 == 0:
                print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4, zero_division=0))
                print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))
                print('-'*150)
        dist.barrier()
        
        if args.tensorboard:
            writer.add_scalar('test: accuracy', test_acc, e)
            writer.add_scalar('test: fscore', test_fscore, e)
            writer.add_scalar('train: accuracy', train_acc, e)
            writer.add_scalar('train: fscore', train_fscore, e)
    
    if args.tensorboard:
        writer.close()
    if rank == 0:    
        print('Test performance..')
        print ('Acc: {}, F-Score: {}'.format(max(all_acc), max(all_fscore)))
        if not os.path.exists("record_{}_{}_{}.pk".format(today.year, today.month, today.day)):
            with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day),'wb') as f:
                pk.dump({}, f)
        with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day), 'rb') as f:
            record = pk.load(f)
        key_ = name_
        if record.get(key_, False):
            record[key_].append(max(all_fscore))
        else:
            record[key_] = [max(all_fscore)]
        if record.get(key_+'record', False):
            record[key_+'record'].append(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4, zero_division=0))
        else:
            record[key_+'record'] = [classification_report(best_label, best_pred, sample_weight=best_mask, digits=4, zero_division=0)]
        with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day),'wb') as f:
            pk.dump(record, f)

        print(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4, zero_division=0))
        print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))
        
    cleanup()

if __name__ == '__main__':
    print(args)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("not args.no_cuda:", not args.no_cuda)
    n_gpus = torch.cuda.device_count()
    print(f"Use {n_gpus} GPUs")
    run_demo(main, n_gpus)