import os
import numpy as np, random
import torch
from sklearn.metrics import f1_score, accuracy_score
from parsers import args
from utils import AutomaticWeightedLoss
import torch.nn.functional as F


seed = 2022
def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train_or_eval_graph_model(model, loss_function, dataloader, epoch, cuda, modals, optimizer=None, train=False, dataset='IEMOCAP'):
    losses, preds, labels = [], [], []
    losses_a, preds_a = [], []
    losses_v, preds_v = [], []
    losses_l, preds_l = [], []
    scores, vids = [], []

    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []

    if cuda:
        ei, et, en = ei.cuda(), et.cuda(), en.cuda()

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything()
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        
        if args.multi_modal:
            textf = textf
        else:
            if modals == 'a':
                textf = acouf
            elif modals == 'v':
                textf = visuf
            elif modals == 'l':
                textf = textf
            else:
                raise NotImplementedError
        
        lengths0 = []

        max_seq_length = len(umask[0])

        for j, umask_ in enumerate(umask):
            lengths0.append((umask[j] == 1).nonzero()[-1][0] + 1)
        lengths = torch.stack(lengths0)

        if len(modals)==3:
            prob_a, prob_v, prob_l, prob = model(textf, qmask, umask, lengths, max_seq_length, acouf, visuf)
        if len(modals)==2:
            prob_0, prob_1, prob = model(textf, qmask, umask, lengths, max_seq_length, acouf, visuf)
        if len(modals)==1:
            prob_0, prob = model(textf, qmask, umask, lengths, max_seq_length, acouf, visuf)

        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])

        loss = loss_function(prob, label)
        if len(modals)==3:
            loss_a = loss_function(prob_a, label)
            loss_v = loss_function(prob_v, label)
            loss_l = loss_function(prob_l, label)
        if len(modals)==2:
            loss_0 = loss_function(prob_0, label)
            loss_1 = loss_function(prob_1, label)
        if len(modals)==1:
            loss_0 = loss_function(prob_0, label)

        preds.append(torch.argmax(prob, 1).cpu().numpy())
        if len(modals)==3:
            preds_a.append(torch.argmax(prob_a, 1).cpu().numpy())
            preds_v.append(torch.argmax(prob_v, 1).cpu().numpy())
            preds_l.append(torch.argmax(prob_l, 1).cpu().numpy())
        if len(modals)==2:
            preds_a.append(torch.argmax(prob_0, 1).cpu().numpy())
            preds_v.append(torch.argmax(prob_1, 1).cpu().numpy())
            preds_l.append(torch.argmax(prob_0, 1).cpu().numpy())
        if len(modals)==1:
            preds_a.append(torch.argmax(prob_0, 1).cpu().numpy())
            preds_v.append(torch.argmax(prob_0, 1).cpu().numpy())
            preds_l.append(torch.argmax(prob_0, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())
        if len(modals)==3:
            losses_a.append(loss_a.item())
            losses_v.append(loss_v.item())
            losses_l.append(loss_l.item())
        if len(modals)==2:
            losses_a.append(loss_0.item())
            losses_v.append(loss_1.item())
            losses_l.append(loss_0.item())
        if len(modals)==1:
            losses_a.append(loss_0.item())
            losses_v.append(loss_0.item())
            losses_l.append(loss_0.item())

        if train:
            loss.backward()
            optimizer.step()

    if preds!=[]:
        preds  = np.concatenate(preds)
        preds_a  = np.concatenate(preds_a)
        preds_v  = np.concatenate(preds_v)
        preds_l  = np.concatenate(preds_l)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    vids += data[-1]
    ei = ei.data.cpu().numpy()
    et = et.data.cpu().numpy()
    en = en.data.cpu().numpy()
    el = np.array(el)
    labels = np.array(labels)
    preds = np.array(preds)
    preds_a = np.array(preds_a)
    preds_v = np.array(preds_v)
    preds_l = np.array(preds_l)
    vids = np.array(vids)

    avg_loss = round(np.sum(losses)/len(losses), 4)
    avg_loss_a = round(np.sum(losses_a)/len(losses_a), 4)
    avg_loss_v = round(np.sum(losses_v)/len(losses_v), 4)
    avg_loss_l = round(np.sum(losses_l)/len(losses_l), 4)
    avg_accuracy = round(accuracy_score(labels, preds)*100, 2)
    avg_accuracy_a = round(accuracy_score(labels, preds_a)*100, 2)
    avg_accuracy_v = round(accuracy_score(labels, preds_v)*100, 2)
    avg_accuracy_l = round(accuracy_score(labels, preds_l)*100, 2)
    avg_fscore = round(f1_score(labels,preds, average='weighted',zero_division=0)*100, 2)
    avg_fscore_a = round(f1_score(labels,preds_a, average='weighted',zero_division=0)*100, 2) 
    avg_fscore_v = round(f1_score(labels,preds_v, average='weighted',zero_division=0)*100, 2) 
    avg_fscore_l = round(f1_score(labels,preds_l, average='weighted',zero_division=0)*100, 2) 

    return avg_loss, avg_accuracy, labels, preds, avg_fscore, vids, ei, et, en, el, avg_loss_a, avg_accuracy_a, avg_fscore_a, avg_loss_v, avg_accuracy_v, avg_fscore_v,avg_loss_l, avg_accuracy_l, avg_fscore_l