import torch
import torch.nn as nn
import torch.nn.functional as F

class Poly1_Cross_Entropy(nn.Module):
    def __init__(self, weight, num_classes=5, epsilon=1.0, size_average=True):
        super(Poly1_Cross_Entropy, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.size_average = size_average
        self.ce_loss_func = nn.CrossEntropyLoss(weight)
    def forward(self, preds, labels):
        poly1 = torch.sum(F.one_hot(labels, self.num_classes).float() * F.softmax(preds,dim=-1), dim=-1)
        ce_loss = self.ce_loss_func(preds, labels)
        poly1_ce_loss = ce_loss + self.epsilon * (1 - poly1)
        if self.size_average:
            poly1_ce_loss = poly1_ce_loss.mean()
        else:
            poly1_ce_loss = poly1_ce_loss.sum()
        return poly1_ce_loss

class Poly1_Focal_Loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=5, epsilon=1.0, size_average=True):
        super(Poly1_Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.size_average = size_average
        self.focal_loss_func = Focal_Loss(self.alpha, self.gamma, self.num_classes, self.size_average)
    def forward(self, preds, labels):
        focal_loss = self.focal_loss_func(preds, labels)
        p = torch.sigmoid(preds)
        labels = F.one_hot(labels, self.num_classes)
        poly1 = labels * p + (1 - labels) * (1 - p)
        poly1_focal_loss = focal_loss + torch.mean(self.epsilon * torch.pow(1 - poly1, 2 + 1), dim=-1)
        if self.size_average:
            poly1_focal_loss = poly1_focal_loss.mean()
        else:
            poly1_focal_loss = poly1_focal_loss.sum()
        return poly1_focal_loss

class Focal_Loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=5, size_average=True):

        super(Focal_Loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1 
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha)

        self.gamma = gamma

    def forward(self, preds, labels):

        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        
        preds_logsoft = F.log_softmax(preds, dim=1) 

        preds_softmax = torch.exp(preds_logsoft)    
   
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1)) 
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))

        self.alpha = self.alpha.gather(0,labels.view(-1))

        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft) 

        loss = torch.mul(self.alpha, loss.t())
        
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
       
        return loss

class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):

        mask_ = mask.view(-1, 1)
        if type(self.weight) == type(None):
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            loss = self.loss(pred * mask_, target) \
                   / torch.sum(self.weight[target] * mask_.squeeze())
        return loss


class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target, mask):

        loss = self.loss(pred * mask, target) / torch.sum(mask)
        return loss


class UnMaskedWeightedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(UnMaskedWeightedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target):

        if type(self.weight) == type(None):
            loss = self.loss(pred, target)
        else:
            loss = self.loss(pred, target) \
                   / torch.sum(self.weight[target])
        return loss

class AutomaticWeightedLoss(nn.Module):

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

if __name__ == '__main__':
    awl = AutomaticWeightedLoss(2)
    print(awl.parameters())