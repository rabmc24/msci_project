import torch
from torch import nn
import torch.nn.functional as F


def distance_corr(var_1,var_2,normedweight,power=1):
    """var_1: First variable to decorrelate (eg mass)
    var_2: Second variable to decorrelate (eg classifier output)
    normedweight: Per-example weight. Sum of weights should add up to N (where N is the number of examples)
    power: Exponent used in calculating the distance correlation
    va1_1, var_2 and normedweight should all be 1D torch tensors with the same number of entries
    Usage: Add to your loss function. total_loss = BCE_loss + lambda * distance_corr
    """
    xx = var_1.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))
    yy = var_1.repeat(len(var_1),1).view(len(var_1),len(var_1))
    amat = (xx-yy).abs()

    xx = var_2.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))
    yy = var_2.repeat(len(var_2),1).view(len(var_2),len(var_2))
    bmat = (xx-yy).abs()

    amatavg = torch.mean(amat*normedweight,dim=1)
    Amat=amat-amatavg.repeat(len(var_1),1).view(len(var_1),len(var_1))\
        -amatavg.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))\
        +torch.mean(amatavg*normedweight)

    bmatavg = torch.mean(bmat*normedweight,dim=1)
    Bmat=bmat-bmatavg.repeat(len(var_2),1).view(len(var_2),len(var_2))\
        -bmatavg.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))\
        +torch.mean(bmatavg*normedweight)

    ABavg = torch.mean(Amat*Bmat*normedweight,dim=1)
    AAavg = torch.mean(Amat*Amat*normedweight,dim=1)
    BBavg = torch.mean(Bmat*Bmat*normedweight,dim=1)

    if(power==1):
        dCorr=(torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight)))
    elif(power==2):
        dCorr=(torch.mean(ABavg*normedweight))**2/(torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))
    else:
        dCorr=((torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))))**power

    return dCorr

class BCEDecorrelatedLoss(nn.Module):
    def __init__(self,lam=0.1,weighted=True):
        super().__init__()

        self.lam = lam
        self.weighted = weighted
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        #self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self,outputs,labels,event,weights):
        if not self.weighted:
            weights = torch.ones((labels.shape[0],1)).to(outputs.device)

        if outputs.dim() == 2 and outputs.shape[1] == 1:
            outputs = outputs[:,0]
        if labels.dim() == 2 and labels.shape[1] == 1:
            labels = labels[:,0]

        bce_loss_value = self.bce_loss(outputs,labels) * weights

        disco_loss_value = distance_corr(
            outputs,
            event,
            weights * len(weights) / sum(weights)
        )

        return {
            'bce': bce_loss_value.mean(),
            'disco': disco_loss_value.mean(),
            'tot' : bce_loss_value.mean() + self.lam * disco_loss_value.mean(),
        }



###############################################################
############ NEW LOSS FUNCTION FOR MULTICLASSIFIER ############
###############################################################

class MulticlassDecorrelatedLoss(nn.Module):
    def __init__(self, lam=0.1, weighted=True):
        super().__init__()
        self.lam = lam
        self.weighted = weighted
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, outputs, labels, event, weights=None):
        # Convert labels to Long type
        labels = labels.long().squeeze()
        
        if not self.weighted or weights is None:
            weights = torch.ones((labels.shape[0],1)).to(outputs.device)
            
        ce_loss_value = self.ce_loss(outputs, labels) * weights.squeeze()
        
        probs = F.softmax(outputs, dim=1)  # Will work for 4 classes automatically
        
        normed_weights = weights.squeeze() * len(weights) / weights.sum()
        disco_loss = 0
        for i in range(outputs.shape[1]):  # Will loop over all 4 classes
            disco_loss += distance_corr(probs[:,i], event, normed_weights)
        disco_loss = disco_loss / outputs.shape[1]
        
        return {
            'ce': ce_loss_value.mean(),
            'disco': disco_loss,
            'tot': ce_loss_value.mean() + self.lam * disco_loss
        }


###########################
#### Redone FOCAL LOSS ####
###########################

class MulticlassFocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, weighted=False):
        """
        gamma: focusing parameter
        alpha: per-class weight factor (list or tensor of shape [num_classes]) or None
        weighted: whether to use provided per-example weights
        """
        super().__init__()
        self.gamma = gamma
        self.weighted = weighted
        
        # Handle class-specific alpha weights
        if alpha is None:
            self.alpha = None
        else:
            # Convert alpha list to tensor and (optionally) normalize
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
            if self.alpha.sum() > 0:
                self.alpha = self.alpha / self.alpha.sum()

    def forward(self, outputs, labels, event, weights=None):
        labels = labels.to(torch.long)
        
        # If labels have an extra singleton dimension, squeeze it.
        if labels.dim() == 2 and labels.shape[1] == 1:
            labels = labels.squeeze(1)

        # Use log_softmax for numerical stability
        log_softmax = F.log_softmax(outputs, dim=-1)
        log_pt = torch.gather(log_softmax, dim=1, index=labels.unsqueeze(1))

        # Convert log_pt to probabilities
        pt = torch.exp(log_pt).clamp(min=1e-10, max=1.0)

        # Calculate the focal weighting factor
        focal_weight = ((1 - pt) ** self.gamma).detach()

        # Apply class-specific alpha weights if provided
        if self.alpha is not None:
            self.alpha = self.alpha.to(outputs.device)
            alpha_t = self.alpha[labels].unsqueeze(1)
            focal_weight = alpha_t * focal_weight

        # Final focal loss calculation
        FL = -focal_weight * log_pt

        # Handle per-sample weights
        if not self.weighted or weights is None:
            weights = torch.ones(labels.shape[0], device=outputs.device)

        # Multiply by weights and average
        weighted_loss = FL.squeeze() * weights
        return weighted_loss.mean()















## FOCAL LOSS ##
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='none'):
        """
        gamma: focusing parameter
        alpha: weight factor per class (tensor) or None
        reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits with shape (N, C), targets: (N,)
        logpt = -F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.alpha is not None:
            # If alpha is given as a tensor of shape (C,)
            alpha_t = self.alpha.gather(0, targets.data.view(-1))
            loss = alpha_t * loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


    
class MulticlassDecorrelatedFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, lam=0.1, weighted=True):
        """
        gamma: focusing parameter for FocalLoss
        alpha: class weighting for FocalLoss (tensor of shape [num_classes] or None)
        lam: lambda to weight the decorrelation loss
        weighted: whether to use provided per-example weights
        """
        super().__init__()
        self.lam = lam
        self.weighted = weighted
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha, reduction='none')

    def forward(self, outputs, labels, event, weights=None):
        # Convert labels to long type and squeeze extra dims
        labels = labels.long().squeeze()
        if not self.weighted or weights is None:
            weights = torch.ones((labels.shape[0], 1), device=outputs.device)
        
        focal_loss_value = self.focal_loss(outputs, labels) * weights.squeeze()

        probs = F.softmax(outputs, dim=1)
        normed_weights = weights.squeeze() * len(weights) / weights.sum()
        disco_loss = 0
        for i in range(outputs.shape[1]):
            disco_loss += distance_corr(probs[:, i], event, normed_weights)
        disco_loss = disco_loss / outputs.shape[1]
        
        return {
            'focal': focal_loss_value.mean(),
            'disco': disco_loss,
            'tot': focal_loss_value.mean() + self.lam * disco_loss
        }