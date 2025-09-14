from torch.autograd import Variable
import torch.nn as nn
import torch
import torch.nn.functional as F

class CrossEntropy2d_ignore(nn.Module):
    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d_ignore, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        
        print('predict.shape' + str(predict.shape))
        print('target.shape' + str(target.shape))
        
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        
        return loss

class RS3MambaLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.loss = CrossEntropy2d_ignore()

    def forward(self, pred, label, weights=None):
        """
        This function returns cross entropy loss for semantic segmentation
        """
        # out shape b x c x h x w -> b x c x h x w
        # label shape h x w x 1 x b  -> b x 1 x h x w
        label = Variable(label.long()).cuda()
        return self.loss(pred, label)


# def loss_calc(pred, label, weights):
#     """
#     This function returns cross entropy loss for semantic segmentation
#     """
#     # out shape batch_size x channels x h x w -> batch_size x channels x h x w
#     # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
#     label = Variable(label.long()).cuda()
#     criterion = CrossEntropy2d_ignore().cuda()
#
#     return criterion(pred, label, weights)




