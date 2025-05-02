# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import pdb
import torch.nn as nn
import torch.nn.functional as F

from nets.ap_loss import APLoss


class PixelAPLoss (nn.Module):
    """ Computes the pixel-wise AP loss:
        Given two images and ground-truth optical flow, computes the AP per pixel.
        
        feat1:  (B, C, H, W)   pixel-wise features extracted from img1
        feat2:  (B, C, H, W)   pixel-wise features extracted from img2
        aflow:  (B, 2, H, W)   absolute flow: aflow[...,y1,x1] = x2,y2
    """
    def __init__(self, sampler, nq=20):
        nn.Module.__init__(self)
        self.aploss = APLoss(nq, min=0, max=1, euc=False)
        self.name = 'pixAP'
        self.sampler = sampler

    def loss_from_ap(self, ap, rel):
        return 1 - ap

    def forward(self, descriptors, aflow, **kw):
        # subsample things
        scores, gt, msk, qconf, all_scores, _, _ = self.sampler(descriptors, kw.get('reliability'), aflow)
        
        # compute pixel-wise AP
        n = qconf.numel()
        if n == 0: return 0
        gt = gt.view(n,-1)

        all_loss = []
        for s in all_scores:
            s = s.view(n,-1)

            ap = self.aploss(s, gt).view(msk.shape)

            pixel_loss = self.loss_from_ap(ap, qconf)
        
            loss = pixel_loss[msk].mean()
            all_loss.append(loss)
            
        return loss.mean()



class ReliabilityLoss (PixelAPLoss):
    """ same than PixelAPLoss, but also train a pixel-wise confidence
        that this pixel is going to have a good AP.
    """
    def __init__(self, sampler, base=0.5, **kw):
        PixelAPLoss.__init__(self, sampler, **kw)
        assert 0 <= base < 1
        self.base = base
        self.name = 'reliability'

    def loss_from_ap(self, ap, rel):
        return 1 - ap*rel - (1-rel)*self.base




class MetricLoss (nn.Module):
    """ Computes the pixel-wise AP loss:
        Given two images and ground-truth optical flow, computes the AP per pixel.
        
        feat1:  (B, C, H, W)   pixel-wise features extracted from img1
        feat2:  (B, C, H, W)   pixel-wise features extracted from img2
        aflow:  (B, 2, H, W)   absolute flow: aflow[...,y1,x1] = x2,y2
    """
    def __init__(self, sampler, loss_name):
        nn.Module.__init__(self)
        # self.aploss = APLoss(nq, min=0, max=1, euc=False)
        self.name = 'Metric Loss'
        self.sampler = sampler
        self.loss_fn = all_losses[loss_name]

    def forward(self, descriptors, aflow, **kw):
        # subsample things
        _,_,_,_,_,feat1, feat2  = self.sampler(descriptors, kw.get('reliability'), aflow)
        # print(feat1.shape, feat2.shape)

        # print(torch.any(torch.isnan(feat1)))
        # print(torch.any(torch.isnan(feat2)))


        labels = torch.arange(len(feat1))
        all_labels = torch.cat([labels, labels])
        all_feat = torch.cat([feat1, feat2])

        # print(all_labels.shape, all_feat.shape)
        loss_value =  self.loss_fn(all_feat, all_labels)
        # print(loss_value)
       
        return loss_value




