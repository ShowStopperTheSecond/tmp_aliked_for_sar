# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import pdb
import torch.nn as nn
import torch.nn.functional as F

# from nets.ap_loss import APLoss
from pytorch_metric_learning import losses



all_losses = {
    "ConstrastiveLoss": losses.ContrastiveLoss(),
    "PNPLoss_Ds_2022" :losses.PNPLoss(variant='Ds'),
    "PNPLoss_Dq_2022" :losses.PNPLoss(variant='Dq'),
    "PNPLoss_Iu_2022" :losses.PNPLoss(variant='Iu'),
    "PNPLoss_Ib_2022" :losses.PNPLoss(variant='Ib'),
    "PNPLoss_O_2022" :losses.PNPLoss(variant='O'),
    "FastAPLoss_2019": losses.FastAPLoss(),
    "NTXentLoss": losses.NTXentLoss(),
    "InstanceLoss_2020": losses.InstanceLoss(),
    "MultiSimilarityLoss_2019": losses.MultiSimilarityLoss(),
    "SignalToNoiseRatioContrastiveLoss_2019": losses.SignalToNoiseRatioContrastiveLoss(),
    "AngularLoss_2017": losses.AngularLoss(),
    "CircleLoss_2020": losses.CircleLoss(),
    "GeneralizedLiftedStructureLoss_2017": losses.GeneralizedLiftedStructureLoss(),
    "IntraPairVarianceLoss_2019": losses.IntraPairVarianceLoss(),
    "LiftedStructureLoss_2016": losses.LiftedStructureLoss(),
    "MarginLoss_2017": losses.MarginLoss(),
         }





loss_fn = all_losses["FastAPLoss_2019"].to('cuda')



class MetricLoss (nn.Module):
    """ Computes the pixel-wise AP loss:
        Given two images and ground-truth optical flow, computes the AP per pixel.
        
        feat1:  (B, C, H, W)   pixel-wise features extracted from img1
        feat2:  (B, C, H, W)   pixel-wise features extracted from img2
        aflow:  (B, 2, H, W)   absolute flow: aflow[...,y1,x1] = x2,y2
    """
    def __init__(self, sampler, nq=20):
        nn.Module.__init__(self)
        # self.aploss = APLoss(nq, min=0, max=1, euc=False)
        self.name = 'Metric Loss'
        self.sampler = sampler

    def forward(self, descriptors, aflow, **kw):
        # subsample things
        feat1, feat2  = self.sampler(descriptors, kw.get('reliability'), aflow)

        labels = torch.arange(len(feat1))
        all_labels = torch.cat([labels, labels])
        all_feat = torch.cat([feat1, feat2])


        loss_value =  loss_fn(all_feat, all_labels)
       
        return loss_value



# all_desc1 = torch.cat([desc1a,desc1b,desc1c])
# all_desc2 = torch.cat([desc2a,desc2b,desc2c])
# labels = torch.arange(len(all_desc1))

# all_desc = torch.cat([all_desc1, all_desc2])

# all_labels = torch.cat([labels, labels])

# loss_value =  loss_fn(all_desc, all_labels)
# loss_value.backward()