import os.path as osp
import torch
from torch import nn
from torchvision.models import resnet

from nets.soft_detect import DKD
from nets.padder import InputPadder
from nets.blocks import *
import time
from torchvision.transforms import ToTensor
from skimage.transform import AffineTransform, warp, ProjectiveTransform, rotate



ALIKED_CFGS = {'aliked-t16':{'c1': 8,
                             'c2': 16,
                             'c3': 32,
                             'c4': 64,
                             'dim': 64,
                             'K': 3,
                             'M': 16},
               'aliked-n16':{'c1': 16,
                             'c2': 32,
                             'c3': 64,
                             'c4': 128,
                             'dim': 128,
                             'K': 3,
                             'M': 16},
               'aliked-n16rot':{'c1': 16,
                             'c2': 32,
                             'c3': 64,
                             'c4': 128,
                             'dim': 128,
                             'K': 3,
                             'M': 16},
               'aliked-n32':{'c1': 16,
                             'c2': 32,
                             'c3': 64,
                             'c4': 128,
                             'dim': 128,
                             'K': 3,
                             'M': 32},

                'aliked-n32_long_desc':{'c1': 8,
                             'c2': 16,
                             'c3': 32,
                             'c4': 2*128,
                             'dim': 2*128,
                             'K': 3,
                             'M': 32},
                             }

            
class ALIKED(nn.Module):

    def __init__(
            self,
            model_name: str = 'aliked-n32_long_desc',
            device: str = 'cuda',
            top_k: int = -1, # -1 for threshold based mode, >0 for top K mode.
            scores_th: float = 0.2,
            n_limit: int = 5000, # Maximum number of keypoints to be detected
            load_pretrained: bool = True,
            ):
        super().__init__()
        
        # get configurations
        c1, c2, c3, c4, dim, K, M = [v for _,v in ALIKED_CFGS[model_name].items()]
        conv_types = ['conv','conv','dcn','dcn']
        conv2D = False
        mask = False
        self.device = device
        
        # build model
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.norm = nn.BatchNorm2d
        self.gate = nn.SELU(inplace=True)
        self.block1 = ConvBlock(3, c1, self.gate, self.norm, conv_type=conv_types[0])
        self.block2 = ResBlock(c1,c2,1,nn.Conv2d(c1, c2, 1),
                                gate=self.gate,
                                norm_layer=self.norm,
                                conv_type=conv_types[1])
        self.block3 = ResBlock(c2,c3,1,nn.Conv2d(c2, c3, 1),
                            gate=self.gate,
                            norm_layer=self.norm,
                            conv_type=conv_types[2],
                            mask = mask)
        self.block4 = ResBlock(c3,c4,1,nn.Conv2d(c3, c4, 1),
                            gate=self.gate,
                            norm_layer=self.norm,
                            conv_type=conv_types[3],
                            mask = mask)
        self.conv1 = resnet.conv1x1(c1, dim // 4)
        self.conv2 = resnet.conv1x1(c2, dim // 4)
        self.conv3 = resnet.conv1x1(c3, dim // 4)
        self.conv4 = resnet.conv1x1(dim, dim // 4)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.score_head = nn.Sequential(resnet.conv1x1(dim, 8), self.gate, resnet.conv3x3(8, 4), self.gate,
                                                resnet.conv3x3(4, 4), self.gate, resnet.conv3x3(4, 1))
        self.desc_head = SDDH(dim, K, M, gate=self.gate, conv2D=conv2D, mask=mask)
        self.dkd = DKD(radius=2, top_k=top_k, scores_th=scores_th, n_limit=n_limit)
        # self.reliability = ConvBlock(dim, 2, self.gate, self.norm, conv_type=conv_types[0])
        # load pretrained
        if load_pretrained:
            pretrained_path = osp.join(osp.split(__file__)[0], f'../models/{model_name}.pth')
            pretrained_path = osp.abspath(pretrained_path)
            if osp.exists(pretrained_path):
                print(f'loading {pretrained_path}')
                state_dict = torch.load(pretrained_path, 'cpu')
                self.load_state_dict(state_dict, strict=True)
                self.to(device)
                self.eval()
            else:
                raise FileNotFoundError(f'cannot find pretrained model: {pretrained_path}')   

    def softmax(self, ux):
        if ux.shape[1] == 1:
            x = F.softplus(ux)
            return x / (1 + x)  # for sure in [0,1], much less plateaus than softmax
        elif ux.shape[1] == 2:
            return F.softmax(ux, dim=1)[:,1:2]


    def normalize(self, x, ureliability, urepeatability):
        return dict(descriptors = F.normalize(x, p=2, dim=1),
                    repeatability = self.softmax( urepeatability ),
                    reliability = None)
     
    
    def extract_dense_map(self, image):
        # Pads images such that dimensions are divisible by
        div_by = 2**5
        padder = InputPadder(image.shape[-2],image.shape[-1], div_by)
        image = padder.pad(image)

        # ================================== feature encoder
        x1 = self.block1(image)  # B x c1 x H x W
        x2 = self.pool2(x1)
        x2 = self.block2(x2)  # B x c2 x H/2 x W/2
        x3 = self.pool4(x2)
        x3 = self.block3(x3)  # B x c3 x H/8 x W/8
        x4 = self.pool4(x3)
        x4 = self.block4(x4)  # B x dim x H/32 x W/32
        # ================================== feature aggregation
        x1 = self.gate(self.conv1(x1))  # B x dim//4 x H x W
        x2 = self.gate(self.conv2(x2))  # B x dim//4 x H//2 x W//2
        x3 = self.gate(self.conv3(x3))  # B x dim//4 x H//8 x W//8
        x4 = self.gate(self.conv4(x4))  # B x dim//4 x H//32 x W//32
        x2_up = self.upsample2(x2)  # B x dim//4 x H x W
        x3_up = self.upsample8(x3)  # B x dim//4 x H x W
        x4_up = self.upsample32(x4)  # B x dim//4 x H x W
        x1234 = torch.cat([x1, x2_up, x3_up, x4_up], dim=1)
        # ================================== score head
        score_map = torch.sigmoid(self.score_head(x1234))
        feature_map = torch.nn.functional.normalize(x1234, p=2, dim=1)
        # reliability = torch.sigmoid(self.reliability(x1234))
        # Unpads images
        feature_map = padder.unpad(feature_map)
        score_map = padder.unpad(score_map)
        # reliability = padder.unpad(reliability)

        return feature_map, score_map

    def forward(self, imgs, **kw):
        res = [self.forward_one(img) for img in imgs]
        # merge all dictionaries into one
        res = {k:[r[k] for r in res if k in r] for k in {k for r in res for k in r}}
        return dict(res, imgs=imgs, **kw)



    def forward_one(self, image):
        torch.cuda.synchronize()
        # t0 = time.time() 
        feature_map, score_map = self.extract_dense_map(image)
        keypoints, kptscores, scoredispersitys = self.dkd(score_map)
        descriptors, offsets = self.desc_head(feature_map, keypoints)
        torch.cuda.synchronize()
        # t1 = time.time()        

        
        return self.normalize(feature_map, score_map, None)
        # return {'keypoints': keypoints,  # B N 2
        #     'descriptors': descriptors,  # B N D
        #     'reliability': reliability,
        #     'scores': kptscores,  # B N
        #     'score_dispersity': scoredispersitys,
        #     'score_map': score_map,  # Bx1xHxW
        #     'time': t1-t0,

        # }
    
    # def run(self, img_rgb):
    #     img_tensor = ToTensor()(img_rgb)
    #     img_tensor = img_tensor.to(self.device).unsqueeze_(0)
        
        
    #     with torch.no_grad():
    #         pred = self.forward(img_tensor)
            
    #     kpts = pred['keypoints'][0]
    #     _, _, h, w = img_tensor.shape
    #     wh = torch.tensor([w - 1, h - 1],device=kpts.device)
    #     kpts = wh*(kpts+1)/2
    #     return {'keypoints': kpts.cpu().numpy(),  # N 2
    #         'descriptors': pred['descriptors'][0].cpu().numpy(),  # N D
    #         'reliability': pred['reliability'][0].cpu().numpy(),
    #         'scores': pred['scores'][0].cpu().numpy(),  # B N D
    #         'score_map': pred['score_map'][0,0].cpu().numpy(),  # Bx1xHxW
    #         'time': pred['time'],
    #     }
