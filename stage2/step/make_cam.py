import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os

import voc12.dataloader
from misc import torchutils, imutils

cudnn.enabled = True
from PIL import Image
                   
def my_work(args):
     
     maskroot=args.mask_root
     imname=os.listdir(maskroot)
     with torch.no_grad():
            for img_name in imname:
                size=torch.tensor([256,256])
                strided_size = imutils.get_strided_size(size, 4)
                strided_up_size = imutils.get_strided_up_size(size, 16)
                valid_cat=torch.zeros(80)
                maskpath=maskroot+img_name
                mask = np.asarray(Image.open(maskpath))
                mask=torch.from_numpy(np.array(mask)).float()
                mask2=torch.unsqueeze(mask,0)
                mask2=torch.unsqueeze(mask2,0)
                mask2 = mask2.cuda()
                strided_cams = F.upsample(mask2, strided_size,  mode='bilinear')
                highres_cams = F.upsample(mask2, strided_up_size,  mode='bilinear')
                strided_cams = strided_cams.squeeze(0)
                highres_cams = highres_cams.squeeze(0)
                print(img_name)
                # save cam
                np.save(os.path.join(args.cam_out_dir, img_name[:-4] + '.npy'),
                        {"keys": valid_cat, "cam": strided_cams.cpu(), "high_res": highres_cams.cpu().numpy()})
#
#                if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
#                    print("%d " % ((5 * iter + 1) // (len(databin) // 20)), end='')

def run(args):
    print('CAMs starting')
    my_work(args)

    torch.cuda.empty_cache()