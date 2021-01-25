#import torch
#info=torch.load('/home/litengpeng/CODE/instance-segmantation/mmdetection-master/results.pkl')
# #make co-labels
# import numpy as np
# import os
# import cv2
# cosal2015={}
# imgroot='/home/litengpeng/CODE/co-segmentation/irn-master/dataset/ECSSD/images-resize/'
# category=np.zeros(50, dtype='float32')
# category[1]=1
# os.chdir(imgroot)
# imname=os.listdir(os.getcwd())
# for j in imname:
#         if j[:-4] not in cosal2015.keys():
#              cosal2015[j[:-4]]=category
#     #index=index+1
# np.save('/home/litengpeng/CODE/co-segmentation/irn-master/dataset/ECSSD/ECSSD.npy', cosal2015)
# ############################
# CAT_LIST = ['aeroplane', 'apple', 'axe', 'babycrib', 'banana', 'baseball', 'bear', 'billiardtable', 'bird', 'boat',
#               'bowl',  'butterfly', 'camel', 'car',   'cat',    'chook',    'coffeecup', 'cow',      'deer',  'dog',
#               'frenchhorn', 'frog', 'goldenfish', 'guitar', 'hammer', 'helmet', 'horse', 'ladybird', 'lemon', 'lizard',
#               'mobulidae', 'monkey', 'motorbike', 'mouse', 'mushroom', 'penguin', 'pepper', 'piano', 'pig', 'pineapple',
#               'rabbit', 'sealion', 'snail', 'snake', 'sofa', 'starfish', 'tiger', 'train', 'turtle', 'viola']
#
# ######
# ######
#import json
#with open("/home/litengpeng/CODE/instance-segmantation/mmdetection-master/results.pkl.segm.json",'r') as load_f:
#     load_dict = json.load(load_f)
## ####
## ####
import numpy as np
import os
import cv2
#import torch
#pred=torch.load('/home/litengpeng/CODE/cosal/HRNet-MaskRCNN-Benchmark/work_dirs/faster_rcnn_hrnet_w18_1x/inference/coco_2017_val/predictions.pth')
#cosal2015={}
#imgroot='/home/litengpeng/CODE/cosal/dataset/MSRC/image/'
##imgroot='/home/litengpeng/CODE/co-segmentation/irn-master/dataset/co-saliency/icoseg/image-all/'
#os.chdir(imgroot)
#foldername=os.listdir(os.getcwd())
#for i in foldername:
#     #index=CAT_LIST.index(i)
#     folderpath=imgroot + i + '/'
#     category=np.zeros(50, dtype='float32')
#     category[1]=1
#     print(i)
#     os.chdir(folderpath)
#     imname=os.listdir(os.getcwd())
#     for j in imname:
#         if j[:-4] not in cosal2015.keys():
#              cosal2015[j[:-4]]=category
#     #index=index+1
#np.save('/home/litengpeng/CODE/co-segmentation/irn-master/dataset/co-saliency/msrc/msrc.npy', cosal2015)
##from pycocotools import coco
import numpy as np
import os
import cv2
#import torch
#pred=torch.load('/home/litengpeng/CODE/cosal/HRNet-MaskRCNN-Benchmark/work_dirs/faster_rcnn_hrnet_w18_1x/inference/coco_2017_val/predictions.pth')
cosal2015={}
#imgroot='/home/litengpeng/CODE/cosal/dataset/MSRC/image/'
imgroot='/data/ltp/CODEs/IRN/dataset/CoCA/image-256-all/'
os.chdir(imgroot)
imname=os.listdir(os.getcwd())
category=np.zeros(80, dtype='float32')
category[1]=1
for j in imname:
         if j[:-4] not in cosal2015.keys():
              cosal2015[j[:-4]]=category
     #index=index+1
np.save('/data/ltp/CODEs/IRN/dataset/CoCA/CoCA.npy', cosal2015)


#ann_path='/media/litengpeng2/litengpeng/data/coco/annotations/instances_train2017.json'
#imgroot='/home/litengpeng/CODE/COCO-dataset/train2017/'
#resroot='/home/litengpeng/CODE/COCO-dataset/train2017-psa/'
#coco_item=coco.COCO(ann_path)
#imgs_dict=coco_item.imgs
#anns_dict=coco_item.imgToAnns
#cat_dict=coco_item.cats
#cat_all=[]
#for j in cat_dict:
#    cat_all.append(j)
######
#coco_train={}
#for i in anns_dict:
#   img_dict=imgs_dict[i]
#   ann_dict=anns_dict[i]
#   category=np.zeros(80,dtype='float32')
#   for j in range(len(ann_dict)):
#       imname=img_dict['file_name']
#       id=ann_dict[j]['category_id']
#       index=cat_all.index(id)
#       category[index]=1
#   coco_train[imname[:-4]]=category
#np.save('/home/litengpeng/CODE/semantic-segmentation/psa-master/dataset/coco/coco_train.npy', coco_train)

######
# len=0
# for i in anns_dict:
#     img_dict=imgs_dict[i]
#     imname=img_dict['file_name']
#     imgpath=imgroot + imname
#     respath=resroot + imname
#     img=cv2.imread(imgpath, 1)
#     cv2.imwrite(respath, img)
#     len=len+1
