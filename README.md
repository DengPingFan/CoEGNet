# CoEGNet
Code for "Re-thinking Co-Salient Object Detection" IEEE TPMAI2021
Please refer to our homepage for more details: http://dpfan.net/CoSOD3K/
## Pipeline
![pipeline](https://github.com/DengPingFan/CoEGNet/blob/master/stage2/figure/pipeline.png)
## Stage 1
### 1.1 Environments (Caffe && Matlab)
The caffe package is borrowed from https://github.com/BVLC/caffe
### 1.2 Pre-trained models in Caffe:
* VGG16 model on ImageNet: ```models/deploy_vgg16CAM.prototxt``` weights:[http://cnnlocalization.csail.mit.edu/demoCAM/models/vgg16CAM_train_iter_90000.caffemodel]
### 1.3 test
* simple run ./stage1/demo.m, then we can obtain the initial cosal activations
## Stage 2
### 2.1 Prerequisite
* Python 3.7, PyTorch 1.1.0
### 2.2 test
* python ./stage2/run_sample.py (Thanks for Ahn et al.'s implement, our postprocessing is inspired by [IRNET](https://github.com/jiwoon-ahn/irn))

## The results of CoEGNET 
Baidu Cloud: https://pan.baidu.com/s/19hIlViLbby-a7vQw17ZTVw Fetchcode: f4p3

Google Cloud: https://drive.google.com/file/d/1AK9UNR5mLHQakOTkRawcKSDg8YwIu-xJ/view?usp=sharing


## Citation
If you use this code, please cite our paper:
```
@article{deng2021re,
  title={Re-thinking co-salient object detection},
  author={Deng-Ping, Fan and Tengpeng, Li and Zheng, Lin and Ge-Peng, Ji and Dingwen, Zhang and Ming-Ming, Cheng and Huazhu, Fu and Jianbing, Shen},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021}
}

```


