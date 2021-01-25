import argparse
import os
import cv2

from misc import pyutils
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument("--cosal_root", default='./datasets/CoCA/image-256/', type=str,
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")
    # Dataset
    parser.add_argument("--train_list", default="./datasets/CoCA/CoCA256_demo.txt", type=str)
    parser.add_argument("--infer_list", default="./datasets/CoCA/CoCA256_demo.txt", type=str,help="supervised model")
    parser.add_argument("--chainer_eval_set", default="train", type=str)

    # Class Activation Map
    parser.add_argument("--cam_network", default="net.resnet50_cam", type=str)
    parser.add_argument("--cam_crop_size", default=256, type=int)
    parser.add_argument("--cam_batch_size", default=16, type=int)
    parser.add_argument("--cam_num_epoches", default=5, type=int)
    parser.add_argument("--cam_learning_rate", default=0.1, type=float)
    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--cam_eval_thres", default=0.15, type=float)
    parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
                        help="Multi-scale inferences")

    # Mining Inter-pixel Relations
    parser.add_argument("--conf_fg_thres", default=0.30, type=float)
    parser.add_argument("--conf_bg_thres", default=0.05, type=float)

    # Inter-pixel Relation Network (IRNet)
    parser.add_argument("--irn_network", default="net.resnet50_irn", type=str)
    parser.add_argument("--irn_crop_size", default=256, type=int)
    parser.add_argument("--irn_batch_size", default=12, type=int)
    parser.add_argument("--irn_num_epoches", default=4, type=int)
    parser.add_argument("--irn_learning_rate", default=0.1, type=float)
    parser.add_argument("--irn_weight_decay", default=1e-4, type=float)

    # Random Walk Params
    parser.add_argument("--beta", default=10)
    parser.add_argument("--exp_times", default=8,
                        help="Hyper-parameter that controls the number of random walk iterations,"
                             "The random walk is performed 2^{exp_times}.")
    parser.add_argument("--ins_seg_bg_thres", default=0.25)
    parser.add_argument("--sem_seg_bg_thres", default=0.25)
    
    ## Input Path
    parser.add_argument("--mask_root", default="./saliency-maps/CoCA-256-demo/", type=str)

    # Output Path
    parser.add_argument("--log_name", default="sample_train_eval", type=str)
    parser.add_argument("--cam_weights_name", default="sess/res50_cam.pth", type=str)
    parser.add_argument("--irn_weights_name", default="sess/res50_irn.pth", type=str)
    parser.add_argument("--cam_out_dir", default="./out/CoCA/cam", type=str)
    parser.add_argument("--ir_label_out_dir", default="./out/CoCA/ir_label", type=str)
    parser.add_argument("--sem_seg_out_dir", default="./out/CoCA/sem_seg", type=str)
    parser.add_argument("--ins_seg_out_dir", default="./out/CoCA/ins_seg", type=str)

    # Step
    parser.add_argument("--make_cam_pass", default=True)      ## True
    parser.add_argument("--cam_to_ir_label_pass", default=True)  ## True
    parser.add_argument("--train_irn_pass", default=True)    ## True
    parser.add_argument("--make_ins_seg_pass", default=True)  ## True
    parser.add_argument("--make_sem_seg_pass", default=True)  ## True

    args = parser.parse_args()

    os.makedirs("sess", exist_ok=True)
    os.makedirs(args.cam_out_dir, exist_ok=True)
    os.makedirs(args.ir_label_out_dir, exist_ok=True)
    os.makedirs(args.sem_seg_out_dir, exist_ok=True)
    os.makedirs(args.ins_seg_out_dir, exist_ok=True)

    pyutils.Logger(args.log_name + '.log')
    print(vars(args))

    if args.make_cam_pass is True:
        import step.make_cam

        timer = pyutils.Timer('step.make_cam:')
        step.make_cam.run(args)

    if args.cam_to_ir_label_pass is True:
        import step.cam_to_ir_label

        timer = pyutils.Timer('step.cam_to_ir_label:')
        step.cam_to_ir_label.run(args)

    if args.train_irn_pass is True:
        import step.train_irn

        timer = pyutils.Timer('step.train_irn:')
        step.train_irn.run(args)

    if args.make_ins_seg_pass is True:
        import step.make_ins_seg_labels

        timer = pyutils.Timer('step.make_ins_seg_labels:')
        step.make_ins_seg_labels.run(args)

    if args.make_sem_seg_pass is True:
        import step.make_sem_seg_labels

        timer = pyutils.Timer('step.make_sem_seg_labels:')
        step.make_sem_seg_labels.run(args)

    ## CO-EGNET results
    res_root1='./out/CoCA/sem_seg/'
    res_root2='./saliency-maps/EGNET-salmap/'
    res_root3='./out/CoCA/results_co_egnet/'
    ##
    foldername=os.listdir(res_root2)
    for i in range(len(foldername)):
      res_folder_root2=res_root2 + foldername[i] + '/'
      res_folder_root3=res_root3 + foldername[i] + '/'
      if not os.path.exists(res_folder_root3):
         os.makedirs(res_folder_root3)
      imnames=os.listdir(res_folder_root2)
      for j in range(len(imnames)):
          respath1=res_root1 + imnames[j]
          respath2=res_folder_root2 + imnames[j]
          respath3=res_folder_root3 + imnames[j]
          res1=cv2.imread(respath1, 0)
          res2=cv2.imread(respath2, 0)/255
          [w,h]=res2.shape
          res1=cv2.resize(res1, (h, w))
          res3=(res1*res2)*255
          cv2.imwrite(respath3, res3)
    print('The entire process is ending')

