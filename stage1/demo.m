clear
addpath(genpath('.'));
addpath('./caffe/matlab/');
caffe.set_mode_gpu();
caffe.set_device(1);
for q=3:m
    folderpath=strcat(folder,foldername(q).name,'/');
    resfolder=strcat(result,foldername(q).name,'/');
    mkdir(resfolder);
    imgname=dir([folderpath,'*.jpg']);
    imgnum=size(imgname,1);
    IDX=[];
    feat_all={};
    feat_all2={};
    pool_all={};
    for qx=1:imgnum
        %img = imresize(img, [256 256]);
        net_weights = ['models/vgg16CAM_train_iter_90000.caffemodel'];
        net_model = ['models/deploy_vgg16CAM.prototxt'];
        net = caffe.Net(net_model, net_weights, 'test');
        imgpath2=strcat(folderpath,imgname(qx).name);
        img=imread(imgpath2);
        [aa,bb,cc]=size(img);
        if cc==1
            img=repmat(img,[1,1,3]);
        end
        scores = net.forward({prepare_image(img)});
        featmap=net.blobs('CAM_conv').get_data();
        % cam_pool=net.blobs('CAM_pool').get_data();
        [w5,h5,c5,n5]=size(featmap);
        featmean=mean(featmap,4);
        featmap1=reshape(featmean,[w5*h5,c5]);
        feat_all{qx,1}=featmap1;
        feat_all2{qx,1}=featmap;
        
    end
    feat_sum=cell2mat(feat_all);
    [coeff,~,latent] = pca(feat_sum);
    trans_matrix=coeff(:,1);
    for qq=1:imgnum
        imgpath2=strcat(folderpath,imgname(qq).name);
        img=imread(imgpath2);
        [aa,bb,cc]=size(img);
        if cc==1
            img=repmat(img,[1,1,3]);
        end
        activation_lastconv=feat_all2{qq,1};
        [curCAMmapAll] = returnCAMmap(activation_lastconv, trans_matrix);
        curCAMmap_crops= squeeze(curCAMmapAll(:,:,1,:));%14*14*1*10
        curCAMmapLarge_crops = imresize(curCAMmap_crops,[256 256]);%256*256*10
        curCAMmap_image = mergeTenCrop(curCAMmapLarge_crops);
        cur=imresize( curCAMmap_image,[aa,bb]);
        cur=(cur-min(cur(:)))./(max(cur(:))-min(cur(:)));
        respath=strcat(resfolder,imgname(qq).name);
        imwrite(cur,respath);
    end
end