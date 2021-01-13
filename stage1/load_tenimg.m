function crops_data = load_tenimg(imgpath)
dim=224;
imgname=dir([imgpath]);
len=size(imgname,1);
crops_data = zeros(dim, dim, 3, 10, 'single');
for i=3:len
   imgpath=strcat(imgpath,imgname(i).name);
   im=imread(imgpath);
   im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
   im_data = permute(im_data, [2, 1, 3]);  % flip width and height
   im_data = single(im_data);  % convert from uint8 to single
   im_data = imresize(im_data, [dim dim], 'bilinear');  % resize im_data
   crops_data(:,:,:,i-2)=im_data;
  
end