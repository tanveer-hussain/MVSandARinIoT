clc
warning off
run 'F:\Research\Deep learning\matconvnet-1.0-beta25\matlab\vl_setupnn.m'
videoNumber=1;


if(~exist('net','var'))
%     net = load('imagenet-vgg-m.mat');
net = load('imagenet-vgg-verydeep-19.mat');
    disp('Model loaded!');
    endlab

mm=1;
analysis=zeros(1,3);
conv3Features=zeros(1,169);
lables=zeros(1,1);


jump=2;
class=0;
DataSet='F:\Research\Multi View Video Summarization\ICCV\Codes\Activity Recognition\UCF101\UCF-101';
MainFolder=dir(DataSet);
MSize=length(MainFolder);
video={};


TotalFeatures = zeros(1,15000);
labels = [];
labelss = {};


for tt=3:MSize
    class=class+1;
    addpath(strcat(strcat(DataSet,'\\',MainFolder(tt).name)));
    CFolder=dir(strcat(strcat(DataSet,'\\',MainFolder(tt).name),'\\*.avi'));
    CSize=length(CFolder);
        for videoNumber=1:CSize
            path=strcat(strcat(DataSet,'\\',MainFolder(tt).name),'\\',CFolder(videoNumber).name);
            vidObj = VideoReader(path); 
            fprintf('%s\n',path);
            numFrames=vidObj.NumberOfFrames;
            if numFrames > 30
                k=0;
                fc8Features = zeros(1,1000);
                for i=1:jump:numFrames - 30
                    k=k+1;
                    img = read(vidObj,i);
                    im_ = single(img) ; % note: 255 range
                    im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
                    im_ = im_ - net.meta.normalization.averageImage ;
                    res = vl_simplenn(net, im_) ;
                    fc8 = res(43).x;
                    fc8Features(k,:)= reshape(fc8,[1 1000]);
                    if mod(k,15)== 0
                        TotalFeatures(mm,:) =reshape(fc8Features,[1 15000]);
                        labels(mm) = class;
                        labelss{mm} = MainFolder(tt).name;
                        mm=mm+1;
                        fc8Features = zeros(1,1000);
                        k=0;
                    end
                    
                end
                
            end
        end
        
end

save('Total_Features_UCF50_VGG19.mat','TotalFeatures','labels','labelss','-v7.3');
% 
% X = TotalFeatures';
% hiddenSize = 1000;
% autoenc = trainAutoencoder(X,hiddenSize,...
%         'L2WeightRegularization',0.004,...
%         'SparsityRegularization',4,...
%         'SparsityProportion',0.15);
%     
% xCons = predict(autoenc, TotalFeatures');
%     

% Features = encode(autoenc,TotalFeatures');
% 
% xx = Features';
% xx = table(xx);
% 
% lablesss = cell2table(labelss');
% 
% TrainFeatures =  [xx lablesss];

% figure, imagesc(xx(1,:));
% figure, imagesc(TotalFeatures(1,:));
