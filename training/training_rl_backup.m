function training_rl()
clear
close all
addpath(genpath('/home/fan/Desktop/Object_Tracking/tracker_benchmark_v1.0/trackers/SSRDCF/Processing/'));
addpath(genpath('/home/fan/Desktop/Object_Tracking/tracker_benchmark_v1.0/trackers/SSRDCF/utils/'));
addpath(genpath('/home/fan/Desktop/Object_Tracking/libsvm/matlab'));
% testing
pathAnno = '/home/fan/Desktop/Object_Tracking/tracker_benchmark_v1.0/anno/';
trainstate=1;
if exist('svmmodel.mat')
    load('svmmodel.mat')
    load('train_set.mat')
else
    svmmodel = [];
    train_set = [];
end
if exist('seqs.mat')
    load('seqs.mat');
else
    seqs = configSeqs_rl;
end

numSeq=length(seqs);
debug = 1;    
while trainstate ==1
    for idxSeq = 3:numSeq
        
        s = seqs{idxSeq};
        nz	= strcat('%0',num2str(s.nz),'d'); %number of zeros in the name of image
        s.len = s.endFrame-s.startFrame+1;
        for i=1:s.len
            image_no = s.startFrame + (i-1);
            id = sprintf(nz,image_no);
            s.s_frames{i} = strcat(s.path,'img/',id,'.',s.ext);
        end
        
        img = imread(s.s_frames{1});
        [imgH,imgW,ch]=size(img);
        
        rect_anno = dlmread([pathAnno s.name '.txt']);
        s.anno = rect_anno;
        s.init_rect = rect_anno(s.startFrame,:);
        if ~isfield(s,'slt_frames')
           s.slt_frames = zeros(s.len,1); 
        end
        if ~isfield(s,'max_tmp')
            s.max_tmp = -Inf;
        end
        [svmmodel,s,train_set,trainstate] = train_SSRDCF(s,svmmodel,train_set);
        close all
        seqs{idxSeq} = s;
        
        if trainstate
           save('svmmodel.mat','svmmodel');
           save('train_set.mat','train_set');
           save('seqs.mat','seqs')
           break; 
        end
        
    end
    if debug
        %% debug show svm results
        data = [];
        gt_data = [];
        
        for ti=1:numel(train_set)
            response = train_set(ti).response;
            objsz = train_set(ti).objsz;
            [data(ti,1),data(ti,2)] = cal_resfeat(response,objsz);
            gt_data(ti,:) = 2*train_set(ti).gt_sltor-1;
        end
    
        d = 0.001;
        [x1Grid,x2Grid] = meshgrid(min(data(:,1)):d:max(data(:,1)),...
            min(data(:,2)):d:max(data(:,2)));
        xGrid = [x1Grid(:),x2Grid(:)];
        [labels,scores] = predict(svmmodel,xGrid);
  
        figure(2)
        plot(data(gt_data>0,1),data(gt_data>0,2),'r.','MarkerSize',15);
        hold on
        plot(data(gt_data<0,1),data(gt_data<0,2),'b.','MarkerSize',15);
        hold on
        plot(data(svmmodel.IsSupportVector,1),data(svmmodel.IsSupportVector,2),'ko');
        contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'g');
    end
end
end

function [max_response,contrast] = cal_resfeat(response,objsz)
    % calculte the maximum of response
    max_response = max(response(:));
%     [row,col] = find(response==max_response);
% 
%     % objsz on response map
%     col_vec = col-ceil(objsz(2)/2):col+floor(objsz(2)/2);
%     row_vec = row-ceil(objsz(1)/2):row+floor(objsz(1)/2);
%     col_vec(col_vec<1) = 1; col_vec(col_vec>size(response,2)) = size(response,2);
%     row_vec(row_vec<1) = 1; row_vec(row_vec>size(response,1)) = size(response,1);
%     [col_vec,ia,~] = unique(col_vec); %row_vec = row_vec(ia);
%     [row_vec,ia,~] = unique(row_vec); %col_vec = col_vec(ia);
% 
%     [res_X,res_Y] = meshgrid(col_vec,row_vec);
%     response_in = response(res_X,res_Y);
%     aver_in = mean(response_in(:));
%     response_out = sum(response(:))-sum(response_in(:));
%     aver_out = response_out./(numel(response(:))-numel(response_in(:)));
%     contrast = aver_in./aver_out;
    min_response = min(response(:));
    contrast = (max_response-min_response).^2./mean((response(:)-min_response).^2);
    
end
