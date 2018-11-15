function [svmmodel,seq,train_set,trainstate] = train_SSRDCF(seq,svmmodel,train_set)


seq.format = 'otb';
params = SetParams(seq);
[params, data] = PrepareData(params);
trainstate = 1;
showImage = 1;
data.reg.model = svmmodel;
init_num_sample = numel(train_set);
for frame = 1:data.seq.num_frames
    
    data.seq.frame = frame;
    data.seq.im = imread(params.s_frames{data.seq.frame});

    if size(data.seq.im,3) > 1 && data.seq.colorImage == false
        data.seq.im = data.seq.im(:,:,1);
    end
    
    [params,data,train_set,trainstate,seq.max_tmp,seq.slt_frames] = Detection_trainRL(params,data,seq.anno(frame,:),seq.max_tmp,train_set,seq.slt_frames,init_num_sample);
    
    if trainstate ==1
        svmmodel = data.reg.model;
        return
    end
    
    [params, data] = FilterUpdate(params, data);
    
    data.seq.im_prev = data.seq.im;
    
    Visualization(showImage, params.selector, data.seq.frame, data.seq.im, data.obj.pos, data.obj.target_sz);
    
end

svmmodel = data.reg.model;
return