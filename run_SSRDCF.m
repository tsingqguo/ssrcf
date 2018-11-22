function results = run_SSRDCF(seq, res_path, bSaveImage, parameters)

% function results = run_SSRDCF()
% bSaveImage = 1;
% video_path = 'sequences/Hiding';
% [seq, ~] = load_video_info_zc(video_path);

close all
addpath(genpath('Processing/'));
addpath(genpath('training/'));

seq.format = 'otb';
params = SetParams(seq);
[params, data] = PrepareData(params);

time = 0;
%% ablation study
params.ablation = parameters.ablation;
switch parameters.ablation
    case '-W_'
        data.reg.model{2}=[];
    case '-W_t'
        data.reg.model{3}=[];
    case '-W_c'
        data.reg.model{1}=[];
    case '-W_t_'
        data.reg.model{2}=[];
        data.reg.model{3}=[];
end
for frame = 1:data.seq.num_frames
    data.seq.frame = frame;
%   frame
    data.seq.im = imread(params.s_frames{data.seq.frame});
    if size(data.seq.im,3) > 1 && data.seq.colorImage == false
        data.seq.im = data.seq.im(:,:,1);
    end

    tic();
    [params, data] = Detection(params, data);
    [params, data] = FilterUpdate(params, data);
    time = time + toc();
    
    data.seq.im_prev = data.seq.im;
    
    if bSaveImage
        Visualization(bSaveImage, params.selector, data.seq.frame, data.seq.im, data.obj.pos, data.obj.target_sz);
    end
end


fps = numel(params.s_frames) / time;

% disp(['fps: ' num2str(fps)])

results.type = 'rect';
results.res = data.obj.rects;%each row is a rectangle
results.fps = fps;
results.selectors = data.seq.selectors;
end