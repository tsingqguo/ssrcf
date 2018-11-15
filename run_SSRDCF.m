function results = run_SSRDCF(seq, res_path, bSaveImage, parameters)

% function results = run_SSRDCF()
bSaveImage = 1;
% video_path = 'sequences/Hiding';
% [seq, ~] = load_video_info_zc(video_path);

close all
addpath(genpath('Processing/'));
addpath(genpath('training/'));

seq.format = 'otb';
paramsCSR = [];
dataCSR = [];
params = SetParams(seq);
[params, data] = PrepareData(params);

time = 0;

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
    
    Visualization(bSaveImage, params.selector, data.seq.frame, data.seq.im, data.obj.pos, data.obj.target_sz);
    
end


fps = numel(params.s_frames) / time;

% disp(['fps: ' num2str(fps)])

results.type = 'rect';
results.res = data.obj.rects;%each row is a rectangle
results.fps = fps;

end