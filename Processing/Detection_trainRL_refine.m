function [params,data,train_set,trainstate] = Detection_trainRL_refine(params, data, gt_rect,train_set)

trainstate =0;
if data.seq.frame > 1
    old_pos = inf(size(data.obj.pos));
    iter = 1;
    
    %translation search
    while iter <= params.refinement_iterations && any(old_pos ~= data.obj.pos)
        % Get multi-resolution image
        for scale_ind = 1:params.number_of_scales
            data.setup.multires_pixel_template(:,:,:,scale_ind) = ...
                get_pixels(data.seq.im, data.obj.pos, round(data.obj.sz*data.obj.currentScaleFactor*data.setup.scaleFactors(scale_ind)), data.obj.sz);
        end
        
        xt = bsxfun(@times,get_features(data.setup.multires_pixel_template,params.t_features,params.t_global),data.setup.cos_window);
        
        xtf = fft2(xt);
        
        %% calculate response of object filters 
        oresponsef = permute(sum(bsxfun(@times, data.objf.hf, xtf), 3), [1 2 4 3]);
        
        % if we undersampled features, we want to interpolate the
        % response so it has the same size as the image patch
        if params.interpolate_response == 2
            % use dynamic interp size
            data.obj.interp_sz = floor(size(data.obj.y) * data.setup.featureRatio * data.obj.currentScaleFactor);
        end
        responsef_padded = resizeDFT2(oresponsef, data.obj.interp_sz);
        
        % response
        response = ifft2(responsef_padded, 'symmetric');
        
        % find maximum
        if params.interpolate_response == 3
            [disp_row, disp_col, sind] = resp_gradient_ascent(response, responsef_padded, params.gradient_ascent_iterations, single(params.gradient_step_size), data.setup.ky, data.setup.kx, data.use_sz);
        elseif params.interpolate_response == 4
            [max_response, disp_row, disp_col, sind] = resp_newton(response, responsef_padded, params.gradient_ascent_iterations, data.obj.ky, data.obj.kx, data.obj.use_sz);
        else
            [row, col, sind] = ind2sub(size(response), find(response == max(response(:)), 1));
            disp_row = mod(row - 1 + floor((data.interp_sz(1)-1)/2), data.interp_sz(1)) - floor((data.interp_sz(1)-1)/2);
            disp_col = mod(col - 1 + floor((data.interp_sz(2)-1)/2), data.interp_sz(2)) - floor((data.interp_sz(2)-1)/2);
        end
        
        % fprintf('Processing frame %d : %f\n', data.seq.frame, max_response);
        
        %% perform selector
        slt_response = response(:,:,sind);
        if params.enableCSR && ~isempty(data.reg.model);
           tmpselector = cal_selector(data.reg.model,ifftshift(slt_response));
           params.selector = tmpselector(1);
        end
         
        %% perform detection
        switch params.selector          
            case -1 % context filter
                [params, data] = FilterUpdate(params, data);
                filters = data.conf;
         
            case 1 % object filter
                filters = data.objf;
        end
          
        if params.selector==-1
            %% calculate response of object filters
            responsef = permute(sum(bsxfun(@times, filters.hf, xtf), 3), [1 2 4 3]);
            
            % if we undersampled features, we want to interpolate the
            % response so it has the same size as the image patch
            if params.interpolate_response == 2
                % use dynamic interp size
                data.interp_sz = floor(size(data.y) * data.setup.featureRatio * data.obj.currentScaleFactor);
            end
            responsef_padded = resizeDFT2(responsef, data.obj.interp_sz);
            
            % response
            response = ifft2(responsef_padded, 'symmetric');
            
            % find maximum
            if params.interpolate_response == 3
                [disp_row, disp_col, sind] = resp_gradient_ascent(response, responsef_padded, params.gradient_ascent_iterations, single(params.gradient_step_size), data.setup.ky, data.setup.kx, data.use_sz);
            elseif params.interpolate_response == 4
                [max_response, disp_row, disp_col, sind] = resp_newton(response, responsef_padded, params.gradient_ascent_iterations, data.obj.ky, data.obj.kx, data.obj.use_sz);
            else
                [row, col, sind] = ind2sub(size(response), find(response == max(response(:)), 1));
                disp_row = mod(row - 1 + floor((data.interp_sz(1)-1)/2), data.interp_sz(1)) - floor((data.interp_sz(1)-1)/2);
                disp_col = mod(col - 1 + floor((data.interp_sz(2)-1)/2), data.interp_sz(2)) - floor((data.interp_sz(2)-1)/2);
            end
        end
        
        % calculate translation
        switch params.interpolate_response
            case 0
                translation_vec = round([disp_row, disp_col] * data.setup.featureRatio * data.obj.currentScaleFactor * data.setup.scaleFactors(sind));
            case 1
                translation_vec = round([disp_row, disp_col] * data.obj.currentScaleFactor * data.setup.scaleFactors(sind));
            case 2
                translation_vec = round([disp_row, disp_col] * data.setup.scaleFactors(sind));
            case 3
                translation_vec = round([disp_row, disp_col] * data.setup.featureRatio * data.obj.currentScaleFactor * data.setup.scaleFactors(sind));
            case 4
                translation_vec = round([disp_row, disp_col] * data.setup.featureRatio * data.obj.currentScaleFactor * data.setup.scaleFactors(sind));
        end
        
        oldcurrentScaleFactor = data.obj.currentScaleFactor;
        % set the scale
        data.obj.currentScaleFactor = data.obj.currentScaleFactor * data.setup.scaleFactors(sind);
        % adjust to make sure we are not to large or to small
        if data.obj.currentScaleFactor < data.setup.min_scale_factor
            data.obj.currentScaleFactor = data.setup.min_scale_factor;
        elseif data.obj.currentScaleFactor > data.setup.max_scale_factor
            data.obj.currentScaleFactor = data.setup.max_scale_factor;
        end
        
        % update position
        old_pos = data.obj.pos;
        data.obj.pos = data.obj.pos + translation_vec;
        data.obj.pos_prev = old_pos;
        data.obj.currentScaleFactor_prev = oldcurrentScaleFactor;
        
        % calcalute the ground truth action
        gt_pos = [gt_rect(2)+gt_rect(4)./2,gt_rect(1)+gt_rect(3)./2];
        cle = norm(gt_pos-data.obj.pos);        
           
        if cle>=5 && params.selector==-1 && max_response>0.1
%             if ~slt_frames(data.seq.frame)
                num_sample = numel(train_set);
                train_set(num_sample+1).response = ifftshift(slt_response);
                train_set(num_sample+1).gt_sltor = 1;
                train_set(num_sample+1).objsz = round(data.obj.target_sz./data.setup.featureRatio);
%                 slt_frames(data.seq.frame)=1;
                for ti=1:numel(train_set)
                    gt_data(ti,:) = train_set(ti).gt_sltor;
                end
                
                classes = unique(gt_data);
                if numel(classes)>1
                    data.reg.model = train_svm(train_set);
                    trainstate = 1;
                end
%             end
        elseif (cle<1||(cle>=5 && max_response<0.1))&& params.selector==1 
%             if ~slt_frames(data.seq.frame)
                num_sample = numel(train_set);
                train_set(num_sample+1).response = ifftshift(slt_response);
                if cle<1
                    train_set(num_sample+1).gt_sltor = 0;
                elseif cle>5 && max_response<0.1
                    train_set(num_sample+1).gt_sltor = -1;
                end
                train_set(num_sample+1).objsz = round(data.obj.target_sz./data.setup.featureRatio);
%                 slt_frames(data.seq.frame)=1;
                for ti=1:numel(train_set)
                    gt_data(ti,:) = train_set(ti).gt_sltor;
                end
                classes = unique(gt_data);
                if numel(classes)>1
                    data.reg.model = train_svm(train_set);
                    trainstate = 1;
                end
%             end
        elseif cle>=5 && params.selector==0
%             if ~slt_frames(data.seq.frame)
                num_sample = numel(train_set);
                train_set(num_sample+1).response = ifftshift(slt_response);
                train_set(num_sample+1).gt_sltor = 1;
                train_set(num_sample+1).objsz = round(data.obj.target_sz./data.setup.featureRatio);
%                 slt_frames(data.seq.frame)=1;
                for ti=1:numel(train_set)
                    gt_data(ti,:) = train_set(ti).gt_sltor;
                end
                classes = unique(gt_data);
                if numel(classes)>1
                    data.reg.model = train_svm(train_set);
                    trainstate = 1;
                end
%             end
        end
        
        
        %
        data.obj.target_sz = floor(data.obj.base_target_sz * data.obj.currentScaleFactor);
        
        %save position and calculate FPS
        data.obj.rects(data.seq.frame,:) = [data.obj.pos([2,1]) - floor(data.obj.target_sz([2,1])/2), data.obj.target_sz([2,1])];

        iter = iter + 1;
    end
end

end

function selector = cal_selector(svmmodel,response)

    [max_response,contrast] = cal_resfeat(response);
    scores = zeros(3,1);
    for si=1:numel(svmmodel)
        if ~isempty(svmmodel{si})
            [~,score]=predict(svmmodel{si},[max_response,contrast]);
            scores(si) = score(2);
        end
    end

    [~,selector] = max(scores);
    if scores(1)==scores(2)
        selector = 2;
    end
    selector = selector -2;
end

function svmmodels = train_svm(train_set)
    
    data = [];
    gt_data = [];
    svmmodels = cell(3,1);
    for ti=1:numel(train_set)
        response = train_set(ti).response;
        objsz = train_set(ti).objsz;
        [data(ti,1),data(ti,2)] = cal_resfeat(response);
        gt_data(ti,:) = train_set(ti).gt_sltor+2; % 1: context 2: no update 3: foreground 
    end
    
    classes = unique(gt_data);
    for ci = 1:numel(classes)
        gt_ci = (gt_data==classes(ci));
        svmmodels{classes(ci)} = fitcsvm(data,gt_ci,'KernelFunction','gaussian','Standardize',true,...
            'BoxConstraint',Inf,'ClassNames',[false true]);
    end

    if 0
        %% debug show svm results
        d2 = 1; d1 = 0.01;
        [x1Grid,x2Grid] = meshgrid(min(data(:,1)):d1:max(data(:,1)),...
            min(data(:,2)):d2:max(data(:,2)));
        xGrid = [x1Grid(:),x2Grid(:)];
        N = size(xGrid,1);
        Scores = zeros(N,numel(classes));

        for j = classes';
            [~,score] = predict(svmmodels{j},xGrid);
            Scores(:,j) = score(:,2); % Second column contains positive-class scores
        end
        
        [~,maxScore] = max(Scores,[],2);
        
        figure(2)
        h(classes') = gscatter(xGrid(:,1),xGrid(:,2),maxScore,...
            [0.1 0.5 0.5; 0.5 0.1 0.5; 0.5 0.5 0.1]);
        hold on
        h(classes'+3) = gscatter(data(:,1),data(:,2),gt_data);
        title('{\bf Iris Classification Regions}');
        xlabel('Petal Length (cm)');
        ylabel('Petal Width (cm)');
        legend(h,{'setosa region','versicolor region','virginica region',...
            'observed setosa','observed versicolor','observed virginica'},...
            'Location','Northwest');
        axis tight
        hold off
    end

end

function [max_response,contrast] = cal_resfeat(response)
% calculte the maximum of response
    max_response = max(response(:));
    min_response = min(response(:));
    contrast = (max_response-min_response).^2./mean((response(:)-min_response).^2);
end