function [params, data,train_set,trainstate,max_tmp,slt_frames] = Detection_trainRL(params, data, gt_rect,max_tmp,train_set,slt_frames,init_num_sample)

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
           tmpselector = cal_selector(data.reg.model,ifftshift(slt_response),round(data.obj.target_sz./data.setup.featureRatio));
           params.selector = tmpselector(1);
        end
         
        %% perform detection
        switch params.selector          
            case {-1,0} % context filter
                [params, data] = FilterUpdate(params, data);
                filters = data.conf;
         
            case 1 % object filter
                filters = data.objf;
        end
          
        if params.selector~=1
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
        
        % if the regression model get gt action
        
        if ((cle<=20) && (params.selector==-1))||((cle>20) && params.selector==1)
            if ~slt_frames(data.seq.frame)
                num_sample = numel(train_set);
                train_set(num_sample+1).response = ifftshift(slt_response);
                train_set(num_sample+1).gt_sltor = (cle<=20);
                train_set(num_sample+1).objsz = round(data.obj.target_sz./data.setup.featureRatio);
                init_num_sample = numel(train_set);
                data.reg.model = train_svm(train_set);
                trainstate = 1;
                slt_frames(data.seq.frame)=1;
            end
        else
            if max_response>max_tmp
                if ~slt_frames(data.seq.frame)
                    train_set(init_num_sample+1).response = ifftshift(slt_response);
                    train_set(init_num_sample+1).gt_sltor = (cle<20);
                    train_set(init_num_sample+1).objsz = round(data.obj.target_sz./data.setup.featureRatio);
                    max_tmp = max_response;
                    slt_frames(data.seq.frame)=1;
                end
            end
        end
        
        %
        data.obj.target_sz = floor(data.obj.base_target_sz * data.obj.currentScaleFactor);
        
        %save position and calculate FPS
        data.obj.rects(data.seq.frame,:) = [data.obj.pos([2,1]) - floor(data.obj.target_sz([2,1])/2), data.obj.target_sz([2,1])];

        iter = iter + 1;
    end
end

end

function selector = cal_selector(svmmodel,response,objsz)

    [max_response,contrast] = cal_resfeat(response,objsz);
    [selector,~]=predict(svmmodel,[max_response,contrast]);
%     selector = score>0;
    
end

function svmmodel = train_svm(train_set)
    
    data = [];
    gt_data = [];
    
    for ti=1:numel(train_set)
        response = train_set(ti).response;
        objsz = train_set(ti).objsz;
        [data(ti,1),data(ti,2)] = cal_resfeat(response,objsz);
        gt_data(ti,:) = 2*train_set(ti).gt_sltor-1; 
    end
    
    svmmodel = fitcsvm(data,gt_data,...
        'BoxConstraint',Inf,'ClassNames',[-1,1]);
%     svmmodel = svmtrain(gt_data,data);


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