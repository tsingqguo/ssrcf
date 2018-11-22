function [params, data] = Detection(params, data)

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
        if params.enableCSR && ~isempty(data.reg.model)
           params.selector  = cal_selector(data.reg.model,ifftshift(slt_response));
        end
        data.seq.selectors(data.seq.frame) = params.selector;
        %% perform detection
        switch params.selector           
            case -1 % context filter
                params.count_c = params.count_c+1;
                if params.count_c>params.count_thr && ~strcmp(params.ablation,'-W_')
                   params.selector = 0; 
                end
                params.update_c = 1;
                [params, data] = FilterUpdate(params, data);
                filters = data.conf;
                params.update_c = 0;
            case 1 % object filter
                filters = data.objf;
                params.count_c = 0;
            case 0
                params.count_c = 0;
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
        data.obj.pos_prev = old_pos;
        data.obj.pos = data.obj.pos + translation_vec;
        data.obj.currentScaleFactor_prev = oldcurrentScaleFactor;
        
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
    scores = -Inf*ones(3,1);%zeros(3,1);
    for si=1:numel(svmmodel)
        if ~isempty(svmmodel{si})
            [~,score]=predict(svmmodel{si},[max_response,contrast]);
            scores(si) = score(2);
        end
    end

    [~,selector] = max(scores);
    if scores(1)==scores(2) && selector ~=3
        selector = 2;
    end
    selector = selector -2;
    
end

function [max_response,contrast] = cal_resfeat(response)
% calculte the maximum of response
    max_response = max(response(:));
    min_response = min(response(:));
    contrast = (max_response-min_response).^2./mean((response(:)-min_response).^2);
end