function psf_map = calc_psf_map_from_sources(image_vol, source_coords, window_size, output_file)
% CALC_PSF_MAP_FROM_SOURCES Calculate 3D PSF map from point sources
%
% Usage:
%   psf_map = calc_psf_map_from_sources(image_vol, source_coords, window_size, output_file)
%
% Inputs:
%   image_vol     - 3D image volume (matrix) or path to .mat file
%   source_coords - Nx3 matrix of [x, y, z] coordinates of point sources (1-based)
%   window_size   - Scalar (radius) or [wx, wy, wz] (full width) for ROI
%   output_file   - (Optional) Path to save the resulting .mat/.bin file
%
% Outputs:
%   psf_map       - 3D volume of Sigma values (in pixels)
%
% Example:
%   coords = [32, 32, 32; 96, 32, 32; ...];
%   psf_map = calc_psf_map_from_sources(recon_img, coords, 15, 'psf_map.mat');

    %% 1. Input Handling
    if ischar(image_vol) || isstring(image_vol)
        fprintf('Loading image from %s...\n', image_vol);
        data = load(image_vol);
        field_names = fieldnames(data);
        % Heuristic to find the image variable
        if isfield(data, 'recon')
            image_vol = data.recon;
        elseif isfield(data, 'img')
            image_vol = data.img;
        else
            image_vol = data.(field_names{1});
            warning('Variable name unknown, using first variable: %s', field_names{1});
        end
    end
    
    [nx, ny, nz] = size(image_vol);
    num_sources = size(source_coords, 1);
    
    if length(window_size) == 1
        wr = round(window_size);
        window_size = [2*wr+1, 2*wr+1, 2*wr+1];
    end
    
    fprintf('Analyzing %d sources in %dx%dx%d volume...\n', num_sources, nx, ny, nz);
    
    %% 2. Analyze Each Source
    measured_sigmas = zeros(num_sources, 1);
    valid_sources = false(num_sources, 1);
    
    % Debug plotting
    figure('Name', 'PSF Analysis');
    
    for i = 1:num_sources
        cx = round(source_coords(i, 1));
        cy = round(source_coords(i, 2));
        cz = round(source_coords(i, 3));
        
        % Define ROI indices
        x_range = max(1, cx - floor(window_size(1)/2)) : min(nx, cx + floor(window_size(1)/2));
        y_range = max(1, cy - floor(window_size(2)/2)) : min(ny, cy + floor(window_size(2)/2));
        z_range = max(1, cz - floor(window_size(3)/2)) : min(nz, cz + floor(window_size(3)/2));
        
        if isempty(x_range) || isempty(y_range) || isempty(z_range)
            warning('Source %d at [%d,%d,%d] is out of bounds', i, cx, cy, cz);
            continue;
        end
        
        roi = image_vol(x_range, y_range, z_range);
        
        % Calculate FWHM/Sigma
        try
            stats = fit_3d_gaussian(roi);
            
            % Convert FWHM to Sigma (assuming Gaussian: FWHM = 2.355 * sigma)
            avg_sigma = mean([stats.sigma_x, stats.sigma_y, stats.sigma_z]);
            
            measured_sigmas(i) = avg_sigma;
            valid_sources(i) = true;
            
            fprintf('Source %d: Pos=[%d,%d,%d]\n', i, cx, cy, cz);
            fprintf('  FWHM=[%.2f, %.2f, %.2f] (Avg: %.2f)\n', stats.fwhm_x, stats.fwhm_y, stats.fwhm_z, mean([stats.fwhm_x, stats.fwhm_y, stats.fwhm_z]));
            fprintf('  FWTM=[%.2f, %.2f, %.2f] (Avg: %.2f)\n', stats.fwtm_x, stats.fwtm_y, stats.fwtm_z, mean([stats.fwtm_x, stats.fwtm_y, stats.fwtm_z]));
            fprintf('  Sigma=%.2f\n', avg_sigma);
                
        catch ME
            warning('Failed to fit source %d: %s', i, ME.message);
        end
    end
    
    %% 3. Fit and Interpolate
    if sum(valid_sources) < 2
        error('Not enough valid sources to generate a map.');
    end
    
    valid_coords = source_coords(valid_sources, :);
    valid_vals = measured_sigmas(valid_sources);
    
    fprintf('Interpolating map from %d valid measurements...\n', length(valid_vals));
    
    % Method 1: Scattered Interpolation (Linear)
    % Good for exact interpolation of measured points
    F = scatteredInterpolant(valid_coords(:,1), valid_coords(:,2), valid_coords(:,3), valid_vals, 'linear', 'nearest');
    
    % Method 2: Polynomial Fit (e.g., if we know it varies linearly with depth Y)
    % poly_model = fitlm(valid_coords, valid_vals, 'linear');
    
    % Generate Grid
    % Use ndgrid for consistent matrix indexing [nx, ny, nz]
    [Xg, Yg, Zg] = ndgrid(1:nx, 1:ny, 1:nz);
    
    psf_map = F(Xg, Yg, Zg);
    
    % Smooth the map slightly to avoid artifacts?
    % psf_map = smooth3(psf_map, 'box', 3);
    
    %% 4. Save Output
    if exist('output_file', 'var') && ~isempty(output_file)
        [~, ~, ext] = fileparts(output_file);
        if strcmpi(ext, '.mat')
            save(output_file, 'psf_map', 'valid_coords', 'valid_vals');
        elseif strcmpi(ext, '.bin')
            fid = fopen(output_file, 'wb');
            fwrite(fid, psf_map, 'single');
            fclose(fid);
        end
        fprintf('Saved PSF map to %s\n', output_file);
    end
    
    %% Visualization
    subplot(1,2,1);
    scatter3(valid_coords(:,1), valid_coords(:,2), valid_coords(:,3), 50, valid_vals, 'filled');
    title('Measured Sigma Points');
    colorbar; axis equal; grid on;
    xlabel('X'); ylabel('Y'); zlabel('Z');
    
    subplot(1,2,2);
    slice_idx = round(nz/2);
    imagesc(psf_map(:,:,slice_idx));
    title(sprintf('Interpolated Map (Slice Z=%d)', slice_idx));
    colorbar; axis image;
    xlabel('Y'); ylabel('X'); % Matrix coords
    
end

function stats = fit_3d_gaussian(roi)
    % Simple 1D Gaussian fits on marginals for robustness
    % ROI: 3D array
    
    % Marginalize
    prof_x = squeeze(sum(sum(roi, 2), 3));
    prof_y = squeeze(sum(sum(roi, 1), 3));
    prof_z = squeeze(sum(sum(roi, 1), 2));
    
    [stats.sigma_x, stats.fwhm_x, stats.fwtm_x] = fit_1d_gaussian(prof_x);
    [stats.sigma_y, stats.fwhm_y, stats.fwtm_y] = fit_1d_gaussian(prof_y);
    [stats.sigma_z, stats.fwhm_z, stats.fwtm_z] = fit_1d_gaussian(prof_z);
end    
    stats.fwtm_x = stats.sigma_x * 4.29; % FWTM = 4.29 * sigma for Gaussian
    stats.fwtm_y = stats.sigma_y * 4.29;
    stats.fwtm_z = stats.sigma_z * 4.29;
end

function [sigma, fwhm, fwtm] = fit_1d_gaussian(profile)
    % Fit A * exp(-(x-mu)^2 / (2*sigma^2)) + B
    profile = double(profile(:));
    x = (1:length(profile))';
    
    % Initial guess
    [max_val, idx] = max(profile);
    min_val = min(profile);
    
    % Estimate width (FWHM)
    half_max = (max_val - min_val) / 2 + min_val;
    above_half = find(profile > half_max);
    if isempty(above_half)
        width_guess = 1;
    else
        width_guess = (above_half(end) - above_half(1)) / 2.355;
    end
    if width_guess < 0.5, width_guess = 0.5; end
    
    % Fit type
    ft = fittype('a*exp(-((x-b)/c)^2) + d');
    opts = fitoptions(ft);
    opts.StartPoint = [max_val-min_val, x(idx), width_guess*sqrt(2), min_val];
    opts.Lower = [0, 1, 0, 0];
    opts.Upper = [Inf, length(profile), Inf, max_val];
    opts.Display = 'off';
    
    try
        [f, ~] = fit(x, profile, ft, opts);
        % c in fit is sigma*sqrt(2) -> sigma = c/sqrt(2)
        sigma = f.c / sqrt(2);
    catch
        % Fallback: Moment estimation
        sum_p = sum(profile);
        mu = sum(x .* profile) / sum_p;
        sigma = sqrt(sum((x - mu).^2 .* profile) / sum_p);
    end
    
    fwhm = sigma * 2.355;
    fwtm = sigma * 4.29;
end
