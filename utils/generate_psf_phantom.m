function [phantom, source_coords] = generate_psf_phantom(img_size, sphere_radius, spacing, output_file)
% GENERATE_PSF_PHANTOM Generate phantom with spherical sources
%
% Usage:
%   [phantom, coords] = generate_psf_phantom(img_size, radius, spacing, output_file)
%
% Inputs:
%   img_size      - [nx, ny, nz]
%   sphere_radius - Radius of spheres (pixels)
%   spacing       - Spacing between spheres (pixels)
%   output_file   - (Optional) Path to save .mat file
%
% Outputs:
%   phantom       - 3D volume
%   source_coords - Nx3 matrix of source centers

    if nargin < 1, img_size = [128, 128, 128]; end
    if nargin < 2, sphere_radius = 2; end
    if nargin < 3, spacing = 20; end
    
    nx = img_size(1); ny = img_size(2); nz = img_size(3);
    phantom = zeros(nx, ny, nz, 'single');
    
    % Generate grid of sources
    % We want them centered and spaced by 'spacing'
    
    % Define grid points
    x_grid = spacing:spacing:(nx-spacing);
    y_grid = spacing:spacing:(ny-spacing);
    z_grid = spacing:spacing:(nz-spacing);
    
    [Xg, Yg, Zg] = meshgrid(x_grid, y_grid, z_grid);
    source_coords = [Xg(:), Yg(:), Zg(:)];
    
    fprintf('Generating %d spherical sources (r=%.1f)...\n', size(source_coords, 1), sphere_radius);
    
    % Create spheres
    [X, Y, Z] = ndgrid(1:nx, 1:ny, 1:nz); % Use ndgrid for matrix indexing consistency
    
    for i = 1:size(source_coords, 1)
        cx = source_coords(i, 1);
        cy = source_coords(i, 2);
        cz = source_coords(i, 3);
        
        dist_sq = (X - cx).^2 + (Y - cy).^2 + (Z - cz).^2;
        mask = dist_sq <= sphere_radius^2;
        phantom(mask) = 1.0;
    end
    
    if nargin >= 4 && ~isempty(output_file)
        save(output_file, 'phantom', 'source_coords', 'img_size', 'sphere_radius', 'spacing');
        fprintf('Saved phantom to %s\n', output_file);
    end
    
    % Visualize
    figure;
    slice_idx = round(nz/2);
    imagesc(phantom(:,:,slice_idx));
    title('Phantom Slice');
    axis image; colorbar;
    hold on;
    % Plot centers on this slice
    in_slice = abs(source_coords(:,3) - slice_idx) < spacing/2;
    plot(source_coords(in_slice, 2), source_coords(in_slice, 1), 'rx'); % Note: plot is x,y (col, row)
end
