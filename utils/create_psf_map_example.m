% create_psf_map_example.m
% 这是一个示例脚本，用于生成 C++ SPECT-recon 程序所需的 PSF Map 文件。
% PSF Map 定义了重建图像空间中每个体素位置的 PSF Sigma 值（像素单位）。

clear; clc;

%% 1. 配置参数
nx = 128;
ny = 128;
nz = 128;
output_mat_file = 'psf_map.mat';
output_bin_file = 'psf_map.bin';

%% 2. 创建 PSF Map 数据
% 这里作为一个示例，我们创建一个随深度（y轴）线性变化的 PSF Map
% 假设探测器在 y = 0 处，物体中心在 y = ny/2
% Sigma = min_sigma + slope * distance

min_sigma = 0.8; % 最小模糊 (像素)
max_sigma = 2.5; % 最大模糊 (像素)

% 初始化 3D 矩阵
psf_map = zeros(nx, ny, nz, 'single'); % 使用 single (float32)

[X, Y, Z] = meshgrid(1:nx, 1:ny, 1:nz);

% 示例逻辑：假设模糊程度主要与 Y 轴（深度）有关
% 归一化 Y 坐标到 [0, 1]
normalized_dist = (Y - 1) / (ny - 1);

% 计算 Sigma
psf_map = min_sigma + (max_sigma - min_sigma) * normalized_dist;

% 可选：添加一些随机扰动或特定区域的模糊
% psf_map(60:70, 60:70, 60:70) = 3.0;

%% 3. 保存为 .mat 文件 (推荐)
% C++ 程序会读取名为 'psf_map' 的变量
fprintf('Saving to %s...\n', output_mat_file);
save(output_mat_file, 'psf_map');

%% 4. 保存为二进制 .bin 文件 (Float32 Raw)
% 如果不使用 MAT 文件，也可以保存为纯二进制
fprintf('Saving to %s...\n', output_bin_file);
fid = fopen(output_bin_file, 'wb');
if fid == -1
    error('Cannot open file for writing');
end
fwrite(fid, psf_map, 'single'); % 必须写入 float32
fclose(fid);

fprintf('Done! You can use these files with SPECT-recon:\n');
fprintf('  SPECT-recon ... -psfMap %s\n', output_mat_file);
fprintf('  SPECT-recon ... -psfMap %s\n', output_bin_file);

% 可视化中间切片
figure;
imagesc(squeeze(psf_map(:, :, round(nz/2))));
colorbar;
title('PSF Sigma Map (Central Slice)');
axis image;
