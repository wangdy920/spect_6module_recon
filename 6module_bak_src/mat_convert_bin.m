function mat_convert_bin(inputMat, outputBin, varargin)
%MAT_CONVERT_BIN Convert projection MAT data to SPECT-recon BIN layout.
%
%   mat_convert_bin(inputMat, outputBin)
%   mat_convert_bin(inputMat, outputBin, 'VarName', 'matrix', ...)
%
% Optional name-value pairs (defaults match legacy script):
%   'VarName'                - MAT variable name (default: 'matrix')
%   'ProjDataColumn'         - 2D list projection column (default: 3)
%   'RaysPerAngle2D'         - rays per angle for 2D list input (default: [])
%   'Sort2DByFirstColumn'    - sort 2D rows by column 1 (default: false)
%   'SortOrder'              - 'ascend' or 'descend' (default: 'ascend')
%
% Output BIN format:
%   int32 num_angles
%   int32 rays_per_angle
%   float64 data[num_angles * rays_per_angle]

p = inputParser;
addRequired(p, 'inputMat', @(x) ischar(x) || isstring(x));
addRequired(p, 'outputBin', @(x) ischar(x) || isstring(x));
addParameter(p, 'VarName', 'matrix', @(x) ischar(x) || isstring(x));
addParameter(p, 'ProjDataColumn', 3, @(x) isnumeric(x) && isscalar(x));
addParameter(p, 'RaysPerAngle2D', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
addParameter(p, 'Sort2DByFirstColumn', false, @islogical);
addParameter(p, 'SortOrder', 'ascend', @(x) ischar(x) || isstring(x));
parse(p, inputMat, outputBin, varargin{:});

input_mat_path = char(p.Results.inputMat);
output_bin_path = char(p.Results.outputBin);
var_name = char(p.Results.VarName);
proj_data_column = p.Results.ProjDataColumn;
rays_per_angle_2d = p.Results.RaysPerAngle2D;
sort_2d_by_first_column = p.Results.Sort2DByFirstColumn;
sort_order = char(p.Results.SortOrder);

fprintf('============================================================\n');
fprintf('投影数据 -> CUDA二进制 转换\n');
fprintf('============================================================\n');

fprintf('\n[1/4] 加载 MAT 文件...\n');
S = load(input_mat_path, var_name);

if ~isfield(S, var_name)
    error('MAT 文件中不存在变量 "%s"', var_name);
end

proj_data = S.(var_name);
clear S;

fprintf('  变量名: %s\n', var_name);
fprintf('  数据维度: %s\n', mat2str(size(proj_data)));

fprintf('\n[2/4] 解析投影数据...\n');

if ndims(proj_data) == 3
    [detZ, detX, num_angles] = size(proj_data);
    rays_per_angle = detZ * detX;

    fprintf('  检测到 3D 投影数据\n');
    fprintf('  detZ = %d, detX = %d, num_angles = %d\n', detZ, detX, num_angles);
    fprintf('  rays_per_angle = %d\n', rays_per_angle);

    proj_perm = permute(proj_data, [2, 1, 3]);
    flat_data = double(proj_perm(:));

elseif ismatrix(proj_data)
    [nrow, ncol] = size(proj_data);

    if ncol <= 10
        fprintf('  检测到 2D 列表投影数据\n');
        fprintf('  数据大小: %d x %d\n', nrow, ncol);

        if isempty(rays_per_angle_2d)
            error(['二维列表数据时，必须设置 RaysPerAngle2D = detZ*detX。\n' ...
                   '例如 ''RaysPerAngle2D'', 64*64']);
        end

        if proj_data_column < 1 || proj_data_column > ncol
            error('ProjDataColumn=%d 超出列范围 1~%d', proj_data_column, ncol);
        end

        if sort_2d_by_first_column
            [~, idx] = sort(proj_data(:,1), sort_order);
            proj_data = proj_data(idx,:);
            fprintf('  已按第1列进行%s排序\n', sort_order);
        end

        flat_data = double(proj_data(:, proj_data_column));
        rays_per_angle = rays_per_angle_2d;

        if mod(numel(flat_data), rays_per_angle) ~= 0
            error(['二维列表数据长度(%d) 不能被 rays_per_angle(%d) 整除。\n' ...
                   '请检查 RaysPerAngle2D 是否正确。'], ...
                   numel(flat_data), rays_per_angle);
        end

        num_angles = numel(flat_data) / rays_per_angle;

        fprintf('  使用第 %d 列作为投影值\n', proj_data_column);
        fprintf('  num_angles = %d\n', num_angles);
        fprintf('  rays_per_angle = %d\n', rays_per_angle);

    else
        error(['检测到二维矩阵，但列数=%d，无法判断是否为投影列表数据。\n' ...
               '若这是 [detZ, detX, angle] 数据，请确认保存为3D数组。'], ncol);
    end

else
    error('不支持的数据维度: ndims = %d', ndims(proj_data));
end

fprintf('\n[3/4] 写入二进制文件...\n');

fid = fopen(output_bin_path, 'wb');
if fid < 0
    error('无法打开输出文件: %s', output_bin_path);
end

fwrite(fid, int32(num_angles), 'int32');
fwrite(fid, int32(rays_per_angle), 'int32');
fwrite(fid, flat_data, 'float64');

fclose(fid);

fprintf('  已写入: %s\n', output_bin_path);
fprintf('  Header:\n');
fprintf('    num_angles     = %d\n', num_angles);
fprintf('    rays_per_angle = %d\n', rays_per_angle);
fprintf('  Data count = %d\n', numel(flat_data));
fprintf('  文件大小 = %.2f MB\n', dir(output_bin_path).bytes / 1024^2);

fprintf('\n[4/4] 回读校验...\n');

fid = fopen(output_bin_path, 'rb');
na_chk  = fread(fid, 1, 'int32');
rpa_chk = fread(fid, 1, 'int32');
data_chk = fread(fid, inf, 'float64');
fclose(fid);

fprintf('  读取到 num_angles     = %d\n', na_chk);
fprintf('  读取到 rays_per_angle = %d\n', rpa_chk);
fprintf('  读取到数据点数        = %d\n', numel(data_chk));

if na_chk ~= num_angles || rpa_chk ~= rays_per_angle || numel(data_chk) ~= numel(flat_data)
    error('回读校验失败！');
end

if max(abs(data_chk - flat_data)) > 0
    error('回读数据与写入数据不一致！');
end

fprintf('\n✅ 转换成功，可直接用于 CUDA 重建。\n');
fprintf('CUDA 调用示例：\n\n');
fprintf('./spect_recon \\\n');
fprintf('  --matrix-dir /mnt/sharedisk2/recon/matrix/PSC_6module_chunks \\\n');
fprintf('  --proj %s \\\n', output_bin_path);
fprintf('  --out recon_results \\\n');
fprintf('  --iters 50 \\\n');
fprintf('  --gpus 1\n\n');

end
