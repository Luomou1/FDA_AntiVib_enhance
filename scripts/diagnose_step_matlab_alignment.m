function diagnose_step_matlab_alignment(input_path, output_path)
% 使用固定参考点运行原 MATLAB 台阶去噪流程，并导出逐阶段对照数据。

data_nm = load(input_path);
[rows, cols] = size(data_nm);
selected_rows = [100; 900; 500];
selected_cols = [100; 100; 200];

three_points_xyz = [
    selected_cols - 1, ...
    selected_rows - 1, ...
    data_nm(sub2ind(size(data_nm), selected_rows, selected_cols))
];
vector1 = three_points_xyz(2, :) - three_points_xyz(1, :);
vector2 = three_points_xyz(3, :) - three_points_xyz(1, :);
plane_normal = cross(vector1, vector2);
raw_coeffs = zeros(3, 1);
raw_coeffs(1) = -plane_normal(1) / plane_normal(3);
raw_coeffs(2) = -plane_normal(2) / plane_normal(3);
raw_coeffs(3) = three_points_xyz(1, 3) ...
    - raw_coeffs(1) * three_points_xyz(1, 1) ...
    - raw_coeffs(2) * three_points_xyz(1, 2);
[X, Y] = meshgrid(1:cols, 1:rows);
raw_fitted_surface = raw_coeffs(1) * (X - 1) ...
    + raw_coeffs(2) * (Y - 1) + raw_coeffs(3);
fitted_surface = raw_fitted_surface - mean(raw_fitted_surface(:));
coeffs = [raw_coeffs(1); raw_coeffs(2); ...
    raw_coeffs(3) - mean(raw_fitted_surface(:))];
leveled = data_nm - fitted_surface;

[hist_counts, hist_centers] = hist(leveled(:), 50);
smoothed_counts = smooth(hist_counts, 3);
[peaks, peak_locs] = findpeaks(smoothed_counts, ...
    'MinPeakHeight', max(smoothed_counts) * 0.05, ...
    'MinPeakDistance', 5, 'SortStr', 'descend');

if length(peak_locs) >= 2
    height1 = hist_centers(peak_locs(1));
    height2 = hist_centers(peak_locs(2));
    if height1 < height2
        temp = height1;
        height1 = height2;
        height2 = temp;
    end
    threshold = (height1 * peaks(1) + height2 * peaks(2)) ...
        / (peaks(1) + peaks(2));
    if abs(height1 - height2) ...
            < (max(leveled(:)) - min(leveled(:))) * 0.1
        threshold = mean(leveled(:));
    end
else
    threshold = (prctile(leveled(:), 30) + prctile(leveled(:), 70)) / 2;
end

cluster_map = zeros(size(data_nm), 'uint8');
cluster_map(leveled > threshold) = 2;
cluster_map(leveled <= threshold) = 1;
initial_counts = [sum(cluster_map(:) == 1), sum(cluster_map(:) == 2)];
if any(initial_counts / numel(cluster_map) < 0.05)
    threshold = prctile(leveled(:), 50);
    cluster_map(leveled > threshold) = 2;
    cluster_map(leveled <= threshold) = 1;
end

se = strel('disk', 2);
raw_cluster_map = cluster_map;
high_mask_opened = imopen(cluster_map == 2, se);
low_mask_opened = imopen(cluster_map == 1, se);
high_mask_dilated = imdilate(high_mask_opened, se);
low_mask_dilated = imdilate(low_mask_opened, se);
high_mask_cleaned = imclose(high_mask_opened, se);
low_mask_cleaned = imclose(low_mask_opened, se);
high_mask_morphology = high_mask_cleaned;
low_mask_morphology = low_mask_cleaned;
overlap = high_mask_cleaned & low_mask_cleaned;
low_mask_cleaned = low_mask_cleaned & ~overlap;
unassigned = ~(high_mask_cleaned | low_mask_cleaned);
unassigned_data = leveled(unassigned);
unassigned_indices = find(unassigned);
high_indices = unassigned_indices(unassigned_data > threshold);
low_indices = unassigned_indices(unassigned_data <= threshold);
high_mask_cleaned(high_indices) = true;
low_mask_cleaned(low_indices) = true;
cluster_map(:) = 0;
cluster_map(high_mask_cleaned) = 2;
cluster_map(low_mask_cleaned) = 1;

pre_layer_std = [
    std(leveled(cluster_map == 1)), ...
    std(leveled(cluster_map == 2))
];
processed = leveled;
noise_counts = zeros(1, 2);
component_counts = zeros(1, 2);
gaussian_kernel = fspecial('gaussian', [3 3], 0.8);
processed_after_low_repairs = [];
processed_after_low_filter = [];

for layer = 1:2
    layer_mask = cluster_map == layer;
    layer_data = leveled(layer_mask);
    layer_mean = mean(layer_data);
    layer_std = std(layer_data);
    layer_noise_mask = layer_mask ...
        & abs(leveled - layer_mean) > 3 * layer_std;
    noise_counts(layer) = sum(layer_noise_mask(:));
    labeled_noise = bwlabel(layer_noise_mask);
    region_stats = regionprops(labeled_noise, 'Area', 'PixelIdxList');
    component_counts(layer) = length(region_stats);

    for i = 1:length(region_stats)
        area = region_stats(i).Area;
        pixel_idx = region_stats(i).PixelIdxList;
        if area <= 4
            window_size = 3;
        elseif area <= 16
            window_size = 5;
        else
            window_size = 7;
        end
        half_win = floor(window_size / 2);

        for j = 1:length(pixel_idx)
            [r, c] = ind2sub(size(layer_noise_mask), pixel_idx(j));
            r_min = max(1, r - half_win);
            r_max = min(rows, r + half_win);
            c_min = max(1, c - half_win);
            c_max = min(cols, c + half_win);
            local_region = leveled(r_min:r_max, c_min:c_max);
            local_layer_mask = layer_mask(r_min:r_max, c_min:c_max);
            local_noise_mask = labeled_noise(r_min:r_max, c_min:c_max) > 0;
            valid_values = local_region(local_layer_mask & ~local_noise_mask);
            if isempty(valid_values)
                processed(r, c) = layer_mean;
            else
                processed(r, c) = median(valid_values);
            end
        end
    end

    if layer == 1
        processed_after_low_repairs = processed;
    end
    large_regions = find([region_stats.Area] > 20);
    for i = large_regions
        pixel_idx = region_stats(i).PixelIdxList;
        [rows_idx, cols_idx] = ind2sub(size(layer_noise_mask), pixel_idx);
        r_min = max(1, min(rows_idx) - 1);
        r_max = min(rows, max(rows_idx) + 1);
        c_min = max(1, min(cols_idx) - 1);
        c_max = min(cols, max(cols_idx) + 1);
        region = processed(r_min:r_max, c_min:c_max);
        region_mask = layer_mask(r_min:r_max, c_min:c_max);
        filtered_region = imfilter(region, gaussian_kernel, 'replicate');
        processed(r_min:r_max, c_min:c_max) = ...
            region_mask .* filtered_region + ~region_mask .* region;
    end
    if layer == 1
        processed_after_low_filter = processed;
    end
end

processed = processed + mean(data_nm(:)) - mean(processed(:));
post_layer_std = [
    std(processed(cluster_map == 1)), ...
    std(processed(cluster_map == 2))
];
final_counts = [sum(cluster_map(:) == 1), sum(cluster_map(:) == 2)];
save(output_path, 'coeffs', 'fitted_surface', 'leveled', ...
    'hist_counts', 'hist_centers', 'smoothed_counts', 'threshold', ...
    'raw_cluster_map', 'high_mask_opened', 'low_mask_opened', ...
    'high_mask_dilated', 'low_mask_dilated', ...
    'high_mask_morphology', 'low_mask_morphology', ...
    'cluster_map', 'initial_counts', 'final_counts', 'pre_layer_std', ...
    'processed', 'post_layer_std', 'noise_counts', 'component_counts');
save(output_path, 'processed_after_low_repairs', ...
    'processed_after_low_filter', '-append');
end
