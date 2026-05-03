clc; clear; close all;

input_folder = 'Tiny-Imagenet/Tiny_Orijinal';
noise_levels = [0.1, 0.3, 0.5, 0.7, 0.9];

image_files = dir(fullfile(input_folder, '*.JPEG'));

for n = 1:length(noise_levels)
    density = noise_levels(n);
    output_folder = sprintf('Tiny-Imagenet/Tiny_Noisy_%d', density * 100);

    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    fprintf('%%%d gurultu seviyesi olusturuluyor...\n', density * 100);

    for i = 1:length(image_files)
        img_name = image_files(i).name;
        I = imread(fullfile(input_folder, img_name));
        Noisy_I = imnoise(I, 'salt & pepper', density);
        [~, name, ~] = fileparts(img_name);
        imwrite(Noisy_I, fullfile(output_folder, [name, '.png']));
    end

    fprintf('%%%d klasoru tamamlandi!\n', density * 100);
end

disp('Tum gurultu seviyeleri uretildi!');
