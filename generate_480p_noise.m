clc; clear; close all;

input_folder = '480p-Test/Originals_resized';
noise_levels = [0.1, 0.3, 0.5, 0.7, 0.9];

image_files = dir(fullfile(input_folder, '*.png'));

for n = 1:length(noise_levels)
    density = noise_levels(n);
    output_folder = sprintf('480p-Test/Noisy_%d', density * 100);

    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    fprintf('%%%d gurultu seviyesi olusturuluyor...\n', density * 100);

    for i = 1:length(image_files)
        img_name = image_files(i).name;
        I = imread(fullfile(input_folder, img_name));
        Noisy_I = imnoise(I, 'salt & pepper', density);
        imwrite(Noisy_I, fullfile(output_folder, img_name));
    end

    fprintf('%%%d klasoru tamamlandi!\n', density * 100);
end

disp('Tum gurultu seviyeleri uretildi!');
