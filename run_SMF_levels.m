clc; clear; close all;

noise_levels = [10, 30, 50, 70, 90];

for n = 1:length(noise_levels)
    level = noise_levels(n);
    input_folder  = sprintf('Tiny-Imagenet/Tiny_Noisy_%d', level);
    output_folder = sprintf('Tiny-Imagenet/Tiny_Cleaned_SMF_%d', level);

    if ~exist(input_folder, 'dir'), continue; end
    if ~exist(output_folder, 'dir'), mkdir(output_folder); end

    image_files = dir(fullfile(input_folder, '*.png'));
    fprintf('\nSMF: %%%d seviyesi basliyor (%d resim)...\n', level, length(image_files));

    for i = 1:length(image_files)
        img_name = image_files(i).name;
        I_noisy = imread(fullfile(input_folder, img_name));

        if size(I_noisy, 3) == 1
            I_noisy = cat(3, I_noisy, I_noisy, I_noisy);
        end

        I_SMF = zeros(size(I_noisy), 'uint8');
        I_SMF(:,:,1) = medfilt2(I_noisy(:,:,1), [3 3]);
        I_SMF(:,:,2) = medfilt2(I_noisy(:,:,2), [3 3]);
        I_SMF(:,:,3) = medfilt2(I_noisy(:,:,3), [3 3]);

        imwrite(I_SMF, fullfile(output_folder, img_name));

        if mod(i, 100) == 0
            fprintf('  %d resim tamamlandi.\n', i);
        end
    end
end

disp('SMF tamamlandi!');
