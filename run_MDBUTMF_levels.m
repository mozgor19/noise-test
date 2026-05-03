clc; clear; close all;

noise_levels = [10, 30, 50, 70, 90];

for n = 1:length(noise_levels)
    level = noise_levels(n);
    input_folder  = sprintf('Tiny-Imagenet/Tiny_Noisy_%d', level);
    output_folder = sprintf('Tiny-Imagenet/Tiny_Cleaned_MDBUTMF_%d', level);

    if ~exist(input_folder, 'dir'), continue; end
    if ~exist(output_folder, 'dir'), mkdir(output_folder); end

    image_files = dir(fullfile(input_folder, '*.png'));
    fprintf('\nMDBUTMF: %%%d seviyesi basliyor...\n', level);

    for i = 1:length(image_files)
        img_name = image_files(i).name;
        I_noisy = imread(fullfile(input_folder, img_name));

        if size(I_noisy, 3) == 1
            I_noisy = cat(3, I_noisy, I_noisy, I_noisy);
        end

        I_out = zeros(size(I_noisy), 'uint8');
        I_out(:,:,1) = mdbutmf_filter(I_noisy(:,:,1));
        I_out(:,:,2) = mdbutmf_filter(I_noisy(:,:,2));
        I_out(:,:,3) = mdbutmf_filter(I_noisy(:,:,3));

        imwrite(I_out, fullfile(output_folder, img_name));

        if mod(i, 100) == 0, fprintf('  %d resim tamamlandi.\n', i); end
    end
end

disp('MDBUTMF tamamlandi!');


function out = mdbutmf_filter(in)
    [r, c] = size(in);
    out = in;
    in  = double(in);

    for i = 2:r-1
        for j = 2:c-1
            if in(i,j) == 0 || in(i,j) == 255
                window  = in(i-1:i+1, j-1:j+1);
                flat    = reshape(window, 1, 9);
                valid   = flat(flat > 0 & flat < 255);

                if ~isempty(valid)
                    out(i,j) = median(valid);
                else
                    out(i,j) = mean(flat);
                end
            end
        end
    end

    out = uint8(out);
end
