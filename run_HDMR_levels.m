clc; clear; close all;

noise_levels = [10, 30, 50, 70, 90];

for n = 1:length(noise_levels)
    level = noise_levels(n);
    % input_folder  = sprintf('Tiny-Imagenet/Tiny_Noisy_%d', level);
    % output_folder = sprintf('Tiny-Imagenet/Tiny_Cleaned_HDMR_%d', level);
    input_folder  = sprintf('480p-Test/Noisy_%d', level);
    output_folder = sprintf('480p-Test/Cleaned_HDMR_%d', level);

    if ~exist(input_folder, 'dir'), continue; end
    if ~exist(output_folder, 'dir'), mkdir(output_folder); end

    image_files = dir(fullfile(input_folder, '*.png'));
    fprintf('\nHDMR: %%%d seviyesi basliyor...\n', level);

    parfor idx = 1:length(image_files)
        img_name = image_files(idx).name;
        I_noisy  = imread(fullfile(input_folder, img_name));

        if size(I_noisy, 3) == 1
            I_noisy = cat(3, I_noisy, I_noisy, I_noisy);
        end

        A  = double(I_noisy);
        A1 = hdmr_filter_channel(A(:,:,1));
        A2 = hdmr_filter_channel(A(:,:,2));
        A3 = hdmr_filter_channel(A(:,:,3));

        AA = zeros(size(I_noisy), 'uint8');
        AA(:,:,1) = uint8(A1);
        AA(:,:,2) = uint8(A2);
        AA(:,:,3) = uint8(A3);

        imwrite(AA, fullfile(output_folder, img_name));

        if mod(idx, 50) == 0
            fprintf('  %d resim tamamlandi.\n', idx);
        end
    end

    fprintf('%%%d seviyesi tamamlandi.\n', level);
end

disp('HDMR tamamlandi!');


function A = hdmr_filter_channel(A)
    [m, n] = size(A);

    for i = 1:m
        for j = 1:n
            if A(i,j) == 255
                A(i,j) = 0;
            end
        end
    end

    pA = padarray(A, [3 3], 'symmetric');
    [m, n] = size(A);

    for i = 1:m
        for j = 1:n
            if A(i,j) == 0
                if sum(sum(pA(i+2:i+4, j+2:j+4))) ~= 0
                    R1 = pA(i+2:i+4, j+2:j+4);
                    [~, ~, b] = standartiki(R1);
                    [f0, f11, f12, ~] = hdmr(b);
                    A(i,j) = (median(f11) + median(f12)) / 2 + f0;

                elseif sum(sum(pA(i+1:i+5, j+1:j+5))) ~= 0
                    R1 = pA(i+1:i+5, j+1:j+5);
                    [~, ~, b] = standartiki(R1);
                    [f0, f11, f12, ~] = hdmr(b);
                    A(i,j) = (median(f11) + median(f12)) / 2 + f0;

                elseif sum(sum(pA(i:i+6, j:j+6))) ~= 0
                    R1 = pA(i:i+6, j:j+6);
                    [~, ~, b] = standartiki(R1);
                    [f0, f11, f12, ~] = hdmr(b);
                    A(i,j) = (median(f11) + median(f12)) / 2 + f0;
                end
            end
        end
    end

    for kk = 1:3
        pA = padarray(A, [1 1], 'symmetric');
        [m, n] = size(A);

        for i = 1:m
            for j = 1:n
                if A(i,j) == 0
                    if sum(sum(pA(i:i+2, j:j+2))) ~= 0
                        R1 = pA(i:i+2, j:j+2);
                        [~, ~, b] = standartiki(R1);
                        [f0, f11, f12, ~] = hdmr(b);
                        A(i,j) = (median(f11) + median(f12)) / 2 + f0;
                    end
                end
            end
        end
    end
end
