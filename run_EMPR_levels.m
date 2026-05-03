clc; clear; close all;

noise_levels = [10, 30, 50, 70, 90];

for n = 1:length(noise_levels)
    level = noise_levels(n);
    input_folder  = sprintf('Tiny-Imagenet/Tiny_Noisy_%d', level);
    output_folder = sprintf('Tiny-Imagenet/Tiny_Cleaned_EMPR_%d', level);

    if ~exist(input_folder, 'dir'), continue; end
    if ~exist(output_folder, 'dir'), mkdir(output_folder); end

    image_files = dir(fullfile(input_folder, '*.png'));
    fprintf('\nEMPR: %%%d seviyesi basliyor...\n', level);

    for i = 1:length(image_files)
        img_name = image_files(i).name;
        I_noisy = imread(fullfile(input_folder, img_name));

        if size(I_noisy, 3) == 1
            I_noisy = cat(3, I_noisy, I_noisy, I_noisy);
        end

        A = double(I_noisy);

        AA = zeros(size(I_noisy), 'uint8');
        AA(:,:,1) = empr_filter_channel(A(:,:,1));
        AA(:,:,2) = empr_filter_channel(A(:,:,2));
        AA(:,:,3) = empr_filter_channel(A(:,:,3));

        imwrite(AA, fullfile(output_folder, img_name));

        if mod(i, 50) == 0
            fprintf('  %d resim tamamlandi.\n', i);
        end
    end

    fprintf('%%%d seviyesi tamamlandi.\n', level);
end

disp('EMPR tamamlandi!');


function A1 = empr_filter_channel(A1)
    [m, n] = size(A1);

    for i = 1:m
        for j = 1:n
            if A1(i,j) == 255
                A1(i,j) = 0;
            end
        end
    end

    pA = padarray(A1, [3 3], 'symmetric');
    [m, n] = size(A1);

    for i = 1:m
        for j = 1:n
            if A1(i,j) == 0
                if sum(sum(pA(i+2:i+4, j+2:j+4))) ~= 0
                    R1 = pA(i+2:i+4, j+2:j+4);
                    [~, ~, b] = standartiki(R1);
                    c(:,:,1) = b; c(:,:,2) = b; c(:,:,3) = b;
                    [s1, s2, s3] = mysupports(c);
                    [~, f0]       = EMPR0(s1, s2, s3, c);
                    [~, f1, f2, ~] = EMPR1(f0, s1, s2, s3, c);
                    A1(i,j) = median(median(f1) + median(f2)) + f0;

                elseif sum(sum(pA(i+1:i+5, j+1:j+5))) ~= 0
                    R1 = pA(i+1:i+5, j+1:j+5);
                    [~, ~, b] = standartiki(R1);
                    c2(:,:,1) = b; c2(:,:,2) = b; c2(:,:,3) = b;
                    [s1, s2, s3] = mysupports(c2);
                    [~, f0]       = EMPR0(s1, s2, s3, c2);
                    [~, f1, f2, ~] = EMPR1(f0, s1, s2, s3, c2);
                    A1(i,j) = median(median(f1) + median(f2)) + f0;

                elseif sum(sum(pA(i:i+6, j:j+6))) ~= 0
                    R1 = pA(i:i+6, j:j+6);
                    [~, ~, b] = standartiki(R1);
                    c3(:,:,1) = b; c3(:,:,2) = b; c3(:,:,3) = b;
                    [s1, s2, s3] = mysupports(c3);
                    [~, f0]       = EMPR0(s1, s2, s3, c3);
                    [~, f1, f2, ~] = EMPR1(f0, s1, s2, s3, c3);
                    A1(i,j) = median(median(f1) + median(f2)) + f0;
                end
            end
        end
    end

    for kk = 1:3
        pA = padarray(A1, [1 1], 'symmetric');
        [m, n] = size(A1);

        for i = 1:m
            for j = 1:n
                if A1(i,j) == 0
                    if sum(sum(pA(i:i+2, j:j+2))) ~= 0
                        R1 = pA(i:i+2, j:j+2);
                        [~, ~, b] = standartiki(R1);
                        c4(:,:,1) = b; c4(:,:,2) = b; c4(:,:,3) = b;
                        [s1, s2, s3] = mysupports(c4);
                        [~, f0]       = EMPR0(s1, s2, s3, c4);
                        [~, f1, f2, ~] = EMPR1(f0, s1, s2, s3, c4);
                        A1(i,j) = median(median(f1) + median(f2)) + f0;
                    end
                end
            end
        end
    end
end
