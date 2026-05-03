clc; clear; close all;

noise_levels = [10, 30, 50, 70, 90];

for n = 1:length(noise_levels)
    level = noise_levels(n);
    input_folder  = sprintf('Tiny-Imagenet/Tiny_Noisy_%d', level);
    output_folder = sprintf('Tiny-Imagenet/Tiny_Cleaned_DAMF_%d', level);

    if ~exist(input_folder, 'dir'), continue; end
    if ~exist(output_folder, 'dir'), mkdir(output_folder); end

    image_files = dir(fullfile(input_folder, '*.png'));
    fprintf('DAMF: %%%d seviyesi basliyor... (%d resim)\n', level, length(image_files));

    for idx = 1:length(image_files)
        img_name = image_files(idx).name;
        I = imread(fullfile(input_folder, img_name));

        if size(I, 3) == 1
            I = cat(3, I, I, I);
        end

        out = zeros(size(I), 'uint8');
        for ch = 1:3
            out(:,:,ch) = damf_channel(double(I(:,:,ch)));
        end

        imwrite(out, fullfile(output_folder, img_name));
    end

    fprintf('DAMF: %%%d tamamlandi.\n', level);
end

disp('DAMF tum seviyeler bitti!');


function out = damf_channel(A)
    [m, n] = size(A);
    out = A;
    max_win = 4;

    for i = 1:m
        for j = 1:n
            if A(i,j) ~= 0 && A(i,j) ~= 255
                continue;
            end

            replaced = false;
            for w = 1:max_win
                r1 = max(1, i-w); r2 = min(m, i+w);
                c1 = max(1, j-w); c2 = min(n, j+w);
                patch = A(r1:r2, c1:c2);
                valid = patch(patch > 0 & patch < 255);
                if ~isempty(valid)
                    out(i,j) = median(valid(:));
                    replaced = true;
                    break;
                end
            end

            if ~replaced
                r1 = max(1, i-max_win); r2 = min(m, i+max_win);
                c1 = max(1, j-max_win); c2 = min(n, j+max_win);
                patch = A(r1:r2, c1:c2);
                out(i,j) = median(patch(:));
            end
        end
    end

    out = uint8(out);
end
