clc; clear; close all;

noise_levels = [10, 30, 50, 70, 90];

for n = 1:length(noise_levels)
    level = noise_levels(n);
    input_folder  = sprintf('Tiny-Imagenet/Tiny_Noisy_%d', level);
    output_folder = sprintf('Tiny-Imagenet/Tiny_Cleaned_AMF_%d', level);

    if ~exist(input_folder, 'dir'), continue; end
    if ~exist(output_folder, 'dir'), mkdir(output_folder); end

    image_files = dir(fullfile(input_folder, '*.png'));
    fprintf('\nAMF: %%%d seviyesi basliyor...\n', level);

    for i = 1:length(image_files)
        img_name = image_files(i).name;
        I_noisy = imread(fullfile(input_folder, img_name));

        if size(I_noisy, 3) == 1
            I_noisy = cat(3, I_noisy, I_noisy, I_noisy);
        end

        I_AMF = zeros(size(I_noisy), 'uint8');
        I_AMF(:,:,1) = adaptive_median(I_noisy(:,:,1), 7);
        I_AMF(:,:,2) = adaptive_median(I_noisy(:,:,2), 7);
        I_AMF(:,:,3) = adaptive_median(I_noisy(:,:,3), 7);

        imwrite(I_AMF, fullfile(output_folder, img_name));

        if mod(i, 100) == 0, fprintf('  %d resim tamamlandi.\n', i); end
    end
end

disp('AMF tamamlandi!');


function f = adaptive_median(g, Smax)
    f = g; f(:) = 0;
    alreadyProcessed = false(size(g));

    for k = 3:2:Smax
        zmin = ordfilt2(g, 1, ones(k, k), 'symmetric');
        zmax = ordfilt2(g, k*k, ones(k, k), 'symmetric');
        zmed = medfilt2(g, [k k], 'symmetric');

        processUsingLevelB = (zmed > zmin) & (zmax > zmed) & ~alreadyProcessed;
        zB = (g > zmin) & (zmax > g);

        outputZxy  = processUsingLevelB & zB;
        outputZmed = processUsingLevelB & ~zB;

        f(outputZxy)  = g(outputZxy);
        f(outputZmed) = zmed(outputZmed);

        alreadyProcessed = alreadyProcessed | processUsingLevelB;
        if all(alreadyProcessed(:)), break; end
    end

    f(~alreadyProcessed) = zmed(~alreadyProcessed);
end
