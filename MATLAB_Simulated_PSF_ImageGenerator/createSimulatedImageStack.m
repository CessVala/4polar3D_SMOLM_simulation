% =========================================================================
% Function Name: createSimulatedImageStack.m
%
% Description:
%   This function generates a stack of simulated 4polar3D images with Gaussian-distributed
%   camera noise. It places multiple simulated molecules at predefined positions across
%   four polarization channels (0°, 45°, 90°, 135°) and adds background intensity.
%
% Inputs:
%   - I0input: Theoretical PSF in 0° polarization channel.
%   - I45input: Theoretical PSF in the 45° polarization channel.
%   - I90input: Theoretical PSF in the 90° polarization channel.
%   - I135input: Theoretical PSF in the 135° polarization channel.
%   - bginput: Background intensity (camera offset).
%   - nImg: Number of images to generate in the stack.
%   - wnGen: Width (in pixels) of the square area representing a molecule.
%
% Outputs:
%   - imageBlurBgNoise: Simulated image stack with background and Gaussian noise.
%
% Authors:
%   Cesar Valades-Cruz - Institute of Hydrobiology (IHB), CAS
%
% Date: July 2025
% =========================================================================

function imageBlurBgNoise = createSimulatedImageStack(I0input, I45input, I90input, I135input, bginput, nImg, wnGen)

% Theoretical PSfs of the four polarization channels
I0 = I0input;
I90 = I90input;
I45 = I45input;
I135 = I135input;

bg = bginput;

% Define image size (square matrix)
sizeMatrix = 256;

% Predefined molecule positions (x, y coordinates)
coordinates = [40 50; 20 90; 70 120; 30 160; 100 40; ...
    200 50; 140 100; 200 130; 170 200; 110 170];

% Initialize empty images for each polarization channel
Img2x2_I0 = zeros(sizeMatrix, sizeMatrix);
Img2x2_I45 = zeros(sizeMatrix, sizeMatrix);
Img2x2_I90 = zeros(sizeMatrix, sizeMatrix);
Img2x2_I135 = zeros(sizeMatrix, sizeMatrix);

% Place simulated molecules at the specified positions
for i = 1:size(coordinates, 1)
    coord1 = coordinates(i, 1);
    coord2 = coordinates(i, 2);

    % Assign molecule intensities to square areas (wnGen x wnGen)
    Img2x2_I0(coord1:coord1+wnGen-1, coord2:coord2+wnGen-1) = I0;
    Img2x2_I45(coord1:coord1+wnGen-1, coord2:coord2+wnGen-1) = I45;
    Img2x2_I90(coord1:coord1+wnGen-1, coord2:coord2+wnGen-1) = I90;
    Img2x2_I135(coord1:coord1+wnGen-1, coord2:coord2+wnGen-1) = I135;
end

% Combine the four polarization channels into one 2x2 tiled image
% [I0  I90]
% [I45 I135]
image2 = [Img2x2_I0 Img2x2_I90; Img2x2_I45 Img2x2_I135];

% Add constant background to the image
image_off_bg = image2 + bg;

% Replicate the image to create a stack of nImg images
stackimageblur = repmat(image_off_bg, 1, 1, nImg);

% Define Gaussian noise coefficient (approximated from experimental data)
kg = 8.8 * sqrt(0.023);

% Calculate the standard deviation of the camera noise for each pixel
sigmaI = kg .* sqrt(stackimageblur);

% Add Gaussian noise to the simulated image stack
imageBlurBgNoise = normrnd(stackimageblur, sigmaI);

% Optional: Uncomment to use Poisson noise instead of Gaussian noise
% imageBlurBgNoise = poissrnd(stackimageblur);

end
