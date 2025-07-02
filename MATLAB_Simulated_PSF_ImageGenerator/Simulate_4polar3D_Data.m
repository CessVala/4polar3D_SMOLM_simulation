% =========================================================================
% Script Name: Simulate_4polar3D_Data.m
%
% Description:
%   This script simulates data of 4polar3D microscopy.
%
% Inputs:
%   - The script defines internal simulation parameters such as:
%       * Microscope configuration
%       * Dipole angle ranges
%       * Intensity and background conditions
%
% Outputs:
%   - Synthetic image stacks are saved as .mat files in the specified directory.
%
% Notes:
%   - The PSF generation relies on the function Polar_PSF_Wobbling_10012024_bin.
%   - Background intensity and total PSF intensity can be adjusted in the parameters.
%
% Authors:
%   Cesar Valades-Cruz - Institute of Hydrobiology (IHB), CAS
%
% Date: July 2025
% =========================================================================

clear all
close all
clc

%% Add path
cpath = mfilename('fullpath');
[dir1, ~, ~] = fileparts(cpath);
addpath(genpath(dir1));

folderExperiment = 'C:\Users\valad\Documents\GitHub\4polar_3D_SMOLM_simulation\MATLAB_Simulated_PSF_ImageGenerator\Synthetic_images\';

if ~exist(folderExperiment, 'dir')
    mkdir(folderExperiment)
end

%% Parameters of the microscope
n_2 = 1.33;      % refractive index before objective (imaging space)
n_1 = 1.515;     % refractive index after objective (sample space - oil)
NA = 1.45;       % numerical aperture
lambda = 0.600;  % wavelength (in um)
red_factor = sqrt(0.75); % reduction factor
f1 = 150; f2 = 150;      % focal distances of 1st and 2nd lens [mm] before the camera
ccd_px_size = 6;         % pixel size on the camera sensor [µm]

deltavalues = [100];     % delta angle  [1:10:170]
etavalues = [20];        % eta angle       [1 10:10:90]
rhovalues = [30];        % rho angle   [0:10:180]

wnGen = 17;              % window size for molecule generation
show_fig = 0;            % flag to display PSF figures
nImg = 100;              % number of simulated images

I_bg = [5000 10];        % [Total intensity per PSF, Background per pixel] in photons

%% Set K matrix
K = [0.4200 0.0049 0.0999 0.0000;
     0.0049 0.4200 0.0999 0.0000;
     0.2876 0.2876 0.2256 0.5304;
     0.2876 0.2876 0.2256 -0.5304];

%% Simulation Data

% Create folders for saving results
folderData = [folderExperiment '\Data\'];

if ~exist(folderData, 'dir')
    mkdir(folderData)
end

h = waitbar(0, 'Please wait...');

Delta_matrix = zeros(size(deltavalues,2), size(etavalues,2), size(rhovalues,2), size(I_bg,1));
Rho_matrix = zeros(size(deltavalues,2), size(etavalues,2), size(rhovalues,2), size(I_bg,1));
Eta_matrix = zeros(size(deltavalues,2), size(etavalues,2), size(rhovalues,2), size(I_bg,1));

I0_matrix = zeros(size(deltavalues,2), size(etavalues,2), size(rhovalues,2), size(I_bg,1));
I45_matrix = zeros(size(deltavalues,2), size(etavalues,2), size(rhovalues,2), size(I_bg,1));
I90_matrix = zeros(size(deltavalues,2), size(etavalues,2), size(rhovalues,2), size(I_bg,1));
I135_matrix = zeros(size(deltavalues,2), size(etavalues,2), size(rhovalues,2), size(I_bg,1));

for i = 1:size(etavalues,2)
    waitbar(i / size(etavalues,2), h);
    for j = 1:size(rhovalues,2)
        for k = 1:size(deltavalues,2)
            for cond = 1:size(I_bg,1)

                ETA_dipole_deg = etavalues(i);
                RHO_dipole_deg = rhovalues(j);
                DELTA_dipole_deg = deltavalues(k);

                Itotal = I_bg(cond,1);
                bginput = I_bg(cond,2);

                namefile = ['Image_ITtheory_' num2str(Itotal) '_wGen_' num2str(wnGen) '_bg_' num2str(bginput) '_delta_' num2str(DELTA_dipole_deg,"%0.3d") '_rho_' num2str(RHO_dipole_deg,"%0.3d") '_eta_' num2str(ETA_dipole_deg,"%0.3d") '_lambda_' num2str(lambda) '_All_GaussNoise_bin'];
                namefile = strrep(namefile, '.', 'p');

                if ~exist([folderData namefile '.mat'])

                    Results = [];
                    Img2x2 = [];
                    tempImg = zeros(256, 512, nImg);

                    %% Generate PSF images and stack
                    [I0_teo, I90_teo, I45_teo, I135_teo] = Polar_PSF_Wobbling_10012024_bin(n_1, n_2, NA, lambda, red_factor, RHO_dipole_deg, ETA_dipole_deg, DELTA_dipole_deg, f1, f2, ccd_px_size, show_fig, 2);

                    I0_teo = Itotal .* I0_teo;
                    I90_teo = Itotal .* I90_teo;
                    I45_teo = Itotal .* I45_teo;
                    I135_teo = Itotal .* I135_teo;

                    Img2x2 = createSimulatedImageStack(I0_teo, I45_teo, I90_teo, I135_teo, bginput, nImg, wnGen);

                    Img2x2 = uint16(permute(Img2x2, [3, 1, 2]));
                    save([folderData namefile '.mat'], 'Img2x2')
                end
            end
        end
    end
end

close(h)
