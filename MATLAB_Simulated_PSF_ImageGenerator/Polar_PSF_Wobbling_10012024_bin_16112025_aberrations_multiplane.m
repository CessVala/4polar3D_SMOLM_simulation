% =========================================================================
% Function Name: Polar_PSF_Wobbling_10012024_bin_16112025_aberrations_multiplane.m
%
% Description:
%   This function models the theoretical PSFs of polarized single-molecule emitters
%   under 4polar3D microscopy conditions, taking into account the polarization-dependent
%   components, microscope parameters, and dipole wobbling angles.
%
%   The output images are resized using a binning factor and are provided for
%   each polarization channel: 0°, 90°, 45°, and 135°.
%
% Inputs:
%   - n_1: Refractive index after objective (sample space - oil)
%   - n_2: Refractive index before objective (imaging space)
%   - NA: Numerical aperture of the objective.
%   - lambda: Wavelength [µm].
%   - red_factor: Reduction factor.
%   - RHO_dipole_deg: Rho angle.
%   - ETA_dipole_deg: Eta angle.
%   - DELTA_dipole_deg: Delta angle.
%   - f1: Focal length of lens 1 before camera.
%   - f2: Focal length of lens 2 before camera.
%   - ccd_px_size: Camera pixel size [µm].
%   - show_fig: Flag to display generated PSF figures (1 = show, 0 = skip).
%   - binsize: Binning factor to resize the output images.
%   - W_astig: H-V astigmatism, factor in units of wavelength
%   - W_coma: V Coma, factor in units of wavelength
% NOTE both aberrations main axis is V (e.g. astig is a V parabola)
% and defined in a normalized pupil
%
% Outputs:
%   - I0_teo: Simulated PSF image in the 0° polarization channel.
%   - I90_teo: Simulated PSF image in the 90° polarization channel.
%   - I45_teo: Simulated PSF image in the 45° polarization channel.
%   - I135_teo: Simulated PSF image in the 135° polarization channel.
%
%
% Authors:
%   Sophie Brasselet - Institut Fresnel
%   Miguel Sison - Institut Fresnel
%   Luis Aleman-Castaneda - Institut Fresnel
%   Cesar Valades-Cruz - Institute of Hydrobiology (IHB), CAS
%
% Date: November 2025 (+Aberration)
% =========================================================================


function [I0_teo,I90_teo,I45_teo,I135_teo]=Polar_PSF_Wobbling_10012024_bin_16112025_aberrations_multiplane(n_1,n_2,NA,lambda,red_factor,RHO_dipole_deg,ETA_dipole_deg,DELTA_dipole_deg,f1,f2,ccd_px_size,show_fig,binsize,W_astig,W_coma)
warning('off')
%% Model the PSF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% to see the BFP and PSFs, chose show_fig

tic

%% Parameters of the microscope
n = n_1/n_2;
OM = 100; %objective magnification
f_TL = 200*1E3; %
focal_length = f_TL/OM; %3.333333*1E3 ;     % focal length of the objective (in um)

K0=2*pi/lambda;
K_3=K0*n_1; % wavevector in the Fourier imaging plane

theta_3max = asin(NA/n_1); % full NA : maximal value of theta_3 (in rad) (masks are introduced below)

% n_z = 3; %distance emitter to coverslip: as a fraction of wavelength: n_z=z/lambda
delta_f = 0.3; %0.4; %distance between two consecutive focal plane [µm]
delta_n_f = -delta_f/lambda; %distance between two consecutive focal plane in lambda unit
n_f0 = 0; %-4.3; %-4.3; %-4.42  position of the emitter (in lambda unit)
% N_f = [n_f0-4*delta_n_f,n_f0-3*delta_n_f,n_f0-2*delta_n_f,n_f0-delta_n_f, n_f0, n_f0+delta_n_f, n_f0+2*delta_n_f, n_f0+3*delta_n_f, n_f0+4*delta_n_f]; %defocus = distance focal plane to coverslip, as a fraction of wavelength: n_f=f/lambda
N_f = [n_f0-delta_n_f, n_f0, n_f0+delta_n_f]; %defocus = distance focal plane to coverslip, as a fraction of wavelength: n_f=f/lambda
z_i = 2 ; % &(+Z) 2(0) 3(-Z) chosen z index of N_f; for the graphs
x0 = 0; % [µm]
y0 = 0; % [µm]

M = f2/f1; % magnification of image plane after 4f system (on the camera )

PSF_size = 1.22*lambda*1e3/2/NA; % theoretical size of the PSF [nm]
PSF_size_z = 2*lambda*1e3/NA^2; % theoretical size of the PSF [nm]


% compute simu over range of delta angles and eta angles.
DELTA_dipole=deg2rad(DELTA_dipole_deg);
ETA_dipole = deg2rad(ETA_dipole_deg);
RHO_dipole = deg2rad(RHO_dipole_deg);

%% Loop over different parameters : noise and n_z
EM_POS = 0; %2.9742; %2.7:0.1:3.3; %2:0.1:4;% loop over different emitter positions n_z %2.9742 = focus
n_z = EM_POS; %distance emitter to coverslip: as a fraction of wavelength: n_z=z/lambda

%% dipoles and electric field in the (p,s,z) frame: set a cartesian coordinates grid

%Initialize results of localization and orientation arrays

%%
% Cartesian coordinate grid for k vectors in the BFP
n_theta_3 =  20;                                            % number of values. --> n_theta_3 high --> delta_theta_3 low (delta_k in BFP low) --> FOV in image plane high
n_phi_3 = 20;
delta_theta_3 = theta_3max/n_theta_3;                       % interval between two theta_3-steps
delta_phi_3 = 2*pi/n_phi_3;                                 % interval between two phi_3-steps
%theta = 0:delta_theta_3:(theta_3max);         % angle created with the optical axis (-Z) -delta_theta_3
phi = 0:delta_phi_3:(2*pi);                     % angle in the plane normal to optical axis  -delta_phi_3
theta = 0:delta_theta_3:(theta_3max);         % angle created with the optical axis (-Z) -delta_theta_3

% Cartesian coordinate grid for k vectors in the BFP
x_max = max(max(focal_length*sin(theta')*cos(phi)));        % set the limits for x and y, in um
x_min = min(min(focal_length*sin(theta')*cos(phi)));
y_max = max(max(focal_length*sin(theta')*sin(phi)));
y_min = min(min(focal_length*sin(theta')*sin(phi)));

delta_x = (x_max-x_min)/n_theta_3;
delta_y = (y_max-y_min)/n_phi_3;

x = x_min: delta_x :(x_max-delta_x);        % create x grid
y = y_min: delta_y :(y_max-delta_y);        % create y grid

x = x' * ones(1,length(y));
y = ones(length(x),1) * y;
x(sqrt(x.^2 + y.^2) > focal_length*sin(theta_3max)) = NaN;      % Select just the values inside the objective's NA
y(sqrt(x.^2 + y.^2) > focal_length*sin(theta_3max)) = NaN;

temp=0; %mask1 = 1; mask2 = 1;
% mask for channel 1 (reduced) = 0/90
mask1    = ( sqrt(x.^2 + y.^2) < (focal_length*n_2/n_1)*red_factor);
mask1(mask1==0)       = temp;
% mask for channel 2 (under critical) = 45/135
mask2    = ( sqrt(x.^2 + y.^2) < ((focal_length*n_2/n_1)) );
mask2(mask2==0)       = temp;

theta_3 = asin(sqrt(x.^2 + y.^2)./focal_length);                  % angle created with the optical axis (-Z)
phi_3 = atan2(y,x);                                               % angle in the plane normal to optical axis

n_x = size(x);                                            % size of x/y matrices (number of values)

% Compute pad_size according to the px size on the camera  (ccd_px_size)
pad_size = round((lambda*f_TL*M-ccd_px_size*(x_max-x_min))/(2*delta_x*ccd_px_size));

% define x and y grid in image plane after fft (if no scaling with respect to the lens)
x_max_im =  1/delta_x/2; % in [ µm^-1]
x_min_im = -1/delta_x/2; % in [ µm^-1]
y_max_im =  1/delta_y/2; % in [ µm^-1]
y_min_im = -1/delta_y/2; % in [ µm^-1]
delta_x_im = 1/(x_max-x_min); % pixel size in image plane (still no padding)
delta_y_im = 1/(y_max-y_min);
x_im = x_min_im:delta_x_im:(x_max_im-delta_x_im);
y_im = y_min_im:delta_y_im:(y_max_im-delta_y_im);

% x and y grid in image plane after fft taking into account the lens (scaling with the tube lens --> *lambda*L_t)
x_im_scaled = x_im*lambda*f_TL; % in  µm
y_im_scaled = y_im*lambda*f_TL; % in  µm

% new FOV in BFP after padding
x_max_pad = x_max + pad_size*delta_x;
x_min_pad = x_min - pad_size*delta_x;
y_max_pad = y_max + pad_size*delta_y;
y_min_pad = y_min - pad_size*delta_y;

% define x y grid in image plane after fft and padding of the BFP (if no scaling with respect to the lens)
delta_x_im_pad = 1/(x_max_pad-x_min_pad); % new pixel size in image plane after padding
delta_y_im_pad = 1/(y_max_pad-y_min_pad);
x_im_pad = -(n_x(1)/2+pad_size)*delta_x_im_pad:delta_x_im_pad:(n_x(1)/2+pad_size-1)*delta_x_im_pad; % in [ µm^-1]
y_im_pad = -(n_x(1)/2+pad_size)*delta_y_im_pad:delta_y_im_pad:(n_x(1)/2+pad_size-1)*delta_y_im_pad; % in [ µm^-1]

% x y grid in image plane after fft and padding of the BFP with scaling with  respect to the tube lens --> *lambda*L_t
x_im_scaled_pad = x_im_pad*lambda*f_TL; % in  µm
y_im_scaled_pad = y_im_pad*lambda*f_TL; % in  µm

% x y grid of the image plane on the camera : magnification M after 4f system
x_im_cam = x_im_scaled_pad*M; % in  µm
y_im_cam = y_im_scaled_pad*M; % in  µm
delta_x_cam = delta_x_im_pad*lambda*f_TL*M; % px size on the camera in  µm
delta_y_cam = delta_y_im_pad*lambda*f_TL*M; % in  µm

% real size of the object ( define X Y grid in the object plane, used for psf display at the end of the simualtion)
X = x_im_cam/OM/M; % [ µm]
Y = y_im_cam/OM/M; % [ µm]


%% radiation dipole vector in (p,s,z) frame
i_delta =0;
Delta_dipole = DELTA_dipole;
i_delta = i_delta +1;
% disp(['delta = ' num2str(rad2deg(Delta_dipole))])

i_eta=0;
Eta_dipole = ETA_dipole;
i_eta= i_eta+1;
% disp(['   eta = ' num2str(rad2deg(Eta_dipole))])

i_rho = 0;
Rho_dipole = RHO_dipole;
i_rho = i_rho + 1;
% disp(['      rho = ' num2str(rad2deg(Rho_dipole))])

%% Initialize final intensities in the image plane
Ix = zeros(2*pad_size+n_x(1),2*pad_size+n_x(2),size(N_f,2));
Iy = zeros(2*pad_size+n_x(1),2*pad_size+n_x(2),size(N_f,2));
Iu = zeros(2*pad_size+n_x(1),2*pad_size+n_x(2),size(N_f,2));
Iv = zeros(2*pad_size+n_x(1),2*pad_size+n_x(2),size(N_f,2));
I  = zeros(2*pad_size+n_x(1),2*pad_size+n_x(2),size(N_f,2));

%%
mu3 = (((cos(Delta_dipole/2))^3-1)/(3*cos(Delta_dipole/2)-3));
mu2 = (1-cos(Delta_dipole/2))*(cos(Delta_dipole/2)+2)/6;
mu1 = mu2;

P_dipole3_x = -sin(Eta_dipole).*cos(Rho_dipole).*sqrt(mu3)*ones(n_x);
P_dipole3_y = -sin(Eta_dipole).*sin(Rho_dipole).*sqrt(mu3)*ones(n_x);
P_dipole3_z = cos(Eta_dipole).*sqrt(mu3)*ones(n_x);

P_dipole2_x = -cos(Eta_dipole).*cos(Rho_dipole).*sqrt(mu2)*ones(n_x);
P_dipole2_y = -cos(Eta_dipole).*sin(Rho_dipole).*sqrt(mu2)*ones(n_x);
P_dipole2_z = -sin(Eta_dipole).*sqrt(mu2)*ones(n_x);

P_dipole1_x = -sin(Rho_dipole).*sqrt(mu1)*ones(n_x);
P_dipole1_y = cos(Rho_dipole).*sqrt(mu1)*ones(n_x);
P_dipole1_z = zeros(n_x).*sqrt(mu1);


cos_2=sqrt(1-(n_1/n_2*sin(theta_3)).^2);
cos_1=cos(theta_3);
sin_1=sin(theta_3);

t_p12 = 2*n_2*cos_2./(n_2*cos_1+n_1*cos_2);             % Fresnel transmission coefficient for p-polar
t_s12 = 2*n_2*cos_2./(n_2*cos_2+n_1*cos_1);             % Fresnel transmission coefficient for s-polar

% Electric fields in the BFP, following Yan, et al., Opt. Expr. (2019), par 2.3

cos_phi3 = cos(phi_3);
sin_phi3 = sin(phi_3);
sin_2phi3 = sin(2*phi_3);
rho = sin_1;

E_xx = n* ((cos_1./cos_2).*t_s12.*sin_phi3.^2 + t_p12.*cos_phi3.^2.*sqrt(1-rho.^2));
E_xy = -n/2*sin_2phi3.* ((cos_1./cos_2).*t_s12 - t_p12.*sqrt(1-rho.^2));
E_xz = -n^2*(cos_1./cos_2).*t_p12.*rho.*cos_phi3;

E_yx = -n/2*sin_2phi3.* ((cos_1./cos_2).*t_s12 - t_p12.*sqrt(1-rho.^2));
E_yy = n* ((cos_1./cos_2).*t_s12.*cos_phi3.^2 + t_p12.*sin_phi3.^2.*sqrt(1-rho.^2));
E_yz = -n^2*(cos_1./cos_2).*t_p12.*rho.*sin_phi3;

for i=1:length(N_f)

    n_f =  N_f(i);

    % axial position of the emitter + defocus + lateral position of the emitter
    if n_f<0
        psy_depth = n_z*n_2*2*pi*cos_2 ...
            + n_f*n_2*2*pi*sqrt(1-sin_1.^2)...
            + 2*pi*rho.*(x0*cos_phi3+y0*sin_phi3)/(lambda); %!! added defocus n_f; if focus above the coverslip use water index n_2 %%CORRECTED
        %                         + 2*pi*NA*OM*rho.*(x0*cos_phi3+y0*sin_phi3)/(lambda*sqrt((OM)^2+NA^2)); %!! added defocus n_f; if focus above the coverslip use water index n_2
    elseif n_f>=0
        psy_depth = n_z*n_2*2*pi*cos_2 ...
            + n_f*n_1*2*pi*sqrt(1-sin_1.^2) ...
            +2*pi*rho.*(x0*cos_phi3+y0*sin_phi3)/(lambda); %!! added defocus n_f; if focus below the coverslip use oil index n_1
        %                         +2*pi*NA*OM*rho.*(x0*cos_phi3+y0*sin_phi3)/(lambda*sqrt((OM)^2+NA^2)); %!! added defocus n_f; if focus below the coverslip use oil index n_1
    end
    
    %% For aberrations we need to normalize rho and the variables @ pupil
    wavefront = W_astig*(rho.*cos_phi3./max(rho(:))).^2 ...
        +W_coma*(rho.*cos_phi3./max(rho(:))).*(rho./max(rho(:))).^2;
    aberrations = exp(-1j*2*pi*wavefront); %Wavefront is in wavelength units
    %imagesc(K0*wavefront)
    %imagesc(angle(aberrations))

    %%

    Ex1_BFP = (E_xx.*P_dipole1_x + E_xy.*P_dipole1_y + E_xz.*P_dipole1_z).*exp(1i*psy_depth).*aberrations;
    Ey1_BFP = (E_yx.*P_dipole1_x + E_yy.*P_dipole1_y + E_yz.*P_dipole1_z).*exp(1i*psy_depth).*aberrations;
    Eu1_BFP = (Ex1_BFP + Ey1_BFP)./sqrt(2); %45 deg projection
    Ev1_BFP = (-Ex1_BFP + Ey1_BFP)./sqrt(2); %135 deg projection

    Ex2_BFP = (E_xx.*P_dipole2_x + E_xy.*P_dipole2_y + E_xz.*P_dipole2_z).*exp(1i*psy_depth).*aberrations;
    Ey2_BFP = (E_yx.*P_dipole2_x + E_yy.*P_dipole2_y + E_yz.*P_dipole2_z).*exp(1i*psy_depth).*aberrations;
    Eu2_BFP = (Ex2_BFP + Ey2_BFP)./sqrt(2); %45 deg projection
    Ev2_BFP = (-Ex2_BFP + Ey2_BFP)./sqrt(2); %135 deg projection

    Ex3_BFP = (E_xx.*P_dipole3_x + E_xy.*P_dipole3_y + E_xz.*P_dipole3_z).*exp(1i*psy_depth).*aberrations;
    Ey3_BFP = (E_yx.*P_dipole3_x + E_yy.*P_dipole3_y + E_yz.*P_dipole3_z).*exp(1i*psy_depth).*aberrations;
    Eu3_BFP = (Ex3_BFP + Ey3_BFP)./sqrt(2); %45 deg projection
    Ev3_BFP = (-Ex3_BFP + Ey3_BFP)./sqrt(2); %135 deg projection


    %% Fft to obtain the PSF
    % added mask before padarray
    Ex1_BFP = Ex1_BFP.*mask1;
    Ex2_BFP = Ex2_BFP.*mask1;
    Ex3_BFP = Ex3_BFP.*mask1;
    Ey1_BFP = Ey1_BFP.*mask1;
    Ey2_BFP = Ey2_BFP.*mask1;
    Ey3_BFP = Ey3_BFP.*mask1;
    Eu1_BFP = Eu1_BFP.*mask2;
    Eu2_BFP = Eu2_BFP.*mask2;
    Eu3_BFP = Eu3_BFP.*mask2;
    Ev1_BFP = Ev1_BFP.*mask2;
    Ev2_BFP = Ev2_BFP.*mask2;
    Ev3_BFP = Ev3_BFP.*mask2;

    % eps = 0; % test: is a little value eps affecting the weight of the SAF --> a priori non
    Ex1_BFP(isnan(Ex1_BFP))=eps;    % Substitute NaN values with eps
    Ey1_BFP(isnan(Ey1_BFP))=eps;
    Eu1_BFP(isnan(Eu1_BFP))=eps;
    Ev1_BFP(isnan(Ev1_BFP))=eps;

    Ex1_BFP = padarray(Ex1_BFP,[pad_size,pad_size],eps);    % zero-pad the matrix to have more resolution in the fft
    Ey1_BFP = padarray(Ey1_BFP,[pad_size,pad_size],eps);
    Eu1_BFP = padarray(Eu1_BFP,[pad_size,pad_size],eps);    % zero-pad the matrix to have more resolution in the fft
    Ev1_BFP = padarray(Ev1_BFP,[pad_size,pad_size],eps);

    Ex2_BFP(isnan(Ex2_BFP))=eps;
    Ey2_BFP(isnan(Ey2_BFP))=eps;
    Eu2_BFP(isnan(Eu2_BFP))=eps;
    Ev2_BFP(isnan(Ev2_BFP))=eps;

    Ex2_BFP = padarray(Ex2_BFP,[pad_size,pad_size],eps);
    Ey2_BFP = padarray(Ey2_BFP,[pad_size,pad_size],eps);
    Eu2_BFP = padarray(Eu2_BFP,[pad_size,pad_size],eps);    % zero-pad the matrix to have more resolution in the fft
    Ev2_BFP = padarray(Ev2_BFP,[pad_size,pad_size],eps);

    Ex3_BFP(isnan(Ex3_BFP))=eps;
    Ey3_BFP(isnan(Ey3_BFP))=eps;
    Eu3_BFP(isnan(Eu3_BFP))=eps;
    Ev3_BFP(isnan(Ev3_BFP))=eps;

    Ex3_BFP = padarray(Ex3_BFP,[pad_size,pad_size],eps);
    Ey3_BFP = padarray(Ey3_BFP,[pad_size,pad_size],eps);
    Eu3_BFP = padarray(Eu3_BFP,[pad_size,pad_size],eps);    % zero-pad the matrix to have more resolution in the fft
    Ev3_BFP = padarray(Ev3_BFP,[pad_size,pad_size],eps);

    % I bfp unpolarized
    I_dipole1_BFP = abs(Ex1_BFP).^2 + abs(Ey1_BFP).^2;
    I_dipole2_BFP = abs(Ex2_BFP).^2 + abs(Ey2_BFP).^2;
    I_dipole3_BFP = abs(Ex3_BFP).^2 + abs(Ey3_BFP).^2;
    I_BFP = nansum(nansum(I_dipole1_BFP + I_dipole2_BFP + I_dipole3_BFP));

    % I bfp 0° polarized
    Ix_dipole1_BFP = abs(Ex1_BFP).^2;
    Ix_dipole2_BFP = abs(Ex2_BFP).^2;
    Ix_dipole3_BFP = abs(Ex3_BFP).^2;
    Ix_BFP = nansum(nansum(Ix_dipole1_BFP + Ix_dipole2_BFP + Ix_dipole3_BFP));

    % I bfp 90° polarized
    Iy_dipole1_BFP = abs(Ey1_BFP).^2;
    Iy_dipole2_BFP = abs(Ey2_BFP).^2;
    Iy_dipole3_BFP = abs(Ey3_BFP).^2;
    Iy_BFP = nansum(nansum(Iy_dipole1_BFP + Iy_dipole2_BFP + Iy_dipole3_BFP));

    % I bfp 45° polarized
    Iu_dipole1_BFP = abs(Eu1_BFP).^2;
    Iu_dipole2_BFP = abs(Eu2_BFP).^2;
    Iu_dipole3_BFP = abs(Eu3_BFP).^2;
    Iu_BFP = nansum(nansum(Iu_dipole1_BFP + Iu_dipole2_BFP + Iu_dipole3_BFP));

    % I bfp 135° polarized
    Iv_dipole1_BFP = abs(Ev1_BFP).^2;
    Iv_dipole2_BFP = abs(Ev2_BFP).^2;
    Iv_dipole3_BFP = abs(Ev3_BFP).^2;
    Iv_BFP = nansum(nansum(Iv_dipole1_BFP + Iv_dipole2_BFP + Iv_dipole3_BFP));


    %% PSFs in the image plane
    Ex1_fft = fft2(ifftshift(Ex1_BFP));
    Ey1_fft = fft2(ifftshift(Ey1_BFP));
    Eu1_fft = fft2(ifftshift(Eu1_BFP));
    Ev1_fft = fft2(ifftshift(Ev1_BFP));

    Ex2_fft = fft2(ifftshift(Ex2_BFP));
    Ey2_fft = fft2(ifftshift(Ey2_BFP));
    Eu2_fft = fft2(ifftshift(Eu2_BFP));
    Ev2_fft = fft2(ifftshift(Ev2_BFP));

    Ex3_fft = fft2(ifftshift(Ex3_BFP));
    Ey3_fft = fft2(ifftshift(Ey3_BFP));
    Eu3_fft = fft2(ifftshift(Eu3_BFP));
    Ev3_fft = fft2(ifftshift(Ev3_BFP));

    %total intensity (unpolarized)
    I_fft1 = abs(fftshift(Ex1_fft)).^2 + abs(fftshift(Ey1_fft)).^2;
    I_fft2 = abs(fftshift(Ex2_fft)).^2 + abs(fftshift(Ey2_fft)).^2;
    I_fft3 = abs(fftshift(Ex3_fft)).^2 + abs(fftshift(Ey3_fft)).^2;
    %x intensity (0 ° polarized)
    Ix_fft1 = abs(fftshift(Ex1_fft)).^2;
    Ix_fft2 = abs(fftshift(Ex2_fft)).^2;
    Ix_fft3 = abs(fftshift(Ex3_fft)).^2;
    %y intensity (90 ° polarized)
    Iy_fft1 = abs(fftshift(Ey1_fft)).^2;
    Iy_fft2 = abs(fftshift(Ey2_fft)).^2;
    Iy_fft3 = abs(fftshift(Ey3_fft)).^2;
    %u intensity (90 ° polarized)
    Iu_fft1 = abs(fftshift(Eu1_fft)).^2;
    Iu_fft2 = abs(fftshift(Eu2_fft)).^2;
    Iu_fft3 = abs(fftshift(Eu3_fft)).^2;
    %v intensity (90 ° polarized)
    Iv_fft1 = abs(fftshift(Ev1_fft)).^2;
    Iv_fft2 = abs(fftshift(Ev2_fft)).^2;
    Iv_fft3 = abs(fftshift(Ev3_fft)).^2;


    %% Intensities in the image plane
    I_fft  = (I_fft1 + I_fft2 + I_fft3);
    Ix_fft = (Ix_fft1 + Ix_fft2 + Ix_fft3);
    Iy_fft = (Iy_fft1 + Iy_fft2 + Iy_fft3);
    Iu_fft = (Iu_fft1 + Iu_fft2 + Iu_fft3);
    Iv_fft = (Iv_fft1 + Iv_fft2 + Iv_fft3);

    %save in 3D matrix
    Ix(:,:,i) = Ix_fft;
    Iy(:,:,i) = Iy_fft;
    Iu(:,:,i) = Iu_fft;
    Iv(:,:,i) = Iv_fft;
    I(:,:,i)  = I_fft;

    Ix_BFPimage(:,:,i) = Ix_dipole1_BFP + Ix_dipole2_BFP + Ix_dipole3_BFP;
    Iy_BFPimage(:,:,i) = Iy_dipole1_BFP + Iy_dipole2_BFP + Iy_dipole3_BFP;
    Iu_BFPimage(:,:,i) = Iu_dipole1_BFP + Iu_dipole2_BFP + Iu_dipole3_BFP;
    Iv_BFPimage(:,:,i) = Iv_dipole1_BFP + Iv_dipole2_BFP + Iv_dipole3_BFP;
    I_BFPimage(:,:,i)  = Ix_BFPimage(:,:,i)+Iy_BFPimage(:,:,i)+Iu_BFPimage(:,:,i)+Iv_BFPimage(:,:,i);

end

%% BFP and PSF K matrices
E_ux = (E_xx+E_yx)./sqrt(2);
E_uy = (E_xy+E_yy)./sqrt(2);
E_uz = (E_xz+E_yz)./sqrt(2);
E_vx = (-E_xx+E_yx)./sqrt(2);
E_vy = (-E_xy+E_yy)./sqrt(2);
E_vz = (-E_xz+E_yz)./sqrt(2);
E_xx(isnan(E_xx))=eps;
E_xy(isnan(E_xy))=eps;
E_xz(isnan(E_xz))=eps;
E_yx(isnan(E_yx))=eps;
E_yy(isnan(E_yy))=eps;
E_yz(isnan(E_yz))=eps;
E_ux(isnan(E_ux))=eps;
E_uy(isnan(E_uy))=eps;
E_uz(isnan(E_uz))=eps;
E_vx(isnan(E_vx))=eps;
E_vy(isnan(E_vy))=eps;
E_vz(isnan(E_vz))=eps;

% added mask before padarray
E_xx = E_xx.*mask1;
E_xy = E_xy.*mask1;
E_xz = E_xz.*mask1;
E_yx = E_yx.*mask1;
E_yy = E_yy.*mask1;
E_yz = E_yz.*mask1;
E_ux = E_ux.*mask2;
E_uy = E_uy.*mask2;
E_uz = E_uz.*mask2;
E_vx = E_vx.*mask2;
E_vy = E_vy.*mask2;
E_vz = E_vz.*mask2;

%sin_BFP = padarray(sin_1,[pad_size,pad_size],eps);
Exx_BFP = padarray(E_xx,[pad_size,pad_size],eps);    % zero-pad the matrix to have more resolution in the fft
Exy_BFP = padarray(E_xy,[pad_size,pad_size],eps);    % zero-pad the matrix to have more resolution in the fft
Exz_BFP = padarray(E_xz,[pad_size,pad_size],eps);    % zero-pad the matrix to have more resolution in the fft
Eyx_BFP = padarray(E_yx,[pad_size,pad_size],eps);    % zero-pad the matrix to have more resolution in the fft
Eyy_BFP = padarray(E_yy,[pad_size,pad_size],eps);    % zero-pad the matrix to have more resolution in the fft
Eyz_BFP = padarray(E_yz,[pad_size,pad_size],eps);    % zero-pad the matrix to have more resolution in the fft
Eux_BFP = padarray(E_ux,[pad_size,pad_size],eps);    % zero-pad the matrix to have more resolution in the fft
Euy_BFP = padarray(E_uy,[pad_size,pad_size],eps);    % zero-pad the matrix to have more resolution in the fft
Euz_BFP = padarray(E_uz,[pad_size,pad_size],eps);    % zero-pad the matrix to have more resolution in the fft
Evx_BFP = padarray(E_vx,[pad_size,pad_size],eps);    % zero-pad the matrix to have more resolution in the fft
Evy_BFP = padarray(E_vy,[pad_size,pad_size],eps);    % zero-pad the matrix to have more resolution in the fft
Evz_BFP = padarray(E_vz,[pad_size,pad_size],eps);    % zero-pad the matrix to have more resolution in the fft

Exx_PSF = fft2(ifftshift(Exx_BFP));
Exy_PSF = fft2(ifftshift(Exy_BFP));
Exz_PSF = fft2(ifftshift(Exz_BFP));
Eyx_PSF = fft2(ifftshift(Eyx_BFP));
Eyy_PSF = fft2(ifftshift(Eyy_BFP));
Eyz_PSF = fft2(ifftshift(Eyz_BFP));
Eux_PSF = fft2(ifftshift(Eux_BFP));
Euy_PSF = fft2(ifftshift(Euy_BFP));
Euz_PSF = fft2(ifftshift(Euz_BFP));
Evx_PSF = fft2(ifftshift(Evx_BFP));
Evy_PSF = fft2(ifftshift(Evy_BFP));
Evz_PSF = fft2(ifftshift(Evz_BFP));

%% generation of the KtheoPSF matrix
%Matrix from the PSF integrated
XX0PSF = nansum(nansum(abs(Exx_PSF).^2));
XX90PSF = nansum(nansum(abs(Eyx_PSF).^2));

YY0PSF = nansum(nansum(abs(Exy_PSF).^2));
YY90PSF = nansum(nansum(abs(Eyy_PSF).^2));

ZZ0PSF = nansum(nansum(abs(Exz_PSF).^2));
ZZ90PSF = nansum(nansum(abs(Eyz_PSF).^2));

XX45PSF = nansum(nansum(abs(Eux_PSF).^2));
XX135PSF = nansum(nansum(abs(Evx_PSF).^2));

YY45PSF = nansum(nansum(abs(Euy_PSF).^2));
YY135PSF = nansum(nansum(abs(Evy_PSF).^2));

ZZ45PSF = nansum(nansum(abs(Euz_PSF).^2));
ZZ135PSF = nansum(nansum(abs(Evz_PSF).^2));

XY0PSF = nansum(nansum(2*real(conj(Exx_PSF).* Exy_PSF)));
XY90PSF = nansum(nansum(2*real(conj(Eyx_PSF).* Eyy_PSF)));

XY45PSF = nansum(nansum(2*real(conj(Eux_PSF).* Euy_PSF)));
XY135PSF = nansum(nansum(2*real(conj(Evx_PSF).* Evy_PSF)));

KPSF=[XX0PSF YY0PSF ZZ0PSF XY0PSF; XX90PSF YY90PSF ZZ90PSF XY90PSF; XX45PSF YY45PSF ZZ45PSF XY45PSF; XX135PSF YY135PSF ZZ135PSF XY135PSF];
sumXXPSF = XX0PSF + XX90PSF + XX45PSF + XX135PSF; % this is needed to normalize the Z column in the experimental K's
KtheoPSF = KPSF./sumXXPSF;
sum0PSF = (XX0PSF+YY0PSF+ZZ0PSF)/sumXXPSF; % this is the intensity on all channels (identical) expected from depolarized dipole
sum90PSF = (XX90PSF+YY90PSF+ZZ90PSF)/sumXXPSF; % this is the intensity on all channels (identical) expected from depolarized dipole
sum45PSF = (XX45PSF+YY45PSF+ZZ45PSF)/sumXXPSF; % this is the intensity on all channels (identical) expected from depolarized dipole
sum135PSF = (XX135PSF+YY135PSF+ZZ135PSF)/sumXXPSF; % this is the intensity on all channels (identical) expected from depolarized dipole

%% generation of the KtheoBFP matrix
% % Matrix from the BFP integrated
XX0BFP = nansum(nansum(abs(Exx_BFP).^2));
XX90BFP = nansum(nansum(abs(Eyx_BFP).^2));

YY0BFP = nansum(nansum(abs(Exy_BFP).^2));
YY90BFP = nansum(nansum(abs(Eyy_BFP).^2));

ZZ0BFP = nansum(nansum(abs(Exz_BFP).^2));
ZZ90BFP = nansum(nansum(abs(Eyz_BFP).^2));

XX45BFP = nansum(nansum(abs(Eux_BFP).^2));
XX135BFP = nansum(nansum(abs(Evx_BFP).^2));

YY45BFP = nansum(nansum(abs(Euy_BFP).^2));
YY135BFP = nansum(nansum(abs(Evy_BFP).^2));

ZZ45BFP = nansum(nansum(abs(Euz_BFP).^2));
ZZ135BFP = nansum(nansum(abs(Evz_BFP).^2));

XY0BFP = nansum(nansum(2*real(conj(Exx_BFP).* Exy_BFP)));
XY90BFP = nansum(nansum(2*real(conj(Eyx_BFP).* Eyy_BFP)));

XY45BFP = nansum(nansum(2*real(conj(Eux_BFP).* Euy_BFP)));
XY135BFP = nansum(nansum(2*real(conj(Evx_BFP).* Evy_BFP)));


KBFP=[XX0BFP YY0BFP ZZ0BFP XY0BFP; XX90BFP YY90BFP ZZ90BFP XY90BFP; XX45BFP YY45BFP ZZ45BFP XY45BFP; XX135BFP YY135BFP ZZ135BFP XY135BFP];
KBFP_2D=[XX0BFP YY0BFP XY0BFP; XX90BFP YY90BFP XY90BFP; XX45BFP YY45BFP XY45BFP; XX135BFP YY135BFP XY135BFP];

sumXXBFP = XX0BFP + XX90BFP + XX45BFP + XX135BFP; % this is needed to normalize the Z column in the experimental K's
KtheoBFP = KBFP./sumXXBFP;
KtheoBFP_2D = KBFP_2D./sumXXBFP;


%% Normalize

I = I/sum(I(:)); %sum over the 3 planes <=> split photons on the 3 planes

m_2 =  sum(Ix(:,:,2),'all')+ sum(Iy(:,:,2),'all')+ sum(Iu(:,:,2),'all')+ sum(Iv(:,:,2),'all') ; % normalize plane 2
m_1 =  sum(Ix(:,:,1),'all')+ sum(Iy(:,:,1),'all')+ sum(Iu(:,:,1),'all')+ sum(Iv(:,:,1),'all') ; % normalize plane 1
m_3 =  sum(Ix(:,:,3),'all')+ sum(Iy(:,:,3),'all')+ sum(Iu(:,:,3),'all')+ sum(Iv(:,:,3),'all') ; % normalize plane 3


Ix(:,:,1)= Ix(:,:,1)/m_1;
Ix(:,:,2)= Ix(:,:,2)/m_2;
Ix(:,:,3)= Ix(:,:,3)/m_3;

Iy(:,:,1)= Iy(:,:,1)/m_1;
Iy(:,:,2)= Iy(:,:,2)/m_2;
Iy(:,:,3)= Iy(:,:,3)/m_3;

Iu(:,:,1)= Iu(:,:,1)/m_1;
Iu(:,:,2)= Iu(:,:,2)/m_2;
Iu(:,:,3)= Iu(:,:,3)/m_3;

Iv(:,:,1)= Iv(:,:,1)/m_1;
Iv(:,:,2)= Iv(:,:,2)/m_2;
Iv(:,:,3)= Iv(:,:,3)/m_3;

% save intensities to have backup if add noise
% this is with no noise
I_save = I;
Ix_save = Ix;
Iy_save = Iy;
Iu_save = Iu;
Iv_save = Iv;

%% parameters for GF (outside of noise realization loop)
parameters.pixel_size.xy = ccd_px_size*1e3/M/OM; % nm
parameters.pixel_size.z =  delta_f*1e3; %nm
sigma = [PSF_size/parameters.pixel_size.xy;PSF_size/parameters.pixel_size.xy;PSF_size_z/parameters.pixel_size.z];
parameters.flags.crop =0;
parameters.flags.output = 0;
parameters.par_crop = [];
parameters.par_microscope.Em = lambda*1E3 ; %nm
parameters.par_microscope.Ex = 480 ; %nm
parameters.par_microscope.NA = NA ;
parameters.par_microscope.RI = n_1 ;
parameters.par_microscope.type = 'widefield' ;

%% Figures in the image plane.

if show_fig ==1
    for ii = 1:length(N_f)

        % figure unpolarized PSFs

        n_f =  N_f(ii);
        f0 = figure(i_rho+i_delta+i_eta+10001); f0.Position = [10 10 1200 400]
        subplot(1,length(N_f),ii)
        imagesc(X,Y,I(:,:,ii));
        colormap(gray)
        colorbar('fontsize',10)
        title({'{\itIntensity unpolarized in the image plane}',...
            ['\eta = ' num2str(rad2deg(Eta_dipole)) ' °; \rho = ' num2str(rad2deg(Rho_dipole))...
            ' °; \delta = ' num2str(rad2deg(Delta_dipole)) ' °']...
            [' z = ' num2str(n_z) '\lambda; d = ' num2str(n_f) '\lambda']},'fontsize',10)
        axis equal; axis image;
        xlabel('y [um]','fontsize',14)
        ylabel('x [um]','fontsize',14)
        xlim([-1 1]);ylim([-1 1])


        % figure unpolarized theoretical PSFs

        n_f =  N_f(ii);
        f1 = figure(i_rho+i_delta+i_eta+10002); f1.Position = [10 10 1200 400]
        subplot(1,length(N_f),ii)
        imagesc(X,Y,I_save(:,:,ii));
        colormap(gray)
        colorbar('fontsize',10)
        title({'{\itIntensity unpolarized in the image plane}',...
            ['\eta = ' num2str(rad2deg(Eta_dipole)) ' °; \rho = ' num2str(rad2deg(Rho_dipole))...
            ' °; \delta = ' num2str(rad2deg(Delta_dipole)) ' °']...
            [' z = ' num2str(n_z) '\lambda; d = ' num2str(n_f) '\lambda']},'fontsize',10)
        axis equal; axis image;
        xlabel('y [um]','fontsize',14)
        ylabel('x [um]','fontsize',14)
        xlim([-1 1]);ylim([-1 1])



        % figure unpolarized theoretical PSFs

        n_f =  N_f(ii);
        f3 = figure(i_rho+i_delta+i_eta+10002); f1.Position = [10 10 1200 400]
        subplot(1,length(N_f),ii)
        imagesc(X,Y,Ix_save(:,:,ii));
        colormap(gray)
        colorbar('fontsize',10)
        % title({'{\itIntensity unpolarized in the image plane}',...
        %     ['\eta = ' num2str(rad2deg(Eta_dipole)) ' °; \rho = ' num2str(rad2deg(Rho_dipole))...
        %     ' °; \delta = ' num2str(rad2deg(Delta_dipole)) ' °']...
        %     [' z = ' num2str(n_z) '\lambda; d = ' num2str(n_f) '\lambda']},'fontsize',10)
        axis equal; axis image;
        xlabel('y [um]','fontsize',14)
        ylabel('x [um]','fontsize',14)
        xlim([-1 1]);ylim([-1 1])



    end
end

[I0_teo(:,:,1),I90_teo(:,:,1),I45_teo(:,:,1),I135_teo(:,:,1)]=binning_normalized(Ix_save(:,:,1),Iy_save(:,:,1),Iu_save(:,:,1),Iv_save(:,:,1),binsize,X,Y);
[I0_teo(:,:,2),I90_teo(:,:,2),I45_teo(:,:,2),I135_teo(:,:,2)]=binning_normalized(Ix_save(:,:,2),Iy_save(:,:,2),Iu_save(:,:,2),Iv_save(:,:,2),binsize,X,Y);
[I0_teo(:,:,3),I90_teo(:,:,3),I45_teo(:,:,3),I135_teo(:,:,3)]=binning_normalized(Ix_save(:,:,3),Iy_save(:,:,3),Iu_save(:,:,3),Iv_save(:,:,3),binsize,X,Y);


warning('on')








