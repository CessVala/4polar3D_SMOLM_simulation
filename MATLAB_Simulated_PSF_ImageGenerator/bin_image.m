% =========================================================================
% Function Name: bin_image.m
%
% Description:
%   This function performs spatial binning on a 2D image by summing blocks
%   of size binSize x binSize.
%
% Inputs:
%   - A: Input 2D image (matrix).
%   - binSize: Binning factor (size of the square block for binning).
%
% Outputs:
%   - C: Binned image (matrix).
%
% Authors:
%   Cesar Valades-Cruz - Institute of Hydrobiology (IHB), CAS
%
% Date: July 2025
% =========================================================================

function C = bin_image(A, binSize)

    % Sum along the first binSize rows repeatedly across the entire image
    C = sum(reshape(A, binSize, []));
    
    % Reshape the summed result to form intermediate reduced image
    C = reshape(C, size(A, 1) / binSize, [])';
    
    % Sum along the first binSize columns repeatedly across the intermediate image
    C = sum(reshape(C, binSize, []));
    
    % Reshape to obtain the final binned image
    C = reshape(C, size(A, 2) / binSize, [])';

end
