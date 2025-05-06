"""
polarFitFuncv2.py
#
# This module contains functions for fitting polarization data
# in the context of 4polar3D Single-Molecule Orientation Localization Microscopy (SMOLM) simulations.
# The functions are designed to process simulated data and extract orientation information.
#
Author: Cesar Valades-Cruz, IHB
Date: April 29, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from PIL import Image
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
import pandas as pd
import math

def getpeaksfromMax(img_arr,thresh,min_distance,window_size):
    # Identify local maxima (peaks) above a threshold using a minimum separation
    
    # Use Otsu's threshold to convert the image to a binary mask
    #thresh = threshold_otsu(img_arr)
    #thresh = 1000
    binary = img_arr > thresh

    # Find the peaks of the spots using peak_local_max
    peaks = peak_local_max(img_arr, threshold_abs=thresh, exclude_border=window_size, min_distance=min_distance)

    # Get the coordinates of the peaks
    x = peaks[:,1]
    y = peaks[:,0]

   
    return x,y

def gauss2d(xy, a, x0, y0, sx, sy, c):
    # 2D Gaussian function used for fitting
    (x,y)=xy
    g2D=a * np.exp(-0.5*((x-x0)/sx)**2 - 0.5*((y-y0)/sy)**2) + c
    return np.ravel(g2D)

def gauss1d(xy, a, x0, y0, sx, c):
    # Simplified isotropic 2D Gaussian model (same sigma in x and y)
    (x,y)=xy
    g2D=a * np.exp(-0.5*((x-x0)/sx)**2 - 0.5*((y-y0)/sx)**2) + c
    return np.ravel(g2D)

# Define the Cramer bound function to estimate parameters 
# To be improved
def cramer_bound(x, sx, sy):
    return np.sqrt(8*np.log(2)) * np.sqrt(sx**2 + sy**2) / x

def divide_coordinates(x,y, width, height):
    # Divide coordinates into 4 quadrants based on image size
    quadrant =np.zeros(len(x))
    
    for i in range(len(x)):
        # Extract the sub-image around the peak
        coord=x[i],y[i]
        if x[i] <= width/2 and y[i] <= height/2:
            quadrant[i]=1
        elif x[i] > width/2 and y[i] <= height/2:
            quadrant[i]=2
        elif x[i] <= width/2 and y[i] > height/2:
            quadrant[i]=3
        else:
            quadrant[i]=4
            
    return quadrant

def fit2Dspot(x,y,img_arr,window_size,PSFmodel,flagplot):
    # Fit each detected peak with selected PSF model and extract fit parameters

    # Loop over each peak and fit a 2D Gaussian
    results = []
    Xres=np.zeros(len(x))
    Yres=np.zeros(len(x))
    intenMat= np.zeros(len(x))
    amplMat= np.zeros(len(x))
    resMat= np.zeros(len(x))
    sigmaX=np.zeros(len(x))
    sigmaY=np.zeros(len(x))
    
    counter=0
    for i in range(len(x)):
        # Extract the sub-image around the peak
        xi = int(x[i])
        yi = int(y[i])
        xstart = max(xi - window_size // 2, 0)
        ystart = max(yi - window_size // 2, 0)
        xend = min(xi + window_size // 2 + 1, img_arr.shape[1])
        yend = min(yi + window_size // 2 + 1, img_arr.shape[0])
        sub_img = img_arr[ystart:yend, xstart:xend]
        
        # Define the initial guess for the Gaussian parameters
        x0_guess = window_size // 2
        y0_guess = window_size // 2
        a_guess = sub_img.max()
        sx_guess = window_size // 4
        sy_guess = window_size // 4
        c_guess = sub_img.min()
        p0 = [a_guess, x0_guess, y0_guess, sx_guess, sy_guess, c_guess]
        
        p0_1 = [a_guess, x0_guess, y0_guess, sx_guess,c_guess]
        
        xymesh = np.meshgrid(np.arange(window_size), np.arange(window_size))
        try:

            if PSFmodel=='2Dgaussian':
                # Fit with 2D Gaussian model using nonlinear least squares
                popt, pcov = curve_fit(gauss2d,xymesh, sub_img.ravel(), p0=p0, maxfev=600,bounds=(0, [np.inf,window_size, window_size,np.inf, np.inf,np.inf]))
                
                # Calculate the corrected position, integrated intensity, amplitude, and resolution
                x0_corr = xi + popt[1] - window_size // 2
                y0_corr = yi + popt[2] - window_size // 2
                inten = 2*popt[0] * np.pi * popt[3] * popt[4]
                ampl = popt[0]
                res_x = cramer_bound(popt[3], popt[3], popt[4])
                res_y = cramer_bound(popt[4], popt[3], popt[4])
                res = (res_x + res_y) / 2
                Xres[counter]= x0_corr
                Yres[counter]= y0_corr
                intenMat[counter]= inten
                amplMat[counter]= ampl
                resMat[counter]= res
                sigmaX[counter]=popt[3]
                sigmaY[counter]=popt[4]
                

            elif PSFmodel=='1Dgaussian':
                # Fit with simplified 1D (isotropic) Gaussian model
                popt, pcov = curve_fit(gauss1d,xymesh, sub_img.ravel(), p0=p0_1, maxfev=600,bounds=(0, [np.inf,window_size, window_size,np.inf,np.inf]))
                
                # Calculate the corrected position, integrated intensity, amplitude, and resolution
                x0_corr = xi + popt[1] - window_size // 2
                y0_corr = yi + popt[2] - window_size // 2 
                inten = 2*popt[0] * np.pi * popt[3] * popt[3]
                ampl = popt[0]
                res_x = cramer_bound(popt[3], popt[3], popt[3])
                res_y=res_x
                res = res_x
                Xres[counter]= x0_corr
                Yres[counter]= y0_corr
                intenMat[counter]= inten
                amplMat[counter]= ampl
                resMat[counter]= res
                sigmaX[counter]=popt[3]
                sigmaY[counter]=popt[3]

            elif PSFmodel=='intwindow':
                # Compute intensity by integrating a window, use local median as background estimate

                # Define the border width
                border_width = 1
                
                # Get the pixels in the border of the image
                border_pixels = np.concatenate((sub_img[:border_width, :],      # Top border
                                                sub_img[-border_width:, :],     # Bottom border
                                                sub_img[:, :border_width],      # Left border
                                                sub_img[:, -border_width:]),    # Right border
                                            axis=None)

                # Calculate the median of the border pixels
                median_border = np.median(border_pixels)

                # Calculate the corrected position, integrated intensity, amplitude, and resolution
                x0_corr = xi
                y0_corr = yi
                inten = sub_img.sum()-median_border*sub_img.size

                if inten<0:
                    inten=1e-6

                ampl =  sub_img.max()-median_border

                if ampl<0:
                    ampl=1e-6

                res_x = 0
                res_y=0
                res = 0
                Xres[counter]= x0_corr
                Yres[counter]= y0_corr
                intenMat[counter]= inten
                amplMat[counter]= ampl
                resMat[counter]= res
                sigmaX[counter]=0
                sigmaY[counter]=0

            else:
                print("Not implemented method")

            
            # Add the results to the list
            results.append((x0_corr, y0_corr, inten, ampl, res))
            counter=counter+1

        except:
            pass
    
    Xres=Xres[0:counter-1]
    Yres=Yres[0:counter-1]
    intenMat= intenMat[0:counter-1]
    amplMat= amplMat[0:counter-1]
    resMat= resMat[0:counter-1]

    if flagplot:
        # Print the results
        print('Results:')
        print('X0_corr\tY0_corr\tInten\tAmpl\tRes')
        for r in results:
            print('{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(r[0], r[1], r[2], r[3], r[4]))

        # Plot the image and the identified peaks
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(img_arr, cmap='gray')
        ax.scatter(Xres, Yres, c='r', marker='x', s=100)
        ax.scatter(x, y, c='b', marker='x', s=100)
        ax.set_title('Check fitted')
        ax.axis('off')
        plt.show()

    return results, Xres, Yres, intenMat, amplMat, resMat,sigmaX,sigmaY

def find_closest_spot(a, b_list, threshold):
    # Find closest point in b_list to point a, within a threshold
    closest_spot = None
    closest_distance = math.inf
    for b in b_list:
        distance = math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
        if distance < closest_distance:
            closest_spot = b
            closest_distance = distance
    if closest_distance <= threshold:
        return closest_spot, closest_distance
    else:
        return None, None

def find_matched_spots(xA,yA,xB,yB,threshold,img_arr,dxB,dyB,flagPlot):
    # Match spots from list A and B based on nearest-neighbor within a threshold
    a_list = list(zip(xA, yA))
    b_list = list(zip(xB+dxB, yB+dyB))
    a_matched = []
    b_matched = []
    a_positions = []
    b_positions = []
    distances = []
    for i, a in enumerate(a_list):
        closest_spot, closest_distance = find_closest_spot(a, b_list, threshold)
        if closest_spot is not None and closest_spot not in b_matched:
            a_matched.append(a)
            b_matched.append(closest_spot)
            a_positions.append(i)
            b_positions.append(b_list.index(closest_spot))
            distances.append(closest_distance)
    df = pd.DataFrame({'positionA_matched': a_positions, 'xA_matched': [a[0] for a in a_matched], 'yA_matched': [a[1] for a in a_matched], 'positionB_matched': b_positions, 'xB_matched': [b[0] for b in b_matched], 'yB_matched': [b[1] for b in b_matched], 'distance': distances})

    num_spots=df.positionA_matched.size
    rgb_range=np.linspace(0,1,num_spots)

    rgb_vals=plt.cm.jet(rgb_range)

    if flagPlot:
        # Plot the image and the identified peaks
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(img_arr, cmap='gray')
        ax.scatter(xA[df.positionA_matched], yA[df.positionA_matched], c=rgb_vals, marker='x', s=100)
        ax.scatter(xB[df.positionB_matched], yB[df.positionB_matched], c=rgb_vals, marker='x', s=100)
        ax.set_title('Identified couple')
        ax.axis('off')
        plt.show()

    return df

def fitting4polar(img_arr,thresh,min_distance,window_size,PSFmodel,flagplot):
    # Main function to process an image and perform multi-PSF fitting across quadrants
   
    width, height=img_arr.shape

    # Detect peaks in image
    x,y=getpeaksfromMax(img_arr,thresh,min_distance,window_size)

    # Optionally show detected peaks
    if flagplot:
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(img_arr, cmap='gray')
        ax.scatter(x, y, c='g', marker='x', s=100)
        ax.set_title('Identified peaks')
        ax.axis('off')
        plt.show()

    
    # Split peaks by quadrant and apply fitting separately

    quadrant=divide_coordinates(x,y, width, height)
    results1, xA, yA, intenMat1, amplMat1, resMat1,sigmaX1,sigmaY1=fit2Dspot(x[quadrant==1],y[quadrant==1],img_arr,window_size,PSFmodel,flagplot)
    results2, xB, yB, intenMat2, amplMat2, resMat2,sigmaX2,sigmaY2=fit2Dspot(x[quadrant==2],y[quadrant==2],img_arr,window_size,PSFmodel,flagplot)
    results3, xC, yC, intenMat3, amplMat3, resMat3,sigmaX3,sigmaY3=fit2Dspot(x[quadrant==3],y[quadrant==3],img_arr,window_size,PSFmodel,flagplot)
    results4, xD, yD, intenMat4, amplMat4, resMat4,sigmaX4,sigmaY4=fit2Dspot(x[quadrant==4],y[quadrant==4],img_arr,window_size,PSFmodel,flagplot)
    
    # Compile results into output dictionary

    output_dict = {}
    output_dict['x']=x
    output_dict['y']=y
    
    output_dict['results1']=results1
    output_dict['xA']=xA
    output_dict['yA']=yA
    output_dict['intenMat1']=intenMat1
    output_dict['amplMat1']=amplMat1
    output_dict['resMat1']=resMat1
    output_dict['sigmaX1']=sigmaX1
    output_dict['sigmaY1']=sigmaY1

    output_dict['results2']=results2
    output_dict['xB']=xB
    output_dict['yB']=yB
    output_dict['intenMat2']=intenMat2
    output_dict['amplMat2']=amplMat2
    output_dict['resMat2']=resMat2
    output_dict['sigmaX2']=sigmaX2
    output_dict['sigmaY2']=sigmaY2

    output_dict['results3']=results3
    output_dict['xC']=xC
    output_dict['yC']=yC
    output_dict['intenMat3']=intenMat3
    output_dict['amplMat3']=amplMat3
    output_dict['resMat3']=resMat3
    output_dict['sigmaX3']=sigmaX3
    output_dict['sigmaY3']=sigmaY3

    output_dict['results4']=results4
    output_dict['xD']=xD
    output_dict['yD']=yD
    output_dict['intenMat4']=intenMat4
    output_dict['amplMat4']=amplMat4
    output_dict['resMat4']=resMat4
    output_dict['sigmaX4']=sigmaX4
    output_dict['sigmaY4']=sigmaY4

    return output_dict

def registration4polar(flagplot,threshDist,img_arr,output_dict,deltaij):
    
    # Unpack detection results and attributes for each quadrant (A-D)
    results1 = output_dict['results1']
    xA = output_dict['xA']
    yA = output_dict['yA']
    intenMat1 = output_dict['intenMat1']
    amplMat1 = output_dict['amplMat1']
    resMat1 = output_dict['resMat1']
    sigmaX1=output_dict['sigmaX1']
    sigmaY1=output_dict['sigmaY1']

    results2 = output_dict['results2']
    xB = output_dict['xB']
    yB = output_dict['yB']
    intenMat2 = output_dict['intenMat2']
    amplMat2 = output_dict['amplMat2']
    resMat2 = output_dict['resMat2']
    sigmaX2=output_dict['sigmaX2']
    sigmaY2=output_dict['sigmaY2']

    results3 = output_dict['results3']
    xC = output_dict['xC']
    yC = output_dict['yC']
    intenMat3 = output_dict['intenMat3']
    amplMat3 = output_dict['amplMat3']
    resMat3 = output_dict['resMat3']
    sigmaX3=output_dict['sigmaX3']
    sigmaY3=output_dict['sigmaY3']

    results4 = output_dict['results4']
    xD = output_dict['xD']
    yD = output_dict['yD']
    intenMat4 = output_dict['intenMat4']
    amplMat4 = output_dict['amplMat4']
    resMat4 = output_dict['resMat4']
    sigmaX4=output_dict['sigmaX4']
    sigmaY4=output_dict['sigmaY4']

    # Find matched spots between A & B, A & C, A & D using spatial offset (deltaij)
    df = find_matched_spots(xA,yA,xB,yB,threshDist,img_arr,-deltaij,0,False)
    df2 = find_matched_spots(xA,yA,xC,yC,threshDist,img_arr,0,-deltaij,False)
    df3 = find_matched_spots(xA,yA,xD,yD,threshDist,img_arr,-deltaij,-deltaij,False)
    
    # Extract indices of matched spots in A
    pos1_2=df.positionA_matched.values
    pos1_3=df2.positionA_matched.values
    pos1_4=df3.positionA_matched.values
    
    # Extract corresponding matched indices in B, C, D
    pos2=df.positionB_matched.values
    pos3=df2.positionB_matched.values
    pos4=df3.positionB_matched.values
    
    # Find spots matched in all four quadrants (intersection of all matched sets)
    pos4quadrant0=np.intersect1d(pos1_2,np.intersect1d(pos1_3,pos1_4))
    pos4quadrant=np.sort(pos4quadrant0)

    # Initialize result array: each row will hold attributes for one matched spot across 4 quadrants
    Results=np.zeros((len(pos4quadrant),28))
    counter=0

    for i in pos4quadrant:
        index2=np.where(pos1_2==i)
        index3=np.where(pos1_3==i)
        index4=np.where(pos1_4==i)

        # Fill in spot info from quadrant A
        Results[counter][0]=xA[i]
        Results[counter][1]=yA[i]
        Results[counter][2]=intenMat1[i]
        Results[counter][3]=amplMat1[i]
        Results[counter][4]=resMat1[i]
        Results[counter][5]=sigmaX1[i]
        Results[counter][6]=sigmaY1[i]

        # Quadrant B
        Results[counter][7]=xB[pos2[index2]]
        Results[counter][8]=yB[pos2[index2]]
        Results[counter][9]=intenMat2[index2]
        Results[counter][10]=amplMat2[index2]
        Results[counter][11]=resMat2[index2]
        Results[counter][12]=sigmaX2[index2]
        Results[counter][13]=sigmaY2[index2]

        # Quadrant C
        Results[counter][14]=xC[pos3[index3]]
        Results[counter][15]=yC[pos3[index3]]
        Results[counter][16]=intenMat3[index3]
        Results[counter][17]=amplMat3[index3]
        Results[counter][18]=resMat3[index3]
        Results[counter][19]=sigmaX3[index3]
        Results[counter][20]=sigmaY3[index3]
        
        # Quadrant D
        Results[counter][21]=xD[pos4[index4]]
        Results[counter][22]=yD[pos4[index4]]
        Results[counter][23]=intenMat4[index4]
        Results[counter][24]=amplMat4[index4]
        Results[counter][25]=resMat4[index4]
        Results[counter][26]=sigmaX4[index4]
        Results[counter][27]=sigmaY4[index4]

        counter=counter+1 

    num_spots=Results.shape[0]

    # Generate a color map for plotting
    rgb_range=np.linspace(0,1,num_spots)
    rgb_vals=plt.cm.jet(rgb_range)

    if flagplot:
        # Plot the image and the identified peaks
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(img_arr, cmap='gray')
        ax.scatter(Results[:,0], Results[:,1], c=rgb_vals, marker='o', s=10,alpha=0.5)
        ax.scatter(Results[:,7], Results[:,8], c=rgb_vals, marker='o', s=10,alpha=0.5)
        ax.scatter(Results[:,14], Results[:,15], c=rgb_vals, marker='o', s=10,alpha=0.5)
        ax.scatter(Results[:,21], Results[:,22], c=rgb_vals, marker='o', s=10,alpha=0.5)
        ax.set_title('Identified couple')
        ax.axis('off')
        plt.show()
    
    return Results

def Fcn_dg_4x4_modif_3D_07122022(I0, I45, I90, I135,K):
    # Convert inputs to numpy arrays
    I_0 = np.array(I0)
    I_90 = np.array(I90)
    I_45 = np.array(I45)
    I_135 = np.array(I135)

    # Inverse of the k matrix
    K_inv = np.linalg.inv(K)

    # Initialize outputs
    rho_all = np.zeros(len(I_0))
    eta_all = np.zeros(len(I_0))
    delta_all = np.zeros(len(I_0))

    for i in range(len(I_0)):
        M = np.dot(K_inv, np.array([I_0[i], I_90[i], I_45[i], I_135[i]]))

        A2 = M[0] + M[1] + M[2]

        Pxy = (M[0] - M[1]) / A2
        Puv = 2 * M[3] / A2
        Pz = M[2] / A2

        # Ensure real values for further computation
        Puv = np.real(Puv)
        Pxy = np.real(Pxy)
        Pz = np.real(Pz)

        rho_all[i] = 0.5 * np.arctan2(Puv, Pxy)

        lambda_3 = Pz + np.sqrt(Puv**2 + Pxy**2)
        lambda_val = (1 - lambda_3) / 2

        try:
            with np.errstate(invalid='ignore'):
                val1=0.5 * np.abs(-1 + np.sqrt(12 * lambda_3 - 3))
            with np.errstate(invalid='ignore'):
                delta_all[i] = 2 * np.arccos(val1)
        except:
            delta_all[i] = np.NaN

        try:
            with np.errstate(invalid='ignore'):
                val2=np.sqrt((Pz - lambda_val) / (1 - 3 * lambda_val))
            with np.errstate(invalid='ignore'):
                eta_all[i] = np.arccos(val2)
        except:
            eta_all[i] = np.NaN

    # Wrap rho values to [0, π]
    rho_all=np.where(rho_all<0,rho_all+np.pi,rho_all)

    return delta_all, rho_all, eta_all
