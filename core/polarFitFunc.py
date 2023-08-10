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
    # Read the TIFF image using PIL
    #img = Image.open('image.tif')

    # Convert the image to a NumPy array
    #img_arr = np.array(img)

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
    (x,y)=xy
    g2D=a * np.exp(-0.5*((x-x0)/sx)**2 - 0.5*((y-y0)/sy)**2) + c
    return np.ravel(g2D)

# Define the Cramer bound function to estimate parameters
def cramer_bound(x, sx, sy):
    return np.sqrt(8*np.log(2)) * np.sqrt(sx**2 + sy**2) / x

def divide_coordinates(x,y, width, height):
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

def fit2Dgaussian(x,y,img_arr,window_size,flagplot):
    # Loop over each peak and fit a 2D Gaussian
    results = []
    Xres=np.zeros(len(x))
    Yres=np.zeros(len(x))
    intenMat= np.zeros(len(x))
    amplMat= np.zeros(len(x))
    resMat= np.zeros(len(x))
    
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
        
        
        xymesh = np.meshgrid(np.arange(17), np.arange(17))
        try:
            popt, pcov = curve_fit(gauss2d,xymesh, sub_img.ravel(), p0=p0, maxfev=5000,bounds=(0, [np.inf,window_size, window_size,np.inf, np.inf,np.inf]))
            
            # Calculate the corrected position, integrated intensity, amplitude, and resolution
            x0_corr = xi + popt[1] - window_size // 2
            y0_corr = yi + popt[2] - window_size // 2
            inten = popt[0] * np.pi * popt[3] * popt[4]
            ampl = popt[0]
            res_x = cramer_bound(popt[3], popt[3], popt[4])
            res_y = cramer_bound(popt[4], popt[3], popt[4])
            res = (res_x + res_y) / 2
            Xres[counter]= x0_corr
            Yres[counter]= y0_corr
            intenMat[counter]= inten
            amplMat[counter]= ampl
            resMat[counter]= res
            
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

    return results, Xres, Yres, intenMat, amplMat, resMat

def find_closest_spot(a, b_list, threshold):
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

# Example usage
#A = [(1,2), (3,4), (5,6)]
#B = [(1,1), (9,9), (5,5), (3,4)]
#xA=[1,3,10]
#yA=[2,4,6]
#xB=[1,9,5,3]
#yB=[1,9,5,4]

def fitting4polar(img_arr,thresh,min_distance,window_size,flagplot):
    # 
    width, height=img_arr.shape

    x,y=getpeaksfromMax(img_arr,thresh,min_distance,window_size)

    if flagplot:
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(img_arr, cmap='gray')
        ax.scatter(x, y, c='g', marker='x', s=100)
        ax.set_title('Identified peaks')
        ax.axis('off')
        plt.show()

    quadrant=divide_coordinates(x,y, width, height)

    results1, xA, yA, intenMat1, amplMat1, resMat1=fit2Dgaussian(x[quadrant==1],y[quadrant==1],img_arr,window_size,flagplot)
    results2, xB, yB, intenMat2, amplMat2, resMat2=fit2Dgaussian(x[quadrant==2],y[quadrant==2],img_arr,window_size,flagplot)
    results3, xC, yC, intenMat3, amplMat3, resMat3=fit2Dgaussian(x[quadrant==3],y[quadrant==3],img_arr,window_size,flagplot)
    results4, xD, yD, intenMat4, amplMat4, resMat4=fit2Dgaussian(x[quadrant==4],y[quadrant==4],img_arr,window_size,flagplot)
    
    return x,y, results1, xA, yA, intenMat1, amplMat1, resMat1,results2, xB, yB, intenMat2, amplMat2, resMat2,results3, xC, yC, intenMat3, amplMat3, resMat3,results4, xD, yD, intenMat4, amplMat4, resMat4


def registration4polar(flagplot,threshDist,img_arr,results1, xA, yA, intenMat1, amplMat1, resMat1,results2, xB, yB, intenMat2, amplMat2, resMat2,results3, xC, yC, intenMat3, amplMat3, resMat3,results4, xD, yD, intenMat4, amplMat4,resMat4):

    df = find_matched_spots(xA,yA,xB,yB,threshDist,img_arr,-256,0,False)
    df2 = find_matched_spots(xA,yA,xC,yC,threshDist,img_arr,0,-256,False)
    df3 = find_matched_spots(xA,yA,xD,yD,threshDist,img_arr,-256,-256,False)
    
    pos1_2=df.positionA_matched.values
    pos1_3=df2.positionA_matched.values
    pos1_4=df3.positionA_matched.values
    
    pos2=df.positionB_matched.values
    pos3=df2.positionB_matched.values
    pos4=df3.positionB_matched.values
    
    pos4quadrant0=np.intersect1d(pos1_2,np.intersect1d(pos1_3,pos1_4))
    pos4quadrant=np.sort(pos4quadrant0)
    Results=np.zeros((len(pos4quadrant),20))
    counter=0

    for i in pos4quadrant:
        index2=np.where(pos1_2==i)
        index3=np.where(pos1_3==i)
        index4=np.where(pos1_4==i)

        Results[counter][0]=xA[i]
        Results[counter][1]=yA[i]
        Results[counter][2]=intenMat1[i]
        Results[counter][3]=amplMat1[i]
        Results[counter][4]=resMat1[i]

        Results[counter][5]=xB[pos2[index2]]
        Results[counter][6]=yB[pos2[index2]]
        Results[counter][7]=intenMat2[index2]
        Results[counter][8]=amplMat2[index2]
        Results[counter][9]=resMat2[index2]

        Results[counter][10]=xC[pos3[index3]]
        Results[counter][11]=yC[pos3[index3]]
        Results[counter][12]=intenMat3[index3]
        Results[counter][13]=amplMat3[index3]
        Results[counter][14]=resMat3[index3]
        
        Results[counter][15]=xD[pos4[index4]]
        Results[counter][16]=yD[pos4[index4]]
        Results[counter][17]=intenMat4[index4]
        Results[counter][18]=amplMat4[index4]
        Results[counter][19]=resMat4[index4]

        counter=counter+1 

    num_spots=Results.shape[0]

    rgb_range=np.linspace(0,1,num_spots)

    rgb_vals=plt.cm.jet(rgb_range)

    if flagplot:
        # Plot the image and the identified peaks
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(img_arr, cmap='gray')
        ax.scatter(Results[:,0], Results[:,1], c=rgb_vals, marker='x', s=100)
        ax.scatter(Results[:,5], Results[:,6], c=rgb_vals, marker='x', s=100)
        ax.scatter(Results[:,10], Results[:,11], c=rgb_vals, marker='x', s=100)
        ax.scatter(Results[:,15], Results[:,16], c=rgb_vals, marker='x', s=100)
        ax.set_title('Identified couple')
        ax.axis('off')
        plt.show()
    
    return Results

def Fcn_dg_4x4_modif_3D_07122022(I0, I45, I90, I135,K):
    

    I_0 = np.array(I0)
    I_90 = np.array(I90)
    I_45 = np.array(I45)
    I_135 = np.array(I135)

    K_inv = np.linalg.inv(K)

    rho_all = np.zeros(len(I_0))
    eta_all = np.zeros(len(I_0))
    delta_all = np.zeros(len(I_0))

    for i in range(len(I_0)):
        M = np.dot(K_inv, np.array([I_0[i], I_90[i], I_45[i], I_135[i]]))

        A2 = M[0] + M[1] + M[2]

        Pxy = (M[0] - M[1]) / A2
        Puv = 2 * M[3] / A2
        Pz = M[2] / A2

        Puv = np.real(Puv)
        Pxy = np.real(Pxy)
        Pz = np.real(Pz)

        rho_all[i] = 0.5 * np.arctan2(Puv, Pxy)

        lambda_3 = Pz + np.sqrt(Puv**2 + Pxy**2)
        lambda_val = (1 - lambda_3) / 2

        delta_all[i] = 2 * np.arccos(0.5 * np.abs(-1 + np.sqrt(12 * lambda_3 - 3)))

        eta_all[i] = np.arccos(np.sqrt((Pz - lambda_val) / (1 - 3 * lambda_val)))

    return delta_all, rho_all, eta_all


