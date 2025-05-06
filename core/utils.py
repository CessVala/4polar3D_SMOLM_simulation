# utils.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
from scipy.io import savemat
import scipy.io
from core import polarFitFuncv2
import os
import re
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm
from scipy.stats import circstd

def load_simulation_images(filepath):
    """
    Load an image given by `filepath`.
    Returns a NumPy array of images.
    """
    
    # Load the MAT file
    mat = scipy.io.loadmat(filepath)
    tif_array= mat['Img2x2']

    return tif_array

def process_image(i,tif_array,thresh, min_distance, window_size, PSFmodel,flagplot):
    img_arr = tif_array[i,:,:]
    return polarFitFuncv2.fitting4polar(img_arr, thresh, min_distance, window_size, PSFmodel,flagplot)


def ensure_directories(base_dir):
    """
    Ensure that Data/, Output/, and Figs/ subdirectories exist inside `base_dir`.
    Returns paths as a dict: {'data': ..., 'output': ..., 'figs': ...}
    """
    subdirs = ['Data', 'Output', 'Figs']
    paths = {}
    for sub in subdirs:
        path = os.path.join(base_dir, sub)+"/"
        os.makedirs(path, exist_ok=True)
        paths[sub.lower()] = path
    return paths

def extract_params_from_filename(filename):
    """
    Extract delta, rho, eta values from filename string.
    Returns a dictionary with integer values.
    """
    params = {}
    for key in ['delta', 'rho', 'eta']:
        match = re.search(rf'{key}_(\d+)', filename)
        if match:
            params[key] = int(match.group(1))
        else:
            raise ValueError(f"{key} not found in filename: {filename}")
    return params

def show_image(image, title='Image', figsize=(8, 8), cmap='gray'):
    """
    Display a single image using matplotlib.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    ax.axis('off')
    plt.show()

def get_cpu_info():
    """
    Returns the number of available CPU cores and prints it.
    """
    num_cores = cpu_count()
    print("Number of CPU cores:", num_cores)
    return num_cores

def run_fitting_parallel(tif_array, num_imgs, thresh, min_distance, window_size,num_cores, flagplot=False):
    """
    Run spot fitting in parallel using multiple PSF models.

    This function fits each image in the stack using three different point spread function (PSF) models:
    2D Gaussian, 1D Gaussian, and integration over a window. The computation is parallelized 
    using joblib to speed up processing.

    Parameters:
        tif_array (ndarray): Input 3D image stack of shape (N, H, W), where N is the number of frames.
        num_imgs (int): Number of frames to process from tif_array.
        thresh (int): Intensity threshold to detect candidate spots.
        min_distance (int): Minimum pixel distance between detected peaks.
        window_size (int): Window size (in pixels) used for local fitting.
        num_cores (int): Number of CPU cores to use for parallel processing.
        flagplot (bool, optional): If True, generate debug plots during fitting. Default is False.

    Returns:
        tuple of dict: 
            - output_Fit4_2G: Results from 2D Gaussian fitting.
            - output_Fit4_1G: Results from 1D Gaussian fitting.
            - output_Fit4_intWn: Results from integration window-based fitting.
    """
    
    output_Fit4_1G = {}
    output_Fit4_2G = {}
    output_Fit4_intWn = {}
    
    # Store raw results from parallel computation
    results2G={}
    results1G={}
    resultsintWn={}

    print('Parallel processing')
    
    # --- 2D Gaussian fitting ---
    PSFmodel='2Dgaussian'
    results2G = Parallel(n_jobs=num_cores)(delayed(process_image)(i,tif_array,thresh, min_distance, window_size, PSFmodel,flagplot) for i in tqdm(range(num_imgs)))
    
    # --- 1D Gaussian fitting ---
    PSFmodel='1Dgaussian'
    results1G = Parallel(n_jobs=num_cores)(delayed(process_image)(i,tif_array,thresh, min_distance, window_size, PSFmodel,flagplot) for i in tqdm(range(num_imgs)))

    # --- Integration window fitting ---
    PSFmodel='intwindow'
    resultsintWn = Parallel(n_jobs=num_cores)(delayed(process_image)(i,tif_array,thresh, min_distance, window_size, PSFmodel,flagplot) for i in tqdm(range(num_imgs)))

    # Convert results to indexed dictionaries for consistency
    for i, result in enumerate(results1G):
        output_Fit4_1G[i] = result
    
    for i, result in enumerate(results2G):
        output_Fit4_2G[i] = result
    
    for i, result in enumerate(resultsintWn):
        output_Fit4_intWn[i] = result

    return output_Fit4_2G, output_Fit4_1G, output_Fit4_intWn

def run_fitting_sequential(tif_array, num_imgs, thresh, min_distance, window_size, flagplot=False):
    """
    Run fitting sequentially for each PSF model.

    Returns:
        Tuple of three dicts with results for 2D Gaussian, 1D Gaussian, and integration window.
    """

    output_2G, output_1G, output_intWn = {}, {}, {}

    for i in tqdm(range(num_imgs)):
        img = tif_array[i, :, :]
        output_2G[i] = polarFitFuncv2.fitting4polar(img, thresh, min_distance, window_size, '2Dgaussian', flagplot)
        output_1G[i] = polarFitFuncv2.fitting4polar(img, thresh, min_distance, window_size, '1Dgaussian', flagplot)
        output_intWn[i] = polarFitFuncv2.fitting4polar(img, thresh, min_distance, window_size, 'intwindow', flagplot)

    return output_2G, output_1G, output_intWn

def run_registration(fitting_results_dict, img_arr, deltaij, threshDist, flagplot=False):
    """
    Run registration4polar on a set of fitting results and concatenate outputs.

    Parameters:
        fitting_results_dict (dict): Dictionary with frame index → fitting results.
        img_arr (ndarray): Full 2D+t image array used during registration.
        deltaij (float or ndarray): Offset parameter for registration.
        threshDist (float): Distance threshold.
        flagplot (bool): If True, enable plotting in registration.

    Returns:
        ndarray: Concatenated registration results for all frames.
    """
    results = []
    for i in range(len(fitting_results_dict)):
        registration = polarFitFuncv2.registration4polar(
            flagplot, threshDist, img_arr, fitting_results_dict[i], deltaij
        )
        results.append(registration)

    return np.concatenate(results, axis=0)

def compute_dg_parameters(results_array, K):
    """
    Compute delta, rho, and eta from registration results using custom fitting function.

    Parameters:
        results_array (ndarray): Output from run_registration
        K: K matrix

    Returns:
        tuple of ndarrays: (delta_all, rho_all, eta_all)
    """
    return polarFitFuncv2.Fcn_dg_4x4_modif_3D_07122022(
        results_array[:, 2],   
        results_array[:, 16],  
        results_array[:, 9],   
        results_array[:, 23],  
        K
    )

def plot_spots_on_image(img_arr, results, rgb_vals, ax):
    """
    Plots identified spots on the given image.
    
    Parameters:
        img_arr (ndarray): The image on which to plot the spots.
        results (ndarray): Array containing spot coordinates and data.
        rgb_vals (ndarray): Color values corresponding to each spot.
        ax (matplotlib axis): Axis to plot on.
    """
    # Plot the image
    ax.imshow(img_arr, cmap='gray')
    
    # Plot the identified spots at four positions (indexes for different spot positions)
    for idx in [0, 7, 14, 21]:
        ax.scatter(results[:, idx], results[:, idx + 1], c=rgb_vals, marker='o', s=5, alpha=0.4)

    ax.axis('off')


def plotfinalfigure(img_arr, results, modeltxt, fig_dir, filename):
    """
    Plot the final figure showing identified spots for a given PSF model.

    Parameters:
        img_arr (ndarray): The image array to display.
        results (ndarray): The results containing spot coordinates.
        modeltxt (str): The PSF model used for fitting.
        fig_dir (str): Directory to save the figure.
        filename (str): Filename to use for saving the figure.
    """
    num_spots = results.shape[0]
    rgb_range = np.linspace(0, 1, num_spots)
    rgb_vals = plt.cm.jet(rgb_range)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the spots on the image
    plot_spots_on_image(img_arr, results, rgb_vals, ax)
    
    # Set title and display the plot
    ax.set_title(f"Identified couple - PSF Model: {modeltxt}", y=1.05)
    plt.show()

    # Save the figure with appropriate filename
    base_name = os.path.splitext(filename)[0]
    figname = f"{base_name}_{modeltxt}_IdentifiedCouples.png"
    fig.savefig(os.path.join(fig_dir, figname))

def plot_orientation_distributions(delta_teo, rho_teo, eta_teo,
                                   delta_all, rho_all, eta_all,
                                   PSFmodel, fig_dir, filename):
    """Plot histograms of δ, ρ, η parameters with theoretical reference lines and metrics."""
    delta_deg = delta_all * 180 / np.pi
    rho_deg   = rho_all * 180 / np.pi
    eta_deg   = eta_all * 180 / np.pi

    mean_delta = np.nanmean(delta_deg)
    mean_rho   = np.nanmean(rho_deg)
    mean_eta   = np.nanmean(eta_deg)

    std_delta = circstd(delta_deg, high=180, low=0, nan_policy='omit')
    std_rho   = circstd(rho_deg, high=180, low=0, nan_policy='omit')
    std_eta   = circstd(eta_deg, high=180, low=0, nan_policy='omit')

    RMSE_delta =np.sqrt(np.nanmean([(delta_teo - real_value)**2 for real_value in delta_all*180/np.pi]))
    RMSE_rho =np.sqrt(np.nanmean([(rho_teo - real_value)**2 for real_value in rho_all*180/np.pi]))
    RMSE_eta =np.sqrt(np.nanmean([(eta_teo - real_value)**2 for real_value in eta_all*180/np.pi]))
    
    eta_nan = np.count_nonzero(np.isnan(eta_all))
    rho_nan = np.count_nonzero(np.isnan(rho_all))
    delta_nan = np.count_nonzero(np.isnan(delta_all))

    xticks = np.linspace(0, 180, 5)
    fig, ax = plt.subplots(1, 3, figsize=(16, 3))

    ax[0].hist(delta_deg, bins=range(0, 190, 5))
    ax[0].set_title(
        f'δ [teo= {delta_teo:.1f}°, mean= {mean_delta:.1f}°, ±{std_delta:.1f}°, RMSE= {RMSE_delta:.1f}°]',
        fontsize=8)
    ax[0].set_xticks(xticks)
    ax[0].axvline(delta_teo, color='red', linestyle='--')

    ax[1].hist(rho_deg, bins=range(0, 190, 5))
    ax[1].set_title(
        f'ρ [teo= {rho_teo:.1f}°, mean= {mean_rho:.1f}°, ±{std_rho:.1f}°, RMSE= {RMSE_rho:.1f}°]',
        fontsize=8)
    ax[1].set_xticks(xticks)
    ax[1].axvline(rho_teo, color='red', linestyle='--')

    ax[2].hist(eta_deg, bins=range(0, 190, 5))
    ax[2].set_title(
        f'η [teo= {eta_teo:.1f}°, mean= {mean_eta:.1f}°, ±{std_eta:.1f}°, RMSE= {RMSE_eta:.1f}°]',
        fontsize=8)
    ax[2].set_xticks(xticks)
    ax[2].axvline(eta_teo, color='red', linestyle='--')

    fig.suptitle(f'PSF Model: {PSFmodel}', y=1.05)
    plt.tight_layout()
    plt.show()

    # Save the figure
    base_name = os.path.splitext(filename)[0]
    figname = f"{base_name}_{PSFmodel}.png"
    fig.savefig(os.path.join(fig_dir, figname), bbox_inches='tight')


def compute_total_intensity(results):
    return results[:,2] + results[:,16] + results[:,9] + results[:,23]


def save_fitting_results_mat(
    Results1G, Results2G, ResultsintWn,
    delta_all1G, rho_all1G, eta_all1G,
    delta_all2G, rho_all2G, eta_all2G,
    delta_allintWn, rho_allintWn, eta_allintWn,
    delta_teo, rho_teo, eta_teo,
    thresh, min_distance, window_size, threshDist,
    filename, output_dir
):
   
    def concatenate_results(results, delta, rho, eta):
        total_intensity = compute_total_intensity(results)
        return np.concatenate((
            results,
            np.expand_dims(delta, axis=1),
            np.expand_dims(rho, axis=1),
            np.expand_dims(eta, axis=1),
            np.expand_dims(total_intensity, axis=1)
        ), axis=1)

    Results1Gmat = concatenate_results(Results1G, delta_all1G, rho_all1G, eta_all1G)
    Results2Gmat = concatenate_results(Results2G, delta_all2G, rho_all2G, eta_all2G)
    ResultsintWnGmat = concatenate_results(ResultsintWn, delta_allintWn, rho_allintWn, eta_allintWn)

    base_name = os.path.splitext(filename)[0]
    mat_file = base_name + '_Results.mat'

    data = {
        'Results1G': Results1Gmat,
        'Results2G': Results2Gmat,
        'ResultsintWn': ResultsintWnGmat,
        'delta_teo': delta_teo,
        'rho_teo': rho_teo,
        'eta_teo': eta_teo,
        'thresh': thresh,
        'min_distance': min_distance,
        'window_size': window_size,
        'threshDist': threshDist,
        'filename': filename
    }

    savemat(os.path.join(output_dir, mat_file), data)

def plot_intensity_histograms(totalInt1G, totalInt2G, totalIntintWn, fig_dir, filename):
    maxVal = 15000
    xticks = np.linspace(0, maxVal, 10)
    binsInt = range(0, maxVal, 100)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 3))

    ax[0].hist(totalInt1G, bins=binsInt)
    ax[0].set_title('1D Gaussian')

    ax[1].hist(totalInt2G, bins=binsInt)
    ax[1].set_title('2D Gaussian')

    ax[2].hist(totalIntintWn, bins=binsInt)
    ax[2].set_title('Integrated window')

    fig.suptitle("Total Intensity", y=1.05)

    plt.show()
    base_name = os.path.splitext(filename)[0]
    figname = base_name + '_TotalIntensity.png'
    fig.savefig(fig_dir + figname, bbox_inches='tight')


