import papermill as pm
import pandas as pd

#filename='Image_ITtheory_10000_r_1p7_w_17_wGen_51_bg_10_delta_001_rho_030_eta_000_lambda_0.6_f_0_z_0_All_PoissonNoise.mat'
dir_='D:/IHB2023/Collaborations/4polar/Simulation-3Dpolar_v3/Synthetic_images/'

df=pd.read_excel(dir_+'listnames.xlsx')

filenames=df['Filenames'].tolist()
filenames0=filenames[190:]


for filename in filenames0:    
    pm.execute_notebook(
        'D:/IHB2023/Collaborations/4polar/4polarSTORMpython/TestDifferentfittingv4.ipynb',
        output_path=None,
        parameters={'filename': filename}
    )
