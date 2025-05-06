
# 4polar3D_SMOLM_simulation

A simulation framework for 3D polarized single-molecule localization microscopy (SMOLM), designed to analyze and compare fitting methods for polarized imaging data.


## Overview

This repository provides tools to evaluate 3D polarized SMOLM data. It includes scripts for batch processing, core computational functions, and a Jupyter notebook for comparing different fitting approaches.

## Features

- Batch Processing: Automate the simulation and analysis of multiple datasets using runBatch.py.

- Fitting Comparison: Evaluate different fitting methods with the ComparingFitting.ipynb notebook.

- Modular Core Functions: Reusable functions located in the core/ directory for simulation and analysis tasks.

## Installation
### Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.
- Python 3.8 or higher

### Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/CessVala/4polar3D_SMOLM_simulation.git
   cd 4polar3D_SMOLM_simulation
   ```

2. **Create and activate the Conda environment:**

   ```bash
   conda create -n smolm_env python=3.8
   conda activate smolm_env
   ```

3. **Create and activate the Conda environment:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage 

### Explore fitting comparisons.

Open the Jupyter Notebook.

```bash
jupyter notebook ComparingFitting.ipynb
```
    

### Run in batch.

```bash
python runBatch.py
```    

## License

This project is licensed under the BSD-3-Clause License. See the [LICENSE](https://github.com/CessVala/4polar3D_SMOLM_simulation/blob/main/LICENSE) file for details.
