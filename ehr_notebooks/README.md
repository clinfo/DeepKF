# Cancer EHR Data Analysis – UMAP, PCA, VAE, and Clustering

This repository contains a collection of Jupyter notebooks for analyzing cancer treatment data, patient timelines, and drug-specific effects.
The analyses employ **UMAP, PCA, VAE, clustering, and visualization techniques** to uncover hidden patterns in patient data, focusing on treatment outcomes, drug categories, and survival.

This script is for the EHR data used in the paper. To obtain the complete data, please contact the paper's authors.

## Key Features

* **Patient-level analysis**: Patient-level time-series analysis and state-space modeling.
* **Clustering**: K-means to construct clusters of patient states.
* **Dimensionality reduction/Visualization**: PCA, VAE, and linear projections and UMAP visualization.
* **Data Sources**: Patient IDs, treatment logs, laboratory/vital sign data.


## Contents

### 1. **all\_data\_ver.ipynb**

* **Purpose**: Prepares the complete dataset for downstream analyses.
* **Processing**:

  * Loads laboratory, vital signs, and patient metadata from pickle files.
  * Handles missing values and constructs time-series data for each patient (`all_data_mask_df_merge`).
  * Assigns time steps to each patient record for sequential analysis.

---

### 2. **for\_other\_anti\_agent\_analysis\_for\_UMAP.ipynb**

* **Purpose**: UMAP-based analysis of anti-agent treatments.
* **Processing**:

  * Loads patient IDs for cancer-related deaths and survivors.
  * Extracts subsets of patients treated with specific anti-agents.
  * Applies **UMAP dimensionality reduction** to visualize treatment patterns.

---

### 3. **for\_other\_anti\_cancer\_agent.ipynb**

* **Purpose**: Constructs patient lists by drug category (anti-cancer agents).
* **Processing**:

  * Loads patient treatment history.
  * Groups patients by administered drug type.
  * Prepares input for survival analysis and embedding models.

---

### 4. **for\_py\_all\_cancer\_inj\_notage\_analysis.py.ipynb**

* **Purpose**: PCA-based analysis of cancer patients (excluding age).
* **Processing**:

  * Loads precomputed embeddings of patient trajectories.
  * Applies **PCA** to explore variance structure.
  * Provides summary statistics (e.g., number of patients).

---

### 5. **for\_py\_all\_cancer\_inj\_notage\_analysis\_for\_UMAP\_all\_data.ipynb**

* **Purpose**: UMAP analysis with random sampling (5 splits).
* **Processing**:

  * Loads patient embeddings (`z_params` from VAE latent states).
  * Splits patients into random subsets to validate reproducibility.
  * Runs UMAP for visualization of temporal trajectories.

---

### 6. **for\_py\_all\_cancer\_p\_or\_i\_notage\_analysis\_for\_UMAP.ipynb**

* **Purpose**: UMAP with random 10-split sampling on patient data.
* **Processing**:

  * Loads patient ID lists and time-step data.
  * Extracts latent state embeddings from trained models.
  * Applies UMAP for **high-dimensional trajectory visualization**.

---

### 7. **linear\_result.ipynb**

* **Purpose**: Linear comparison of UMAP embeddings.
* **Processing**:

  * Uses the same patient ID and latent state pipeline as above.
  * Compares linear projection results with UMAP embeddings.
  * Assesses the stability of latent representations.

---

### 8. **p\_for\_clustering\_analysis\_kmeans.ipynb**

* **Purpose**: Patient clustering via K-means.
* **Processing**:

  * Loads patient embeddings.
  * Applies **K-means clustering** to group patients by treatment response.
  * Prepares results for comparison with survival and drug classes.

---

### 9. **PCA\_result.ipynb**

* **Purpose**: PCA + UMAP hybrid analysis.
* **Processing**:

  * Reads PCA-transformed patient embeddings.
  * Runs UMAP on PCA outputs for enhanced visualization.
  * Exports results to `pca_umap_result.csv`.

---

### 10. **staked\_bar.ipynb**

* **Purpose**: Visualization of drug usage and survival with stacked bar plots.
* **Processing**:

  * Loads merged patient data.
  * Performs random sampling to balance patient groups.
  * Uses **stacked bar charts** to illustrate drug usage frequency and outcomes.

---

### 11. **VAE\_result.ipynb**

* **Purpose**: Visualization and analysis of latent variables obtained from a **Variational Autoencoder (VAE)** trained on cancer patient data.
* **Processing**:

  * Imports numerical, visualization, and deep learning libraries (`NumPy`, `Matplotlib`, `Torch`).
  * Loads VAE latent state outputs (`df_vae_dim16_zs_*`) from precomputed CSV files.
  * Applies **UMAP dimensionality reduction** on concatenated latent embeddings to project patients into a 2D space.
  * Exports results to `vae_mew_umap_result_1e3.csv` for further visualization.

---


