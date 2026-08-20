# HiSpecAMP: Antimicrobial Peptide Activity Prediction Based on Hierarchical Species Representation Learning

This repository contains the implementation of **HiSpecAMP**, a species-aware framework for antimicrobial peptide (AMP) activity prediction. HiSpecAMP is designed to predict the minimum inhibitory concentration (MIC) of antimicrobial peptides against specific microbial species by explicitly modeling pathogen-specific information and peptide–species interactions.

## Overview

Accurate prediction of antimicrobial peptide activity requires not only modeling peptide sequence characteristics but also considering the target microbial species. Existing AMP prediction methods often focus primarily on peptide representations while providing limited modeling of species-specific effects.

To address this limitation, HiSpecAMP introduces a **hierarchical species representation learning** framework that learns biologically meaningful representations of microbial species from their taxonomic relationships. The learned species representations are subsequently incorporated into MIC prediction through a **Feature-wise Linear Modulation (FiLM)** module, enabling species-conditioned modulation of peptide representations.

The major components of HiSpecAMP include:

* **ESM-2-based peptide representation** for extracting sequence-level peptide features.
* **Hierarchical species representation learning** to encode microbial taxonomic information.
* **Species-aware Feature-wise Linear Modulation (FiLM)** to dynamically modulate peptide representations according to the target species.
* **Peptide–species interaction modeling** to capture complementary and multiplicative interactions between peptide and species representations.
* **Ranking consistency constraints** to improve the consistency of predicted antimicrobial activity rankings.
* Evaluation under **WarmStart, Species-ColdStart, and Peptide-ColdStart** settings to assess both conventional prediction performance and generalization to unseen species or peptides.

The overall objective of HiSpecAMP is to provide accurate and biologically informed MIC prediction for AMP–species interactions.

## Model Architecture

The HiSpecAMP framework consists of two major stages:

### 1. Hierarchical Species Representation Learning

Microbial species are represented using taxonomic information and pretrained with hierarchical supervision. The resulting species embeddings capture relationships among microorganisms at multiple taxonomic levels and provide species-aware information for downstream MIC prediction.

### 2. Species-aware MIC Prediction

Given an antimicrobial peptide and a target microbial species:

1. The peptide sequence is encoded using a pretrained **ESM-2** model.
2. The target species is represented using the pretrained species embedding.
3. The species representation generates FiLM parameters, including scaling and shifting factors.
4. The peptide representation is dynamically modulated according to the target species.
5. The modulated peptide representation and species representation are combined with interaction features.
6. The resulting peptide–species representation is passed to the prediction network to estimate MIC.

The model is trained using a combination of regression and ranking-related objectives.

## Environment

The experiments were conducted using the following software environment:

* Python
* PyTorch 2.5.1
* CUDA 12.6
* Transformers
* ESM-2
* NumPy
* Pandas
* Scikit-learn
* SciPy
* PyYAML
* tqdm
* Biopython

A GPU with CUDA support is recommended for model training and inference.

### Recommended Environment

```bash
conda create -n hispecamp python=3.10
conda activate hispecamp
```

Install PyTorch according to your CUDA environment. For example:

```bash
pip install torch==2.5.1
```

Then install the required Python packages:

```bash
pip install transformers numpy pandas scipy scikit-learn pyyaml tqdm biopython matplotlib
```

If a `requirements.txt` file is provided in this repository, the recommended installation method is:

```bash
pip install -r requirements.txt
```

## Data Preparation

HiSpecAMP uses peptide–species interaction data containing antimicrobial peptide sequences, target microbial species, and experimentally measured MIC values.

The processed dataset should contain, at minimum:

* Peptide sequence
* Peptide identifier
* Target species
* MIC value
* Peptide representation or the corresponding ESM-2 embedding
* Species representation for the target microorganism

The preprocessing pipeline removes duplicated peptide–species interactions and filters peptide sequences according to the predefined sequence-length criteria.

The species representation used by HiSpecAMP is generated during the hierarchical species representation learning stage and saved for downstream MIC prediction.

Place the processed data and pretrained embeddings in the corresponding paths specified by the configuration files before training.

## Running HiSpecAMP

### Step 1. Prepare peptide representations

Peptide sequences are encoded using the pretrained ESM-2 model. The resulting peptide embeddings can be saved and reused during downstream training to avoid repeatedly computing sequence representations.

The expected peptide embedding dimension used in the MIC prediction model is:

```text
ESM-2 peptide embedding: 320
```

### Step 2. Pretrain species representations

Run the hierarchical species representation learning module to obtain species embeddings.

The pretrained species embeddings are then saved for downstream MIC prediction.

The resulting embedding file should contain one representation for each target microbial species.

### Step 3. Train the MIC prediction model

After preparing the peptide and species representations, train the HiSpecAMP MIC prediction model.

The training procedure supports the evaluation settings used in the paper:

* **WarmStart**
* **Species-ColdStart**
* **Peptide-ColdStart**

The model uses species-conditioned FiLM modulation together with peptide–species interaction features for MIC prediction.

### Step 4. Model evaluation

After training, evaluate the model on the corresponding test set.

HiSpecAMP reports both regression and ranking-related performance metrics, including:

* MSE
* RMSE
* MAE
* $R^2$
* PCC
* Kendall's $\tau$

For antimicrobial activity classification analysis, MIC values are further converted into active/inactive labels according to the predefined MIC threshold.

### Step 5. External validation

The trained model can also be evaluated on independent external AMP datasets to assess cross-dataset generalization.

The external validation procedure follows the same MIC prediction protocol while keeping the trained model parameters fixed.

## Experimental Settings

To evaluate different aspects of model generalization, HiSpecAMP uses three experimental settings.

### WarmStart

Both peptides and microbial species observed during training can occur in the test set, while the specific peptide–species interactions are held out.

### Species-ColdStart

The target microbial species in the test set are unseen during model training. This setting evaluates the ability of HiSpecAMP to generalize to previously unseen species based on their learned taxonomic representations.

### Peptide-ColdStart

The peptides in the test set are unseen during training. This setting evaluates the ability of the model to generalize to novel peptide sequences.

## Reproducibility

The main training configurations, model hyperparameters, and data-processing settings are provided in the repository configuration files.

The main model configuration used in the experiments includes:

```text
Hidden dimension: 512
Learning rate: 0.0005
Batch size: 32
Number of folds: 5
Maximum epochs: 100
Early stopping patience: 8
```

The model uses fixed ESM-2 peptide representations and pretrained species embeddings during downstream MIC prediction.

## Citation

If you find HiSpecAMP useful in your research, please cite our work:

```bibtex
@article{HiSpecAMP,
  title   = {Antimicrobial Peptide Activity Prediction Based on Hierarchical Species Representation Learning},
  author  = {Zheng, Jingbo and Wang, Yongkang and Zhang, Wen and Liu, Shichao},
  journal = {},
  year    = {}
}
```

## License

This project is intended for academic and research purposes. Please refer to the repository license for details.
