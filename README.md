# HiSpecAMP: Antimicrobial Peptide Activity Prediction Based on Hierarchical Species Representation Learning

This repository contains the implementation of **HiSpecAMP**, a species-aware framework for antimicrobial peptide (AMP) activity prediction. HiSpecAMP is designed to predict the minimum inhibitory concentration (MIC) of antimicrobial peptides against specific microbial species by explicitly modeling species-specific information and peptide–species interactions.

## Evaluation Scenarios and Data Splitting

HiSpecAMP is evaluated under three different scenarios to assess its generalization ability:

* **Warmstart:** Both peptides and species in the test set have been observed during training, while the peptide–species interactions are unseen.
* **Species-coldstart:** The species in the test set are completely unseen during training, evaluating generalization to new microbial species.
* **Peptide-coldstart:** The peptides in the test set are completely unseen during training, evaluating generalization to new peptide sequences.

## Data Preparation

HiSpecAMP uses peptide–species interaction data containing antimicrobial peptide sequences, target microbial species, and experimentally measured MIC values.

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
