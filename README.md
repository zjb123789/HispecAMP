# HiSpecAMP: Antimicrobial Peptide Activity Prediction Based on Hierarchical Species Representation Learning

This repository contains the implementation of **HiSpecAMP**, a species-aware framework for antimicrobial peptide (AMP) activity prediction. HiSpecAMP is designed to predict the minimum inhibitory concentration (MIC) of antimicrobial peptides against specific microbial species by explicitly modeling species-specific information and peptide–species interactions.

## Evaluation Scenarios and Data Splitting

HiSpecAMP is evaluated under three different scenarios to assess its generalization ability:

* **Warmstart:** Both peptides and species in the test set have been observed during training, while the peptide–species interactions are unseen.
* **Species-coldstart:** The species in the test set are completely unseen during training, evaluating generalization to new microbial species.
* **Peptide-coldstart:** The peptides in the test set are completely unseen during training, evaluating generalization to new peptide sequences.

## Data Preparation

HiSpecAMP uses peptide–species interaction data containing antimicrobial peptide sequences, target microbial species, and experimentally measured MIC values.

## Running HiSpecAMP

### Environment

```bash
pip install -r requirements.txt
```

### Step 1. Prepare peptide and species representations

Peptide sequences are encoded using the pretrained ESM-2 (esm2_t6_8M_UR50D) model.

Run the hierarchical species representation learning module to obtain species embeddings.

### Step 2. Train the MIC prediction model

After preparing the peptide and species representations, train the HiSpecAMP MIC prediction model.

The training procedure supports three evaluation settings used in the paper: *Warmstart*, *Species-coldstart*, *Peptide-coldstart*.

Run the corresponding train.py script in each setting's folder:

```bash
python train.py
```

## License

This project is intended for academic and research purposes. Please refer to the repository license for details.
