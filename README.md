# GNN Age Prediction

Age prediction with **Graph Neural Networks (GNNs)**, developed for the manuscript:

*"Similar Minds Age Alike: An MRI Similarity Approach for Predicting Age-Related Cognitive Decline"*

---

## 📖 Overview

This repository provides the codebase for predicting age-related cognitive decline using MRI-derived brain networks.  
It leverages Graph Neural Networks (GNNs) to model relationships between brain regions and estimate age from structural or multimodal neuroimaging data.

Key features:

- Modular architecture for training and evaluation of GNN models
- Supports single-modality and multimodal MRI inputs
- Includes scripts for data handling, model training, evaluation, and visualization

---

## 🗂 Repository Structure

GNN_Age_Prediction/
├── datasets/ # Data preprocessing and storage
├── models/ # GNN architectures
├── Config.py # Default configuration
├── Config_LEMON.py # LEMON dataset configuration
├── Config_atrophy.py # Atrophy-based configuration
├── Dockerfile # Docker environment setup
├── connections_plots.py # Visualizations of brain network connections
├── evaluate.py # Evaluation scripts for trained models
├── run.py # Main script for training/testing
├── run_atrophy.py # Atrophy-specific run scripts
├── run_multimodal.py # Multimodal GNN training/testing
├── train.py # Training pipeline
└── utils.py # Utility functions
