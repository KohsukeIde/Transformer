## Transformer Model for Machine Translation
This repository contains a PyTorch implementation of a Transformer model for machine translation. The Transformer model, introduced in the paper "Attention is All You Need" by Vaswani et al., is a deep learning model that utilizes self-attention mechanisms to process sequential data such as text. This implementation includes the necessary components to train and evaluate a Transformer model on a bilingual dataset for the task of translating text from one language to another.
## Repository Structure
The repository is organized into three main Python files:
- model.py: Contains the implementation of the Transformer model, including the encoder, decoder, and other necessary components such as multi-head attention, positional encoding, and feed-forward networks.
- train.py: Contains the training loop, validation, and utility functions for training the Transformer model on a bilingual dataset. It also includes functions for tokenization and dataset preparation.
- dataset.py: Defines a PyTorch Dataset class for handling bilingual datasets. It preprocesses text data for training, including tokenization and padding.
## Setup
Before running the training script, run pip install -r requirements.txt
