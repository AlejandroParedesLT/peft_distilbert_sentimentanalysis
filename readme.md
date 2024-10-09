```
# README: Parameter Tuning of BERT for Sentiment Classification

## Overview

This project demonstrates the fine-tuning of a `DistilBERT` model for sentiment classification on a truncated IMDb dataset. The aim is to enhance training efficiency by applying parameter-efficient fine-tuning (PEFT) using LoRA (Low-Rank Adaptation). The model classifies movie reviews into positive and negative sentiments.

## Prerequisites

Ensure the following Python libraries are installed:

```bash
pip install transformers datasets peft evaluate
```

The required libraries:
- `transformers`: Provides the BERT model and utilities for fine-tuning.
- `datasets`: Loads and manages the IMDb dataset.
- `peft`: Enables parameter-efficient fine-tuning, especially LoRA.
- `evaluate`: Helps compute metrics like accuracy.

### Dataset  

The **shawhin/imdb-truncated** dataset from Hugging Face is used. This dataset contains truncated IMDb reviews labeled as either positive or negative.

### Steps

1. **Model and Tokenizer Setup**  
   - The **DistilBERT** model checkpoint (`distilbert-base-uncased`) is loaded.
   - Label mappings are defined: "Negative" is mapped to 0, and "Positive" is mapped to 1.
   - The **AutoTokenizer** from the transformers library is used to prepare the text data for input to the model.

2. **Tokenization**  
   - A tokenization function is defined, which truncates inputs to a maximum length of 512 tokens to handle long reviews.
   - Padding is applied to ensure sequences are of equal length during processing.
   - If the tokenizer does not already have a padding token, a special padding token (`[PAD]`) is added to the tokenizer's vocabulary.

3. **Data Preparation**  
   - The IMDb dataset is tokenized by applying the tokenization function to each review. The tokenized dataset is prepared for further use in training and evaluation. Sequences are truncated, padded, and converted into a suitable format for the model.

4. **Model Fine-Tuning**  
   - A fine-tuning configuration is set up using **LoRA (Low-Rank Adaptation)**, a parameter-efficient tuning method. The LoRA configuration includes parameters such as rank, dropout rate, and target modules for adaptation within the model.
   - Model parameters are made trainable according to the LoRA configuration, focusing on specific parts of the model for adaptation while keeping other layers frozen to reduce computational cost.

5. **Training Setup**  
   - The training configuration is defined, which includes settings like learning rate, batch size, number of epochs, and weight decay. Additionally, evaluation and checkpoint strategies are specified to monitor the model’s performance on a validation set during training.
   - The **Trainer** class is used to manage the training process, applying the defined arguments, training and validation datasets, tokenized inputs, and evaluation metrics.

6. **Model Evaluation and Predictions**  
   - After training, the model is evaluated using the validation dataset. The performance is assessed using an accuracy metric to compare predicted labels against true labels.
   - The model's predictions are tested on sample text inputs to verify how well it classifies new reviews as either positive or negative.

7. **Saving the Model**  
   - The fine-tuned model and its tokenizer are saved, allowing for future use in making predictions or for further fine-tuning. The model’s parameters and vocabulary are stored in binary files for efficient loading and inference.