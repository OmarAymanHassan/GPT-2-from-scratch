# LLMs From Scratch

This project implements a complete Large Language Model (LLM) from scratch, specifically a GPT (Generative Pre-trained Transformer) model, following a step-by-step approach. The implementation covers everything from data preprocessing and tokenization to model architecture, pretraining, and various fine-tuning techniques.

## Project Overview

The project is structured as a comprehensive tutorial/course on building LLMs from the ground up. It includes Jupyter notebooks that walk through each concept, Python scripts for practical implementation, and saved models for inference and further experimentation.

## Key Components

### 1. Data Preparation and Tokenization (`2_working_with_data.ipynb`)
- Loading and preprocessing text data
- Implementing custom tokenization using regex patterns
- Understanding different tokenization techniques
- Preparing data for model training

### 2. Attention Mechanism (`3_attention_mehcanism.ipynb`)
- Implementing attention scores using dot product and cosine similarity
- Understanding how attention works in transformer models
- Building foundational concepts for the transformer architecture

### 3. GPT Model Architecture (`4_train_gpt_model.ipynb`)
- Defining GPT model configurations (124M parameters)
- Implementing the GPT model class with embeddings and transformer blocks
- Setting up the basic structure for training

### 4. Pretraining the LLM (`5_pretraining_llm.ipynb`)
- Loading pre-built GPT configurations
- Implementing the complete GPT model with transformer blocks
- Pretraining on large text corpora using next-token prediction

### 5. Classification Fine-Tuning (`6_classification_fine_tuning.ipynb`)
- Downloading and preparing the SMS Spam Collection dataset
- Fine-tuning the pretrained GPT model for binary classification
- Implementing classification heads and training procedures

### 6. Instruction Fine-Tuning (`7_instruction_fine_tuning.ipynb` & `7_instruction_fine_tuning_pt2.ipynb`)
- Preparing instruction-response datasets
- Fine-tuning the model for instruction following
- Implementing chat/instruction capabilities

### Additional Chapters
- `ch06.ipynb` & `ch07.ipynb`: Further explorations and implementations

## Python Scripts

### Core Implementation
- `gpt.py`: Complete GPT model implementation with dataset handling
- `gpt_download.py`: Scripts for downloading pretrained models and data
- `gpt_training.py`: Training scripts for pretraining the model
- `gpt_finetuning_classifier.py`: Fine-tuning scripts for classification tasks

### Inference and Deployment
- `inference_fine_tuning_classifier.py`: Inference scripts for fine-tuned classifiers
- `main.py`: Main entry point for the project

## Data

The project uses several datasets:

### Pretraining Data
- `training-text.txt`: Raw text data for pretraining
- `gutenberg/`: Project Gutenberg texts processed for training
  - Raw text files, tokenized data, and metadata

### Fine-Tuning Data
- `sms_spam_collection/`: SMS Spam Collection dataset for classification fine-tuning
- `data/`: Train/validation/test CSV files
- `instruction-data.json` & `instruction-data-with-response.json`: Instruction fine-tuning datasets

## Models

Trained models are saved in the root directory:
- `gpt2-small-124M.pth`: Pretrained GPT-2 small model
- `gpt2-instruct.pth`: Instruction-fine-tuned model
- `review_classifier.pth`: Spam classification model
- `model.pth` & `model_and_optimimzer.pth`: General saved models

## Dependencies

The project uses a comprehensive set of libraries including:
- PyTorch for deep learning
- tiktoken for GPT-2 tokenization
- pandas, numpy for data handling
- scikit-learn for evaluation
- Chainlit for chat interface
- Various NLP and ML libraries (transformers, datasets, etc.)

See `pyproject.toml` for complete dependency list.

## Chat Interface

The project includes a Chainlit-based chat interface for interacting with the trained models. The interface allows users to have conversations with the fine-tuned LLM.

## Project Structure

```
LLMs-From-Scratch/
├── Notebooks/ (2-7, ch06, ch07)
├── Scripts/ (gpt.py, training, fine-tuning, inference)
├── Data/ (gutenberg, sms_spam, csv files)
├── Models/ (saved .pth files)
├── chainlit.md (chat interface config)
├── pyproject.toml (project configuration)
├── requirements.txt
└── README.md
```

## Key Learnings and Implementations

1. **Tokenization**: Custom regex-based tokenization vs. subword tokenization (tiktoken)
2. **Attention**: Self-attention mechanism implementation
3. **Transformer Architecture**: Multi-head attention, feed-forward networks, layer normalization
4. **Pretraining**: Next-token prediction on large corpora
5. **Fine-Tuning**: Task-specific adaptation for classification and instruction following
6. **Model Saving/Loading**: PyTorch model serialization
7. **Data Handling**: Custom datasets and dataloaders for text data

## Usage

1. **Setup**: Install dependencies from `pyproject.toml`
2. **Data Preparation**: Run notebooks 2-3 for understanding data processing
3. **Model Training**: Follow notebooks 4-5 for pretraining
4. **Fine-Tuning**: Use notebooks 6-7 for specific tasks
5. **Inference**: Use provided scripts for model inference
6. **Chat**: Run Chainlit interface for interactive conversations

## Future Extensions

The project provides a solid foundation for:
- Scaling to larger models
- Implementing other transformer variants
- Adding more fine-tuning tasks
- Deploying models in production
- Experimenting with different architectures

This implementation demonstrates the complete lifecycle of building, training, and deploying a modern LLM from scratch.