# Emotion Recognition from Conversational Context (SS-BED)

This project focuses on emotion recognition from conversational context using a combination of **Sentiment-Specific Word Embeddings (SSWE)** and **GloVe embeddings**. The goal is to classify emotions such as **happy**, **sad**, **angry**, and **others** based on the context and target utterances in a conversation.

## Overview

The project implements two models:

1. **SS-BED Model**: A baseline model that processes context and target utterances using separate LSTMs for SSWE and GloVe embeddings. The hidden states from both LSTMs are concatenated and used for emotion classification.
  
2. **Enhanced SS-BED Context Model**: An enhanced version of the SS-BED model that uses **bidirectional LSTMs (BiLSTMs)** to capture both forward and backward context, providing better handling of context and target utterances.

Both models are trained to classify emotions into four categories: **happy**, **sad**, **angry**, and **others**.

## Requirements

Before running the project, ensure you have the following installed:

- Python 3.x
- PyTorch
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

You can install the necessary dependencies using the following command:

```bash
pip install -r requirements.txt
```

## File Structure

```
.
├── data.py                 # Data preprocessing and dataset class for the baseline SS-BED
├── data_enhanced.py        # Data preprocessing and dataset class for our new architecture
├── embeddings.py           # Loading and building embedding matrices (SSWE and GloVe)
├── model.py                # Model architectures (SSBedModel and SSBedContextModel)
├── train.py                # Training and evaluation functions
├── evaluate.py             # Model evaluation and performance metrics
├── main.py                 # Main script to run the experiments
├── word_embeddings/        # Directory containing SSWE and GloVe embedding files
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation 
```

## Data

The dataset used in this project is based on a conversational dataset with three turns per conversation. The dataset includes:

- **Turn1** and **Turn2**: Contextual utterances
- **Turn3**: The target utterance that contains the emotion to be classified

The data is preprocessed, tokenized, and padded/truncated to a fixed sequence length.

## Training and Evaluation

### Hyperparameters

- **Batch Size**: 128 (for SS-BED) / 4000 (for enhanced SS-BED context model)
- **Maximum Sequence Length**: 30 (basic) / 60 (enhanced)
- **SSWE Embedding Dimension**: 50
- **GloVe Embedding Dimension**: 100
- **Hidden Dimension (LSTM)**: 64
- **Number of LSTM Layers**: 2
- **Dropout Rate**: 0.25 (SS-BED) / 0.3 (Enhanced SS-BED)
- **Learning Rate**: 0.005
- **Epochs**: 8
- **Device**: CUDA (GPU) if available, otherwise CPU

### Running the Experiments

To run the experiments, simply execute the `main.py` script. The script will:

1. Load SSWE and GloVe embeddings.
2. Preprocess the dataset and build the necessary DataLoaders.
3. Initialize and train the models.
4. Evaluate the models on the test set and print performance metrics (accuracy, F1 score, confusion matrix).

You can run the basic SS-BED model or the enhanced SS-BED context model by modifying the `main.py` script to choose the appropriate model.

```bash
python main.py
```

### Model Evaluation

The evaluation results include:

- **Classification Report**: Precision, recall, F1-score for each class (happy, sad, angry, others).
- **Confusion Matrix**: To visualize the classification performance across all classes.

### Example Output

```
Loading embeddings...
Building DataLoaders...
Building embedding matrices...
Initializing SS-BED model...
Training...
Epoch [1/8] Train Loss: 1.2345 Val Loss: 1.0150 Val Acc: 0.7012
Epoch [2/8] Train Loss: 1.1020 Val Loss: 0.9874 Val Acc: 0.7105
...
Final evaluation on test set...
Test Loss: 0.9652, Test Acc: 0.7251
Full Classification Report (including 'others'):
              precision    recall  f1-score   support

      others       0.70      0.80      0.74       350
       happy       0.76      0.70      0.73       300
         sad       0.65      0.55      0.60       220
       angry       0.80      0.80      0.80       250

    accuracy                           0.73      1120
   macro avg       0.73      0.71      0.72      1120
weighted avg       0.73      0.73      0.73      1120

Confusion Matrix:
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
