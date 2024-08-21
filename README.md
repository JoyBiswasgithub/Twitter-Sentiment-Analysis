# Twitter-Sentiment-Analysis
# Sentiment Analysis with RNN using TensorFlow

This repository contains a sentiment analysis project that uses a Recurrent Neural Network (RNN) model built with TensorFlow. The project includes data preprocessing, model training, and deployment using Streamlit.

## Project Overview

The goal of this project is to classify text into one of four sentiment categories:
- **Negative**
- **Positive**
- **Neutral**
- **Irrelevant**

The model is trained on a dataset containing text reviews, including emojis, and it uses a SimpleRNN layer to capture the sequential nature of the text data.

## Features

- **Text Preprocessing**: Includes cleaning text data and handling emojis.
- **Model Training**: Uses TensorFlow to build and train an RNN model.
- **Deployment**: Deploys the trained model using Streamlit, allowing users to input text and receive sentiment predictions.

## Model Architecture

The sentiment analysis model uses a Recurrent Neural Network (RNN) with the following architecture:

1. **Embedding Layer**:
   - **Purpose**: Converts words into dense vectors of fixed size.
   - **Parameters**:
     - `input_dim`: Size of the vocabulary (number of unique words).
     - `output_dim`: Dimensionality of the embedding vectors (e.g., 100).
     - `input_length`: Length of the input sequences (e.g., the maximum length of padded sequences).

   **Explanation**: This layer maps each word to a dense vector, capturing semantic meaning and relationships between words. It helps the model understand text better by representing words in a continuous space.

2. **SimpleRNN Layer**:
   - **Purpose**: Captures sequential dependencies and temporal patterns in the text.
   - **Parameters**:
     - `units`: Number of RNN cells or hidden units (e.g., 128).
     - `return_sequences`: Whether to return the full sequence of outputs (set to `False` to return only the final output).

   **Explanation**: The SimpleRNN layer processes the sequence of word embeddings, maintaining an internal state that captures information about previous words in the sequence. This helps the model understand the context and dependencies between words.

3. **Dense Layer**:
   - **Purpose**: Performs the final classification based on the features extracted by the RNN.
   - **Parameters**:
     - `units`: Number of output units (equal to the number of sentiment categories, which is 4).
     - `activation`: Activation function used (e.g., `softmax`).

   **Explanation**: This layer outputs probabilities for each of the sentiment categories. The `softmax` activation function ensures that the output values are in the range [0, 1] and sum up to 1, representing the likelihood of each category.

### Summary

- **Embedding Layer**: Transforms words into dense vectors.
- **SimpleRNN Layer**: Captures sequential patterns and dependencies in the text.
- **Dense Layer**: Outputs classification probabilities for the sentiment categories.

This architecture enables the model to handle sequential text data effectively and make accurate predictions based on learned patterns and representations.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/sentiment-analysis-rnn.git
   cd sentiment-analysis-rnn
