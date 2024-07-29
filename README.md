# Text Generation with LSTM

Author: Amir Abu Hani  


## Overview
This project preprocesses the text data from "A Tale of Two Cities" file, builds
an LSTM neural network model using Keras, trains the model, and generates text sequences
based on the trained model.

### Note: I ran the model using Google Colab on a GPU runtime type.

## Required Libraries
- numpy
- nltk
- keras

## Functionality
This project code includes five main parts:

### Part 1: Data Preprocessing
1. **Loading the Text**: 
    - Read the text file "A Tale of Two Cities".
2. **Tokenization**: 
    - Tokenize the text into words.
3. **Building Vocabulary**: 
    - Create a sorted list of unique words.
4. **Mapping Words to Numbers**: 
    - Create a dictionary to map each word to a unique number.
5. **Creating Sequences**: 
    - Generate input sequences (X) and corresponding output words (y) for the LSTM model.
6. **Reshaping and Normalizing Data**: 
    - Reshape the input data into a 2D array and normalize it.
7. **Encoding Output**: 
    - Use integer encoding for the output words instead of one-hot encoding for better efficiency.

### Part 2: Building the LSTM
- **Architecture**:
    - Embedding Layer: Convert input word indices into dense vectors.
    - Two LSTM Layers: With 256 units each.
    - Output Layer: Dense layer with a softmax activation function.

### Part 3: Training the LSTM
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Parameters**: 
    - Batch Size: 256
    - Epochs: 70
    - accuracy: about 85%

### Part 4: Saving the Model
- Save the trained model to `word_trained_model_70.h5`.

### Part 5: Testing the Model
1. **Loading the Model**: 
    - Load the model from the saved file using `keras.models.load_model`.
2. **Seed Generation**: 
    - Use a random sequence from the text to start the text generation.
3. **Text Generation**: 
    - Predict the next word and update the seed sequence iteratively to generate new text.

## How to Run the Code
1. Execute Parts 1, 2, and 3 together to preprocess data, build, and train the model.
2. Run Part 4 to save the trained model.
3. Run Part 5 without the load.

Alternatively:
1. Load the saved model: 
    ```python
    from keras.models import load_model
    model = load_model('word_trained_model_70.h5')
    ```
2. Make predictions on new text sequences as demonstrated in Part 5.

### Different Text Samples:
1. First Sample:
    Random Seed: " it , monsieur . '' Mr. Lorry took it in "
    Result: his hand . `` Tell me what the prisoner are
2. Second Sample:
    Random Seed: " see only Madame Defarge in her seat , presiding over "
    Result: the distribution of wine , with a bowl of battered
2. Third Sample:
    Random Seed: " and which had been described to all the world . "
    Result: He besought her -- though they were as they had