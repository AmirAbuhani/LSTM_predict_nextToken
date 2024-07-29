# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 18:00:18 2024

@author: AMIR
"""

"""
Note: I ran the model using Google Colab on a GPU runtime type.
see tests in the READMD.md file.
Recommended: Run the model using the Google Colab on a GPU runtime type.
 """
 
# Importing libraries
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding


nltk.download('punkt')

################### Part 1  - Data Preprocessing ##############################
# Open the txt file (the book) and read it
file = open("A-Tale-of-Two-Cities.txt").read()

# Tokenize the text into words
words = word_tokenize(file)  # List of words

# To get unique(set) and sorted words
unique_words = sorted(list(set(words)))

# Building the dictionary
word_to_num = dict((c, i) for i, c in enumerate(unique_words))

input_len = len(words)
vocab_len = len(unique_words)
seq_length = 10  # Fixed number of words in each sequence
x = []  # train_input_list
y = []  # train_output_list

for i in range(0, input_len - seq_length, 1):
    in_seq = words[i:i + seq_length]
    out_seq = words[i + seq_length]
    # Transform the words to numbers using the word_to_num dictionary
    x_to_append = [word_to_num[word] for word in in_seq]
    y_to_append = word_to_num[out_seq]
    x.append(x_to_append)
    y.append(y_to_append)

n_patterns = len(x)

# Reshape x to 2D and y to 1D 
X = np.array(x)
y = np.array(y)

####################### Part 2 - Building the LSTM ############################
model = Sequential()
# Adding the embedding layer: Because representing one hot encoder for each word 
# is making very large vectors, using Embedding layer making Dimensionality Reduction and Efficient Representation.
model.add(Embedding(input_dim=vocab_len, output_dim=100, input_length=seq_length))
# Adding LSTM layers
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(256))
# Adding the output layer
model.add(Dense(vocab_len, activation='softmax'))

##################### Part 3 - Training the LSTM ##############################
# Compiling the LSTM
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Training the LSTM on the Training set
model.fit(X, y, batch_size=256, epochs=70)

######################## Part 4 - Saving the Model ###########################
# Save the trained model
model.save('word_trained_model_70.h5')

######################### Part 5 - Testing the model ##########################

'''
#In order to load the model - Load the entire model from the HDF5 file 
from keras.models import load_model
model = load_model("word_trained_model_70.h5")
'''

"""
Since we converted the words to numbers earlier, 
we need to define a dictionary variable that will convert the output 
of the model back into 
"""
num_to_word = dict((i, c) for i, c in enumerate(unique_words))

# Generate text
if len(x) <= seq_length:
    print("Error: The length of the data is too short for the specified sequence length.")
else:
    start = np.random.randint(0, len(x) - seq_length)
    pattern = x[start]
    print("Random Seed:")
    print("\"", ' '.join([num_to_word[value] for value in pattern]), "\"")

# Generate text
for i in range(10):  # Adjust the number of words to generate
    x_input = np.reshape(pattern, (1, len(pattern)))
    # Predict the next word
    prediction = model.predict(x_input, verbose=0)
    index = np.argmax(prediction)
    result = num_to_word[index]

    # Print the predicted word
    sys.stdout.write(result + ' ')

    # Update the seed sequence for the next iteration
    pattern.append(index)
    pattern = pattern[1:len(pattern)]

print("\nDone.")