import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def load_data(file_path):
    data = open(file_path, 'r', encoding="utf8").read().lower()
    data = data.replace('\n', ' ')
    data = data.split(' ')
    data = np.array(data)
    data = np.array([char for char in data if char])
    return data

def tokenize_data(data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)
    total_words = len(tokenizer.word_index) + 1
    return tokenizer, total_words

def pad_data(data, tokenizer, max_sequence_len):
    data = np.char.split(np.array(data), sep=' ')     #np.item().split(' ')
    data = [tokenizer.word_index[word] for word in data[0] if word in tokenizer.word_index]
    return pad_sequences([data], maxlen=max_sequence_len, padding='post')

def prepare_dataset(data, max_sequence_len):
    tokenizer, total_words = tokenize_data(data)
    input_data = pad_data(data, tokenizer, max_sequence_len)
    target_data = input_data[:,:-1]
    target_data = np.expand_dims(target_data, -1)
    output_data = input_data[:,1:]
    output_data = np.expand_dims(output_data, -1)
    return input_data, target_data, output_data, tokenizer, total_words

def create_model(input_shape, total_words):
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=input_shape[1]-1))
    model.add(LSTM(150, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        tokenized_text = tokenizer.texts_to_sequences([seed_text])[0]
        tokenized_text = pad_sequences([tokenized_text], maxlen=max_sequence_len-1, padding='post')
        predicted_probs = model.predict(tokenized_text)[0]
        index = np.random.choice(range(len(predicted_probs)), p=predicted_probs)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == seed_text:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Parameters
file_path = "input.txt"
data = load_data(file_path)
input_data, target_data, output_data, tokenizer, total_words = prepare_dataset(data, 40)

# Create model
model = create_model((input_data.shape[0], input_data.shape[1]-1), total_words)

# Train model
model.fit(target_data, output_data, epochs=100, verbose=1)

# Generate text
seed_text = "This is a seed sentence"
next_words = 100
generated_text = generate_text(seed_text, next_words, model, 40)
print(generated_text)