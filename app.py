import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load model and tokenizer
model = load_model('model.h5')

with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

st.title("Next Word Prediction in a Sentence")
st.subheader("Enter your incomplete sentence below")

# Use normal text_input — NOT chat_input
user_input = st.text_input("Write your incomplete sentence")

if st.button('Predict'):
    
    if user_input.strip() == "":
        st.warning("Please enter a sentence first.")
    else:
        max_length = model.input_shape[1] + 1     # model.input_shape = (None, max_len-1)
        
        # Convert text → tokens
        token_list = tokenizer.texts_to_sequences([user_input])[0]

        # Handle long sequences
        if len(token_list) >= max_length:
            token_list = token_list[-(max_length-1):]

        # Padding
        token_list = pad_sequences([token_list], maxlen=max_length-1, padding='pre')

        # Predict next word
        predictions = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predictions)

        # Reverse lookup: index → word
        predicted_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                predicted_word = word
                break

        # Output
        st.write(f"**Predicted next word:** `{predicted_word}`")
        st.write(f"**Full sentence:** {user_input} {predicted_word}")
