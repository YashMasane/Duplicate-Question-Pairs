import streamlit as st 
from tensorflow.keras.models import load_model
import pickle
from mode import preprocessing

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.header('Duplicate Question Pairs')

q1 = st.text_input('Enter question 1')
q2 = st.text_input('Enter question 2')

if st.button('Find'):
    if len(q1) == 0 or len(q2) == 0:
        st.write("Please enter both questions")
    else:    
        inp = preprocessing(q1, q2)
        result = model.predict(inp)

        if result:
            st.header('Duplicate')
        else:
            st.header('Not Duplicate')    

