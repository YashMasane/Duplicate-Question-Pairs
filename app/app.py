import streamlit as st 
import pickle
from mode import preprocessing

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.header('Duplicate Question Pairs')

q1 = st.text_input('Enter question 1')
q2 = st.text_input('Enter question 2')

if st.button('Find'):

    inp = preprocessing(q1, q2)
    result = model.predict(inp)

    if result:
        st.header('Duplicate')
    else:
        st.header('Not Duplicate')    

