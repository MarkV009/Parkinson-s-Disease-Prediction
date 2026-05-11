import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('parkinsons_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Parkinson Disease Prediction App')
st.write('Enter the patient voice measurements below.')

# Input fields
# Input fields
fo = st.number_input('MDVP:Fo(Hz) - Average vocal pitch (Fundamental frequency)', value=119.992)

fhi = st.number_input('MDVP:Fhi(Hz) - Highest vocal pitch', value=157.302)

flo = st.number_input('MDVP:Flo(Hz) - Lowest vocal pitch', value=74.997)

Jitter_percent = st.number_input('MDVP:Jitter(%) - Percentage variation in pitch', value=0.00784)

Jitter_Abs = st.number_input('MDVP:Jitter(Abs) - Absolute pitch variation', value=0.00007)

RAP = st.number_input('MDVP:RAP - Short-term pitch irregularity', value=0.00370)

PPQ = st.number_input('MDVP:PPQ - Pitch perturbation quotient', value=0.00554)

DDP = st.number_input('Jitter:DDP - Average pitch difference between cycles', value=0.01109)

Shimmer = st.number_input('MDVP:Shimmer - Loudness variation', value=0.04374)

Shimmer_dB = st.number_input('MDVP:Shimmer(dB) - Loudness variation in decibels', value=0.426)

APQ3 = st.number_input('Shimmer:APQ3 - Amplitude variation over 3 cycles', value=0.02182)

APQ5 = st.number_input('Shimmer:APQ5 - Amplitude variation over 5 cycles', value=0.03130)

APQ = st.number_input('MDVP:APQ - General amplitude perturbation quotient', value=0.02971)

DDA = st.number_input('Shimmer:DDA - Average amplitude difference', value=0.06545)

NHR = st.number_input('NHR - Noise-to-Harmonics Ratio', value=0.02211)

HNR = st.number_input('HNR - Harmonics-to-Noise Ratio (voice clarity)', value=21.033)

RPDE = st.number_input('RPDE - Speech irregularity measure', value=0.414783)

DFA = st.number_input('DFA - Long-range speech correlation measure', value=0.815285)

spread1 = st.number_input('spread1 - Nonlinear frequency variation', value=-4.813031)

spread2 = st.number_input('spread2 - Secondary nonlinear variation measure', value=0.266482)

D2 = st.number_input('D2 - Complexity of voice dynamics', value=2.301442)

PPE = st.number_input('PPE - Pitch period entropy (voice unpredictability)', value=0.284654)

# Prediction button
if st.button('Predict'):

    input_data = np.array([[fo, fhi, flo, Jitter_percent, Jitter_Abs,
                            RAP, PPQ, DDP, Shimmer, Shimmer_dB,
                            APQ3, APQ5, APQ, DDA, NHR, HNR,
                            RPDE, DFA, spread1, spread2, D2, PPE]])

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)

    if prediction[0] == 1:
        st.error('The person is likely to have Parkinson disease.')
    else:
        st.success('The person is unlikely to have Parkinson disease.')
        