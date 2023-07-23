#Importing the libraries
import numpy as np
import pickle
import streamlit as st
from sklearn.utils import check_array

#loading the saved model
loaded_model = pickle.load (open("D:\\BE\\BE Project\\BE Project\\rand_model.sav", 'rb'))

#Creating a function for Prediction
def che_pred (input_data):
# changing the input data to a numpy array
    numpy_data= np.asarray (input_data, dtype=float)
#Reshaping the numpy array as we are predicting for only on instance
    input_reshaped = numpy_data.reshape (1,-1)
    prediction = loaded_model.predict(input_reshaped)
   
    if (prediction[0] == 1):
        st.success ('Under povertyline')
    else:
        st.warning ('Not Under povertyline')

        
#Adding title to the page
st.title ('CHE prediction')

#Getting the input data from the user
hhid = st.text_input ('Household Id')
hh_size = st.text_input ('Size of Household')
paid_share_childbirth_expen_for_non_HHD_female_member= st.text_input ('Share Paid by Non Household Female')
hh_type= st.text_input ('Type Household')
religion = st.text_input ('Religion')
social_group = st.text_input ('social group')
latrine_use = st.text_input ('latrine use')
acc_to_latrine = st.text_input ('access to latrine')
members_use_latrine = st.text_input ('mem use latrine')
src_drinking_water = st.text_input ('src of drinking water')
arrangement_garbage_disposal = st.text_input ('garbage disposal')
src_of_energy_cooking= st.text_input ('src of energy cooking')
outbreak_comm_disease = st.text_input ('outbreak of disease')
med_insurance_premium = st.text_input ('med insurance premium')
hh_consumer_expen= st.text_input ('hh consumer expenses')
stratum = st.text_input ('stratum')
substratum = st.text_input ('substratum')
subsample = st.text_input ('subsample')
weight_sc = st.text_input ('weight_sc')
weight_ss = st.text_input(' weight_ss')
dep_ratio = st.text_input('dep_ratio')





# code for Prediction
diagnosis = []
# creating a button for Prediction
if st.button ('CHE prediction'):
    diagnosis=che_pred([hhid, hh_size,paid_share_childbirth_expen_for_non_HHD_female_member, hh_type,religion,
     social_group, latrine_use, acc_to_latrine,members_use_latrine, src_drinking_water,
    arrangement_garbage_disposal, src_of_energy_cooking,
       outbreak_comm_disease, med_insurance_premium, hh_consumer_expen,
       stratum, substratum, subsample, weight_sc, weight_ss,dep_ratio])