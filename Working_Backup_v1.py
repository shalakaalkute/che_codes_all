#Importing the libraries
import numpy as np
import pickle
import streamlit as st
from sklearn.utils import check_array

#loading the saved model
loaded_model = pickle.load (open("D:\\BE\\BE Project\\BE Project\\trained_model.sav", 'rb'))

#Creating a function for Prediction
def che_pred (input_data):
# changing the input data to a numpy array
    numpy_data= np.asarray (input_data, dtype=float)
#Reshaping the numpy array as we are predicting for only on instance
    input_reshaped = numpy_data.reshape (1,-1)
    prediction = loaded_model.predict(input_reshaped)
   
    if (prediction[0] == 0):
        st.success ('Not under povertyline')
    else:
        st.warning ('Under povertyline')

        
#Adding title to the page
st.title ('CHE prediction')

#Getting the input data from the user
hhid = st.text_input ('Hhid')
hh_size = st.text_input ('hh size')
paid_share_childbirth_expen_for_non_HHD_female_member= st.text_input ('paid share')
hh_type= st.text_input ('hh type')
religion = st.text_input ('religion')
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
monthly_hh_income = st.text_input('monthly hh income')
final_expenditure = st.text_input('final_expenditure')
final_hh_loss = st.text_input('final_hh_loss')
final_reim_amt = st.text_input('final_reim_amt')




# code for Prediction
diagnosis = []
# creating a button for Prediction
if st.button ('CHE prediction'):
    diagnosis=che_pred([hhid, hh_size,paid_share_childbirth_expen_for_non_HHD_female_member, hh_type,religion,
     social_group, latrine_use, acc_to_latrine,members_use_latrine, src_drinking_water,
    arrangement_garbage_disposal, src_of_energy_cooking,
       outbreak_comm_disease, med_insurance_premium, hh_consumer_expen,
       stratum, substratum, subsample, weight_sc, weight_ss,monthly_hh_income,final_expenditure, final_hh_loss, final_reim_amt])