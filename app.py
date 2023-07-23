import streamlit as st
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv("data.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df.isnull().sum()


# In[7]:


for item in df.columns:
    print(df[item].name," "*(55-len(df[item].name)), df[item].isnull().sum())


# In[8]:


df["expenditure"].isnull().sum()


# In[9]:


#inpatient
df["expenditure"] = df["expenditure"].fillna(df["expenditure"]. median())


# In[10]:


df["loss_of_hh_income_due_to_hospitalisation"] = df["loss_of_hh_income_due_to_hospitalisation"].fillna(df["loss_of_hh_income_due_to_hospitalisation"]. median())


# In[11]:


df["total_amt_reimb_by_med_insurance"] = df["total_amt_reimb_by_med_insurance"].fillna(df["total_amt_reimb_by_med_insurance"]. median())


# In[12]:


#outpatient
df["total_expenditure_Rs"] = df["total_expenditure_Rs"].fillna(df["total_expenditure_Rs"]. median())


# In[13]:


df["hh_income_loss_Rs"] = df["hh_income_loss_Rs"].fillna(df["hh_income_loss_Rs"]. median())


# In[14]:


df["total_amount_reimbursed_Rs"] = df["total_amount_reimbursed_Rs"].fillna(df["total_amount_reimbursed_Rs"]. median())


# In[15]:


#children
df["immunisation_expenditure_last_365_days"] = df["immunisation_expenditure_last_365_days"].fillna(df["immunisation_expenditure_last_365_days"]. median())


# In[16]:


#pregnant
df["prenatal_care_total_expenditure"] = df["prenatal_care_total_expenditure"].fillna(df["prenatal_care_total_expenditure"]. median())


# In[17]:


df["delivery_expenditure_home"] = df["delivery_expenditure_home"].fillna(df["expenditure"]. median())


# In[18]:


df["postnatal_care_expenditure"] = df["postnatal_care_expenditure"].fillna(df["expenditure"]. median())


# In[19]:


#df[tot_expen] = prenatal+postnatal+immunisation+delievry
col_list= ['prenatal_care_total_expenditure', 'immunisation_expenditure_last_365_days', 'delivery_expenditure_home','postnatal_care_expenditure']
df['tot_expen'] = df[col_list].sum(axis=1)


# In[20]:


for item in df.columns:
    print(df[item].name," "*(55-len(df[item].name)), df[item].isnull().sum())


# In[21]:


df["loss_of_hh_income_due_to_hospitalisation"].isnull().sum()


# In[22]:


#final = expen(inpatient)+expen(outpatient)+expen(preg)
col_list= ['expenditure','total_expenditure_Rs','tot_expen']
df['final_expenditure'] = df[col_list].sum(axis=1)


# In[23]:


df


# In[24]:


col_list_hh_loss = ['hh_income_loss_Rs','loss_of_hh_income_due_to_hospitalisation']
df['final_hh_loss'] =df[col_list_hh_loss].sum(axis=1)


# In[25]:


df


# In[26]:


col_list_reim = ['total_amount_reimbursed_Rs','total_amt_reimb_by_med_insurance']
df['final_reim_amt'] =df[col_list_reim].sum(axis=1)


# In[27]:


df


# In[28]:


df = df.assign(oope1=df['final_expenditure']-df['final_reim_amt'])
df


# In[29]:


df = df.assign(oope2=df['final_expenditure']+df['final_hh_loss']-df['final_reim_amt'])
df


# In[30]:


df_oope1 = df.groupby('hhid').apply(lambda x: np.sum(x['oope1']))


# In[31]:


df4 = df_oope1.to_frame()
df4 = df4.reset_index(inplace=False)


# In[32]:


df4.rename(columns = {0:'OOPE1'}, inplace = True)


# In[33]:


df4


# In[34]:


df_oope2 = df.groupby('hhid').apply(lambda x: np.sum(x['oope2']))


# In[35]:


df5 = df_oope2.to_frame()
df5 = df5.reset_index(inplace=False)
df5


# In[36]:


df5.rename(columns = {0:'OOPE2'}, inplace = True)


# In[37]:


df5


# In[38]:


df7 = pd.merge(df4, df5.rename(columns={'hhid':'hhid'}), on='hhid', how='left')
df7


# In[39]:


df_hh_consumer_expen = df.groupby('hhid').apply(lambda x: np.max(x['hh_consumer_expen']))


# In[40]:


df6 = df_hh_consumer_expen.to_frame()
df6 = df6.reset_index(inplace=False)


# In[41]:


df6.rename(columns = {0:'monthly_hh_income'}, inplace = True)
df6


# In[42]:


df_hh_final_expen = df.groupby('hhid').apply(lambda x: np.sum(x['final_expenditure']))


# In[43]:


df9 = df_hh_final_expen.to_frame()
df9 = df9.reset_index(inplace=False)


# In[44]:


df9.rename(columns = {0:'final_expenditure'}, inplace = True)
df9


# In[45]:


df_hh_final_loss = df.groupby('hhid').apply(lambda x: np.sum(x['final_hh_loss']))


# In[46]:


df10 = df_hh_final_loss.to_frame()
df10 = df10.reset_index(inplace=False)


# In[47]:


df10.rename(columns = {0:'final_hh_loss'}, inplace = True)
df10


# In[48]:


df_hh_final_reim_amt = df.groupby('hhid').apply(lambda x: np.sum(x['final_reim_amt']))


# In[49]:


df11 = df_hh_final_reim_amt.to_frame()
df11 = df11.reset_index(inplace=False)


# In[50]:


df11.rename(columns = {0:'final_reim_amt'}, inplace = True)
df11


# In[51]:


df8 = pd.merge(df7, df6.rename(columns={'hhid':'hhid'}), on='hhid', how='left')
df8


# In[52]:


df8.columns


# In[53]:


df8.loc[:,"hhid"]


# In[54]:


conditions1 = [(df8['OOPE1']>=0.05*df8['monthly_hh_income']*12), (df8['OOPE1']<0.05*df8['monthly_hh_income']*12)]
values = [1,0]
df8['che_1_5']= np.select(conditions1, values)


# In[55]:


df8


# In[56]:


conditions2 = [(df8['OOPE1']>=0.1*df8['monthly_hh_income']*12), (df8['OOPE1']<0.1*df8['monthly_hh_income']*12)]
values = [1,0]
df8['che_1_10']= np.select(conditions2, values)
df8


# In[57]:


conditions3 = [(df8['OOPE1']>=0.15*df8['monthly_hh_income']*12), (df8['OOPE1']<0.15*df8['monthly_hh_income']*12)]
values = [1,0]
df8['che_1_15']= np.select(conditions3, values)
df8


# In[58]:


conditions4 = [(df8['OOPE1']>=0.25*df8['monthly_hh_income']*12), (df8['OOPE1']<0.25*df8['monthly_hh_income']*12)]
values = [1,0]
df8['che_1_25']= np.select(conditions4, values)
df8


# In[59]:


conditions5 = [(df8['OOPE2']>=0.05*df8['monthly_hh_income']*12), (df8['OOPE2']<0.05*df8['monthly_hh_income']*12)]
values = [1,0]
df8['che_2_5']= np.select(conditions5, values)
df8


# In[60]:


conditions6 = [(df8['OOPE2']>=0.1*df8['monthly_hh_income']*12), (df8['OOPE2']<0.1*df8['monthly_hh_income']*12)]
values = [1,0]
df8['che_2_10']= np.select(conditions6, values)
df8


# In[61]:


conditions7 = [(df8['OOPE2']>=0.15*df8['monthly_hh_income']*12), (df8['OOPE2']<0.15*df8['monthly_hh_income']*12)]
values = [1,0]
df8['che_2_15']= np.select(conditions7, values)
df8


# In[62]:


conditions8 = [(df8['OOPE2']>=0.25*df8['monthly_hh_income']*12), (df8['OOPE2']<0.25*df8['monthly_hh_income']*12)]
values = [1,0]
df8['che_2_25']= np.select(conditions8, values)
df8


# In[63]:


df_hh = pd.read_csv("hh_info.csv")


# In[64]:


df_hh.head()


# In[65]:


df_hh = df_hh.drop('common_id',axis=1)


# In[66]:


df_hh


# In[67]:


df_hh.isnull().sum()


# In[68]:


df_hh["med_insurance_premium"] = df_hh["med_insurance_premium"].fillna(df_hh["med_insurance_premium"]. median())


# In[69]:


df_hh["acc_to_latrine"] = df_hh["acc_to_latrine"].fillna(df_hh["acc_to_latrine"]. median())
df_hh["members_use_latrine"] = df_hh["members_use_latrine"].fillna(df_hh["members_use_latrine"]. median())


# In[70]:


df_hh = df_hh.drop('nic_code', axis=1)


# In[71]:


df_hh = df_hh.drop('nco_code', axis=1)


# In[72]:


df_hh.head()


# In[73]:


df_hh.isnull().sum()


# In[74]:


df_hh["src_of_energy_cooking"] = df_hh["src_of_energy_cooking"].fillna(df_hh["src_of_energy_cooking"]. median())


# In[75]:


df_hh.isnull().sum()


# In[76]:


df_hh.shape


# In[77]:


df_hh = pd.merge(df_hh, df9.rename(columns={'hhid':'hhid'}), on='hhid', how='left')
df_hh


# In[78]:


df_hh = pd.merge(df_hh, df10.rename(columns={'hhid':'hhid'}), on='hhid', how='left')
df_hh


# In[79]:


df_hh = pd.merge(df_hh, df11.rename(columns={'hhid':'hhid'}), on='hhid', how='left')
df_hh


# In[80]:


df_hh = pd.merge(df_hh, df8.rename(columns={'hhid':'hhid'}), on='hhid', how='left')
df_hh


# In[81]:


df_hh.columns


# In[119]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# X = df.drop(columns=['Purchased'])

## Scaling the data
# X = scaler.fit_transform(X)
# Y = df.Purchased

X = df_hh[['hhid', 'hh_size','paid_share_childbirth_expen_for_non_HHD_female_member', 'hh_type','religion',
     'social_group', 'latrine_use', 'acc_to_latrine','members_use_latrine', 'src_drinking_water',
    'arrangement_garbage_disposal', 'src_of_energy_cooking',
       'outbreak_comm_disease', 'med_insurance_premium', 'hh_consumer_expen',
       'stratum', 'substratum', 'subsample', 'weight_sc', 'weight_ss','monthly_hh_income','final_expenditure', 'final_hh_loss', 'final_reim_amt']]
print(X)


# In[83]:


X = scaler.fit_transform(X)
Y = df_hh['che_2_25']


# In[84]:


print(X)


# In[115]:


print(df_hh['med_insurance_premium'].to_string(index=False))


# In[85]:


print(Y)


# In[86]:


from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state=5)


# In[87]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,Y_train)


# In[88]:


y_train_predicted = model.predict(X_train)
y_test_predicted = model.predict(X_test)
y_test_predicted


# In[89]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(Y_test,y_test_predicted)

sns.heatmap(pd.DataFrame(cnf_matrix),annot=True)
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[90]:


print("Accuracy:",metrics.accuracy_score(Y_test, y_test_predicted))


# In[91]:


cnf_matrix


# In[92]:


tp = cnf_matrix[0][0]
fp = cnf_matrix[0][1]
fn = cnf_matrix[1][0]
tn = cnf_matrix[1][1]


print("\nTrue Positives : ",cnf_matrix[0][0])
print("\nFalse Positives : ",cnf_matrix[0][1])
print("\nFalse Negatives : ",cnf_matrix[1][0])
print("\nTrue Negatives : ",cnf_matrix[1][1])


print("\n Accuracy : ",metrics.accuracy_score(Y_test,y_test_predicted))

print()
from sklearn.metrics import classification_report
print(classification_report(Y_test, y_test_predicted))


# In[93]:


X = scaler.fit_transform(X)
Y = df_hh['che_2_25']


# In[94]:


print(X)


# In[95]:


print(Y)


# In[96]:


from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state=5)


# In[97]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,Y_train)


# In[98]:


y_train_predicted = model.predict(X_train)
y_test_predicted = model.predict(X_test)
y_test_predicted


# In[99]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(Y_test,y_test_predicted)

sns.heatmap(pd.DataFrame(cnf_matrix),annot=True)
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[100]:


print("Accuracy:",metrics.accuracy_score(Y_test, y_test_predicted))


# In[101]:


tp = cnf_matrix[0][0]
fp = cnf_matrix[0][1]
fn = cnf_matrix[1][0]
tn = cnf_matrix[1][1]


print("\nTrue Positives : ",cnf_matrix[0][0])
print("\nFalse Positives : ",cnf_matrix[0][1])
print("\nFalse Negatives : ",cnf_matrix[1][0])
print("\nTrue Negatives : ",cnf_matrix[1][1])


print("\n Accuracy : ",metrics.accuracy_score(Y_test,y_test_predicted))

print()
from sklearn.metrics import classification_report
print(classification_report(Y_test, y_test_predicted))


# In[102]:


from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import accuracy_score


# In[103]:


# svc = SVC(C=1.0, kernel='rbf', gamma='auto') 
# svc.fit(X_train, Y_train)
# Y_pred = svc.predict(X_test)


# In[104]:


# print("MSE:", mean_squared_error(Y_test, Y_pred))
# print("MAE:", mean_absolute_error(Y_test, Y_pred))
# print("RMSE:", np.sqrt(mean_squared_error(Y_test, Y_pred)))
# print("R2 Score:",metrics.r2_score(Y_test, Y_pred))
# print("Accuracy Score for SVC:", accuracy_score(Y_test, Y_pred))


# # In[126]:


# input_data = (500001101,3,2,1,2,9,2,9.0,4,9,1,2,1,0.0,7000,2,4,1,330.315,660.63,16500, 51330.0, 5600.0, 0.0 )
# # Change the input data to a numpy array
# numpy_data= np.asarray (input_data)
# # reshape the numpy array as we are predicting for only on instance
# input_reshaped = numpy_data.reshape (1,-1)
# prediction = model.predict (input_reshaped)
# if (prediction[0]== 0):  
#     print ("Not under povertyline")
# else:  
#     print ("Under povertyline")


# # In[108]:


# #Saving the trained model
# import pickle
# filename = 'trained_model.sav'
# #dump=save your trained model
# pickle.dump (model,open (filename,'wb'))
# #loading the saved model
# loaded_model = pickle.load (open ('trained_model.sav','rb'))


# In[ ]:


# import pickle
# saved_model = pickle.dumps(model)
  
# # Load the pickled model
# model_from_pickle = pickle.loads(saved_model)
  
# # Use the loaded pickled model to make predictions
# model_from_pickle.predict(X_test)

#     #Saving the trained model
# from joblib import Parallel, delayed
# import joblib

# joblib.dump(model, 'filename.pkl')
  
# # Load the model from the file
# model_from_joblib = joblib.load('filename.pkl')
  
# # Use the loaded model to make predictions
# model_from_joblib.predict(X_test)



st.title("Would you have survived the Titanic Disaster?")
st.subheader("This model will predict if a passenger would survive the Titanic Disaster or not")
st.table(df_hh.head(5))
confusion = confusion_matrix(y_test, y_predict)
FN = confusion[1][0]
TN = confusion[0][0]
TP = confusion[1][1]
FP = confusion[0][1]
# st.subheader("Train Set Score: {}".format ( round(train_score,3)))
# st.subheader("Test Set Score: {}".format(round(test_score,3)))
plt.bar(['False Negative' , 'True Negative' , 'True Positive' , 'False Positive'],[FN,TN,TP,FP])
st.pyplot()
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
final_expenditure = st.text_input(' final_expenditure')
final_hh_loss = st.text_input('final_hh_loss')
final_reim_amt = st.text_input('final_reim_amt')

input_data = scaler.transform([[hhid, hh_size,paid_share_childbirth_expen_for_non_HHD_female_member, hh_type,religion,
     social_group, latrine_use, acc_to_latrine,members_use_latrine, src_drinking_water,
    arrangement_garbage_disposal, src_of_energy_cooking,
       outbreak_comm_disease, med_insurance_premium, hh_consumer_expen,
       stratum, substratum, subsample, weight_sc, weight_ss,monthly_hh_income,final_expenditure, final_hh_loss, final_reim_amt]])
prediction = model.predict(input_data)
predict_probability = model.predict_proba(input_data)

if (prediction[0] == 0):
        st.success ('Not under povertyline')
else:
        st.warning ('Under povertyline')
