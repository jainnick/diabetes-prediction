# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 15:41:59 2022

@author: NIKITA JAIN
"""
#streamlit used for deployment
import numpy as np
import pickle

loaded_model=pickle.load(open("C:/projectML/trainedModel.sav",'rb'))
scaler=pickle.load(open("scaler.pkl",'rb'))

#creating a function for prediction

def diabetes_prediction(input_data):
    
    input_predict_data=np.asarray(input_data)
    input_data_reshaped=input_predict_data.reshape(1,-1)
    std_data=scaler.transform(input_data_reshaped)
    prediction=loaded_model.predict(std_data)
    if prediction[0]==0:
      #print("Congratulations! you are not diabetic")
      return "Congratulations! you are not diabetic"
    else:
      #print( "Hey! You may have debetes. Please consult a doctor")
      return "Hey! You may have debetes. Please consult a doctor"
 
     
def main():
    
    st.title("Diabetes prediction Web app")
    
    #Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
    
    num_preg=st.text_input('Number of Pregnancies')
    Glucose=st.text_input('Glucose Level')
    BloodPressure=st.text_input('BloodPressure value')
    SkinThickness=st.text_input('SkinThickness value')
    Insulin=st.text_input('Insulin Level')
    BMI=st.text_input('BMI value')
    DiabetesPedigreeFunction=st.text_input('Diabetes Predigree Function')
    Age=st.text_input('Age')
    
    
    diagnosis=''
    
    #creating a button
    if st.button('Diabetes test Result'):
        diagnosis=diabetes_prediction([8,125,96,0,0,0,0.232,54])
    
    st.success(diagnosis)
    
 #for command prompt run only   
if __name__ == '__main__':
    main()
    
    
    
    
    
    
