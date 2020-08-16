from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('deployment_28042020_classification')
#model = load_model('deployment_28042020')


def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():

    from PIL import Image
    #image = Image.open('logo.png')
    image_hospital = Image.open('hospital.jpg')

    #st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to predict patient emergency risk')
    st.sidebar.success('https://www.pycaret.org')
    
    st.sidebar.image(image_hospital)

    st.title("Alder Hey Emergency Triage App")
    
    
    if add_selectbox == 'Online':

        Age = st.number_input('Age', min_value=0, max_value=20, value=6)
     
     
        Gender = st.selectbox('Gender', ['male', 'female'])
        Transport = st.selectbox('Transport', ['Ambulance', 'Helicopter', 'Other'])
        Visit_Reason = st.selectbox('Visit_Reason', ['Allergy (Including Anaphylaxis)',  'Infectious Disease', 'Diagnosis Not Classifiable', 'Gastrointestinal Conditions', 'Other Vascular Conditions', 'Respiratory Conditions', 'Burns and Scalds', 'Sprain/Ligament Injury', 'Nothing Abnormal Detected', 'Head Injury', 'Facio-Maxillary Conditions', 'ENT Conditions', 'Urological Conditions (Including Cystitis)', 'Local Infection', 'Soft Tissue Inflammation', 'Muscle/Tendon Injury', 'Central Nervous System Conditions (Excluding Strokes)', 'Psychiatric Conditions', 'Contusion/Abrasion', 'Dislocation/Fracture/Joint Injury/Amputation', 'Laceration', 'Dermatological Conditions', 'Poisoning (Including Overdose)', 'Diabetes and Other Endocrinological Conditions', 'Foreign Body', 'Social Problem (Includes Chronic Alcoholism and Homelessness)', 'Cardiac Conditions', 'Obstetric Conditions', 'Bites/Stings', 'Gynaecological Conditions', 'Ophthalmological Conditions', 'Haematological Conditions', 'Cerebro-Vascular Conditions', 'Visceral Injury', 'Vascular Injury', 'Nerve Injury', 'Near Drowning'])
        REFER_SOURCE= st.selectbox(' REFER_SOURCE', ['SELF', 'CONS IN HOSP', 'AE', 'OTHER'])
        AVPU = st.selectbox('AVPU', ['Alert', 'Verbal', 'Pain', 'Unresponsive'])
        PulseRate = st.number_input('PulseRate', min_value=10, max_value=150, value=70)
        RespiratoryRate	 = st.number_input('RespiratoryRate', min_value=10, max_value=100, value=25)
        SP02 = st.number_input('SP02', min_value=80, max_value=100, value=100)
        Temperature = st.number_input('Temperature', min_value=10, max_value=50, value=37)

        output=""
        
        input_dict = {'Age' : Age, 'Gender' : Gender, 'Transport' : Transport, 
        'Visit_Reason' : Visit_Reason, 'REFER_SOURCE' : REFER_SOURCE, 
        'AVPU' : AVPU, 'PulseRate' : PulseRate, 'RespiratoryRate' : RespiratoryRate	, 
        'SP02' : SP02, 'Temperature' : Temperature }
        
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = round(output*100)
   
        st.success('This patient triage risk is  {}'.format(output))
        
        

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()