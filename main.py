import streamlit as st
import pickle
import numpy as np
import pandas as pd





# import the model
pipe_lr = pickle.load(open('pipe_lr.pkl','rb'))
pipe_knn = pickle.load(open('pipe_knn.pkl','rb'))
pipe_dt = pickle.load(open('pipe_dt.pkl','rb'))
pipe_rf = pickle.load(open('pipe_rf.pkl','rb'))
pipe_svm = pickle.load(open('pipe_svm.pkl','rb'))
pipe_gb = pickle.load(open('pipe_gb.pkl','rb'))
pipe_ab = pickle.load(open('pipe_ab.pkl','rb'))
pipe_xg= pickle.load(open('pipe_xg.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))


st.title("LAPTOP PRICE PREDICTOR")

# brand
company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

# Ram
ram = st.selectbox('Ram(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the lap')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])


# Ips
ips = st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.number_input("Screen size")

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])


# cpu
cpu = st.selectbox('CPU',df['Cpu brand'].unique())
# hdd
hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])
# ssd
ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])
# gpu
gpu = st.selectbox('GPU',df['Gpu Brand'].unique())
# os
os = st.selectbox('OS',df['os'].unique())

# Model
model_list = ['Linear Regression', 'KNN', 'Decision Tree','Random Forest','SVM','GradientBoost','AdaBoost','XgBoost']
model = st.selectbox('Model', model_list )


if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/(screen_size+1)
    # query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
    # query = query.reshape(1,12)


    # Create the query array
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

    # Convert to a DataFrame to handle data types and preprocessing
    query_df = pd.DataFrame([query],
                            columns=['company', 'type', 'ram', 'weight', 'touchscreen', 'ips', 'ppi', 'cpu', 'hdd',
                                     'ssd', 'gpu', 'os'])

    # Example preprocessing (replace with your actual preprocessing steps)
    # Convert categorical variables to numeric, handle missing values, etc.
    # Assuming these preprocessing steps match those used during model training

    # Example: Converting some categorical variables to numeric
    query_df['ram'] = pd.to_numeric(query_df['ram'], errors='coerce')
    query_df['weight'] = pd.to_numeric(query_df['weight'], errors='coerce')
    query_df['ppi'] = pd.to_numeric(query_df['ppi'], errors='coerce')
    query_df['hdd'] = pd.to_numeric(query_df['hdd'], errors='coerce')
    query_df['ssd'] = pd.to_numeric(query_df['ssd'], errors='coerce')

    # Fill missing values if any (using 0 or another value as appropriate)
    query_df = query_df.fillna(0)

    # Convert DataFrame back to a NumPy array if needed
    query_array = query_df.to_numpy()

    # Ensure the array is in the correct shape for prediction
    query_array = query_array.reshape(1, -1)

    # Predict the price using the selected model
    if model == 'Linear Regression':
        predicted_price = np.exp(pipe_lr.predict(query_array)[0])

    elif model == 'KNN':
         predicted_price = np.exp(pipe_knn.predict(query_array)[0])

    elif model == 'Decision Tree':
         predicted_price = np.exp(pipe_dt.predict(query_array)[0])

    elif model == 'Random Forest':
         predicted_price = np.exp(pipe_rf.predict(query_array)[0])

    elif model == 'SVM':
         predicted_price = np.exp(pipe_svm.predict(query_array)[0])

    elif model == 'GradientBoost':
         predicted_price = np.exp(pipe_gb.predict(query_array)[0])

    elif model == 'AdaBoost':
         predicted_price = np.exp(pipe_ab.predict(query_array)[0])

    elif model == 'XgBoost':
         predicted_price = np.exp(pipe_xg.predict(query_array)[0])
    # Display the result
    st.title("The predicted price of this configuration is " + str(int(predicted_price)))

    # st.title("The predicted price of this configuration is " + str(int(np.exp(pipe_xg.predict(query)[0]))))
