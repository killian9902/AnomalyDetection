import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import time
import torch
import torch.nn as nn
import joblib
from DataStream import data_generation, window_length




# Title
#   - For UI, HTML is passed via markdown for custom styling of the title
#   - 'unsafe_allow_html', allows the use of raw HTML
st.markdown("<h1 style='text-align: center;'>Data Stream Anomaly Detection</h1>", unsafe_allow_html=True)
st.markdown("For demonstration purposes: \n\n The data stream represents a system metric: **Gas Flow Rates** ($m^{3}/s$) in a **pipeline**")


# Subtitle
st.markdown("<h2 style='text-align: center;'>Creating Data Generator</h2>", unsafe_allow_html=True)
st.markdown("For demonstration purposes: all charts have the same intial **rate** of 250 to display how the applied effect alters the rate. Each chart only applies the given effect.")




# Data generator: display general noise
st.markdown("<h4 style='text-align: left;'>1. General Noise (Gaussian Mask)</h4>", unsafe_allow_html=True)
st.markdown("The Gaussian mask is ideal becasue it follows a normal distirbution, adding non-disruptive noise. This mask has a **mean** of 0 and **standard deviation** of 1.5.")

if 'data_values_noise' not in st.session_state:
    data_stream_noise = data_generation(flag_rate=False, flag_noise=True, flag_seasonality=False, flag_trend = False, flag_anomalies = False)
    st.session_state.data_values_noise = pd.DataFrame({'Time': range(800), 'Rate': [next(data_stream_noise) for _ in range(800)]})

chart_noise = alt.Chart(st.session_state.data_values_noise).mark_line().encode(
    x='Time',
    y=alt.Y('Rate', scale=alt.Scale(domain=[0, 500]), axis=alt.Axis(title='Rate (m³/s)'))
).properties(title=alt.TitleParams('General Noise', anchor='middle', fontSize=24))

st.altair_chart(chart_noise, use_container_width=True)




# Data generator: display seasonality
st.markdown("<h4 style='text-align: left;'>2. Seasonality (Sine Wave)</h4>", unsafe_allow_html=True)
st.markdown("A sine wave is used to introduce seasonal cyclical effects to the rate, with an **amplitude** of 4 and a **period** of 200 time steps.")


if 'data_values_season' not in st.session_state:
    data_stream_season = data_generation(flag_rate=False, flag_noise=False, flag_seasonality=True, flag_trend = False, flag_anomalies = False)
    st.session_state.data_values_season = pd.DataFrame({'Time': range(800), 'Rate': [next(data_stream_season) for _ in range(800)]}) 

chart_season = alt.Chart(st.session_state.data_values_season).mark_line().encode(
    x='Time',
    y=alt.Y('Rate', scale=alt.Scale(domain=[0, 500]), axis=alt.Axis(title='Rate (m³/s)'))
).properties(title=alt.TitleParams('Seasonality', anchor='middle', fontSize=24))

st.altair_chart(chart_season, use_container_width=True)



# Data generator: display trend
st.markdown("<h4 style='text-align: left;'>3. Trends</h4>", unsafe_allow_html=True)
st.markdown("A trend is introduced to create gradual upward or downward shifts in the rate. The trend has a **random direction** (up or down), a **random duration** between 20 and 80 steps, and a **magnitude** randomly scaled between 0.5 and 2. The trends occur with a **10% chance**.")


if 'data_values_trend' not in st.session_state:
    data_stream_trend = data_generation(flag_rate=False, flag_noise=False, flag_seasonality=False, flag_trend = True, flag_anomalies = False)
    st.session_state.data_values_trend = pd.DataFrame({'Time': range(800), 'Rate': [next(data_stream_trend) for _ in range(800)]}) 

chart_trend = alt.Chart(st.session_state.data_values_trend).mark_line().encode(
    x='Time',
    y=alt.Y('Rate', scale=alt.Scale(domain=[0, 500]), axis=alt.Axis(title='Rate (m³/s)'))
).properties(title=alt.TitleParams('Trend', anchor='middle', fontSize=24))

st.altair_chart(chart_trend, use_container_width=True)


# Data generator: display  anomalies
st.markdown("<h4 style='text-align: left;'>4. Anomalies</h4>", unsafe_allow_html=True)
st.markdown("Anomalies are introduced to create sudden, unexpected changes in the rate. Each anomaly has a **random value** between -50 and 50. Anomalies occur with a **5% chance**.")


if 'data_values_anomalies' not in st.session_state:
    data_stream_anomalies = data_generation(flag_rate=False, flag_noise=False, flag_seasonality=False, flag_trend = False, flag_anomalies = True)
    st.session_state.data_values_anomalies = pd.DataFrame({'Time': range(800), 'Rate': [next(data_stream_anomalies) for _ in range(800)]}) 

chart_anomalies = alt.Chart(st.session_state.data_values_anomalies).mark_line().encode(
    x='Time',
    y=alt.Y('Rate', scale=alt.Scale(domain=[0, 500]), axis=alt.Axis(title='Rate (m³/s)'))
).properties(title=alt.TitleParams('Anomalies', anchor='middle', fontSize=24))

st.altair_chart(chart_anomalies, use_container_width=True)




st.markdown("<h2 style='text-align: center;'>Algorithm</h2>", unsafe_allow_html=True)

st.markdown("""
**Choice:** LSTM
            
**Reason:**

I considered both **LSTM** and **ARIMA** due to their reputation for handling time series data. **ARIMA**, while useful for stationary linear data, struggles with non-stationary data, multiple seasonal trends, and concept drift. On the other hand, **LSTM** captures complex, non-linear patterns and long-term dependencies.""")

st.markdown("""
**Method:**

1. The LSTM model takes input from a sliding window to increase efficiency:  
   
   Processing manageable chunks of data instead of the entire sequence reduces memory usage and speeds up computations while still capturing temporal dependencies.  
   
   A window size of 100 time points was found to be optimal. This makes sense because a complete oscillation of the sine wave was set to 200 time points, and trends could only run for a maximum of 80 time points. The size of the sliding window allowed it to capture changing cyclical effects and new trends.
   
2. The LSTM model was trained using values from the data generator **without anomalies**:  
   
   Two LSTM layers were used, with 50 and 25 hidden units respectively, and dropout was introduced to avoid local minima.

3. The live generated data **with anomalies** is sent to the LSTM model with optimized parameters.

4. The predicted data value from the LSTM model is compared with the actual value from the data stream.

5. If the difference between the predicted value and actual value is above a certain threshold, the data point is flagged and considered either an anomaly or part of an irregular pattern.
""")


st.markdown("""
**Thoughts:**

Irregular patterns tend to be identified after an anomaly is detected, which leads me to believe that:
            
Either the anomaly has triggered the beginning of an irregular pattern, or the anomaly has skewed the model's predictions, causing the difference between the predicted value and actual value to exceed the threshold, therefore being flagged.

**Recommendations for further work:**

Consider further training of the LSTM model with anomalies included in the training data. The training data should contain a flag for each data point to indicate whether an anomaly occurred at that time point. If the anomaly flag is raised, the loss function should reward the model for predicting values far from the data point and penalize predictions that are too close to it.
""")

st.markdown("""
**FYI:**

The live data stream has 'slinding window' number of data values generated intially, so the anomaly detection algorithm can start running straight away.""")













# Title
#   - For UI, HTML is passed via markdown for custom styling of the title
#   - 'unsafe_allow_html', allows the use of raw HTML
st.markdown("<h2 style='text-align: center;'>Live Data Stream Anomaly Detection</h2>", unsafe_allow_html=True)




# Initilaise Streamlit session states
#   - Maintains internal states across user interactions 
#   - i.e. button clicked

# Flag for when data stream is generating
if 'stream_running' not in st.session_state:
    st.session_state.stream_running = False

# Empty DataFrame for data stream: time, rate, flag
if 'data_values' not in st.session_state:
    st.session_state.data_values = pd.DataFrame(columns=['Time', 'Rate', 'Anomaly_Flag'])





# Buttons to control live data stream
#   - For UI, the buttons are horizontally aligned via column insertions and centered via css styling
col1, col2 = st.columns(2)

st.markdown("""
    <style>
    div.stButton > button {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """, unsafe_allow_html=True)


# Start Button
#   - This button not only starts the live data stream but it also resets eveything for new data streams:
#   - stream_running flag set to True
#   - Resets data_values
#   - Resets time_point
#   - Resets data_stream to new data_generation instance
#
#   - Define LSTM Model that was chosen in research
#   - Load saved parameters from trained model
#
#   - Define anomaly threshold
with col1:
    if st.button('Start', key='start'):
        print('reset')
        st.session_state.stream_running = True
        
        st.session_state.data_values = pd.DataFrame(columns=['Time', 'Rate', 'Anomaly_Flag'])
        st.session_state.time_point = 0
        st.session_state.data_stream = data_generation(flag_rate=True, flag_noise=True, flag_seasonality=True, flag_trend=True, flag_anomalies=True)
        st.session_state.window = [next(st.session_state.data_stream) for _ in range(window_length)]


        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
                super(LSTMModel, self).__init__()
                self.lstm1 = nn.LSTM(input_size, hidden_size1, num_layers=1, batch_first=True)
                self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, num_layers=1, batch_first=True, dropout=0.2)  # Add dropout here
                self.fc = nn.Linear(hidden_size2, output_size)

            def forward(self, x):
                lstm_out1, _ = self.lstm1(x)
                lstm_out2, _ = self.lstm2(lstm_out1)
                out = self.fc(lstm_out2[:, -1, :])  # Take the last output
                return out
            
        model = LSTMModel(input_size=1, hidden_size1=50, hidden_size2=25, output_size=1)


        model.load_state_dict(torch.load('Algorithm4.pth'))
        model.eval()


        st.session_state.anomaly_threshold = 20
        st.session_state.scaler = joblib.load('scaler4.pkl')

        


# Stop Button
#   - Sets stream_running to False
with col2:
    if st.button('Stop', key='stop'):
        st.session_state.stream_running = False






# Placeholder for the chart
chart_placeholder = st.empty()





# Proces when data_stream is set to True
#   - Retrive new value from data generation
#   - Detect if data point is an anomaly
#   - Create DF of new values: Time, Rate, Anomaly Flag
#   - Update session data values with new values via concatenation
#
#   - Create line chart to plot Time and Rate values
#   - Record points where Anomaly_Flag is raised
#   - Layer anomaly points over line chart
#   - Display chart with responsive sizing
#
#   - Increment time_point by 1
#   - Pause process for 0.25 second

while st.session_state.stream_running:

    scaled_window = st.session_state.scaler.transform(np.array(st.session_state.window).reshape(-1, 1)).reshape(1, window_length, 1)
   

    with torch.no_grad():
        prediction = model(torch.tensor(scaled_window, dtype=torch.float32))


    
    unscaled_prediction = st.session_state.scaler.inverse_transform(prediction.detach().numpy())
    prediction = unscaled_prediction

    new_data = next(st.session_state.data_stream)


    
    
    st.session_state.window.pop(0)
    st.session_state.window.append(new_data)

    # Anomaly detection: compare predicted vs actual
    anomaly_flag = abs(prediction.item() - new_data) > st.session_state.anomaly_threshold

    new_row = pd.DataFrame({'Time': [st.session_state.time_point], 'Rate': [new_data], 'Anomaly_Flag': [anomaly_flag]})
    
    st.session_state.data_values = pd.concat([st.session_state.data_values, new_row], ignore_index=True)

    
    line = alt.Chart(st.session_state.data_values).mark_line().encode(
        x='Time',
        y=alt.Y('Rate', scale=alt.Scale(domain=[0, 500]), axis=alt.Axis(title='Rate (m³/s)')))
    
    points = alt.Chart(st.session_state.data_values[st.session_state.data_values['Anomaly_Flag'] == True]).mark_point(color='red').encode(
        x='Time',
        y=alt.Y('Rate', scale=alt.Scale(domain=[0, 500]), axis=alt.Axis(title='Rate (m³/s)')))

    
    chart = line + points
    chart = chart.properties(title=alt.TitleParams('Pipeline Gas Flow', anchor='middle', fontSize=32))
    chart_placeholder.altair_chart(chart, use_container_width=True)

    st.session_state.time_point += 1
    
    time.sleep(0.05)





# Proces when data_stream is set to False after it was previously True
#   - Used to keep chart after data generation has stopped
#
#   - Create line chart to plot Time and Rate values
#   - Record points where Anomaly_Flag is raised
#   - Layer anomaly points over line chart
#   - Display chart with responsive sizing

if not st.session_state.stream_running and not st.session_state.data_values.empty:
    
    line = alt.Chart(st.session_state.data_values).mark_line().encode(
        x='Time', 
        y=alt.Y('Rate', scale=alt.Scale(domain=[0, 500]), axis=alt.Axis(title='Rate (m³/s)')))
    
    points = alt.Chart(st.session_state.data_values[st.session_state.data_values['Anomaly_Flag'] == True]).mark_point(filled=True, color='red').encode(
        x='Time', 
        y=alt.Y('Rate', scale=alt.Scale(domain=[0, 500]), axis=alt.Axis(title='Rate (m³/s)')))
    
    chart = line + points
    chart = chart.properties(title=alt.TitleParams('Pipeline Gas Flow', anchor='middle', fontSize=32))
    chart_placeholder.altair_chart(chart, use_container_width=True)
