import numpy as np
import math



# Varibale used in both Visuals.py and Algorithm.ipynb
# Set here to keep consistency 
window_length = 100

# The Data Generator geenrates values called the 'rate (m^3/s)', the function consists of:
#   - A range of 0-500
#   - The 'while True' and 'yield rate' structure continuously generates and outputs new data
#   - Flag Rate: if True starting rate is random, if False rate set to 250 (for demo purposes)
#   - Flag Noise: if True a Guassian mask is applied to the rate for general noise
#   - Flag Seasonality: if True a Sin function is applied to the rate to casue seasonal cycles
#   - Flag Trend: if True trends can occur with a 10% chance, if a trend occurs the direcition can be upwards or downwards, the duration of the trend is in the range 20-80 time points, and wihtin each active time step the rate can be increased/decreased +-0.5 to +-2
#   - Flag Anomalies: if True anomalies can occur with a 5% chance, if an anomaly occurs it can can decrease the rate -0.00001 to -50 or increase the rate +0.00001 to +50

def data_generation(flag_rate, flag_noise, flag_seasonality, flag_trend, flag_anomalies):
    min_rate = 0
    max_rate = 500
    time_step = 0
    

    if flag_rate:
        rate = np.random.uniform(min_rate, max_rate)
    else:
        rate = 250

    if flag_trend:
            trend_active = False
            trend_duration = 0
            trend_direction = 0


    
    while True:
        
        if flag_noise:
            noise = np.random.normal(0,1.5)
            rate = np.clip(rate + noise, min_rate, max_rate)

        if flag_seasonality:
            seasonal_component = 4 * math.sin(2 * math.pi * (time_step / 200) + math.pi/2)
            rate = np.clip(rate + seasonal_component, min_rate, max_rate)

        if flag_trend and not trend_active and np.random.random() < 0.1:
            trend_direction = np.random.choice([-1, 1])
            trend_duration = np.random.randint(20, 80)
            trend_active = True

        if flag_trend and trend_active:
            trend = trend_direction * np.random.uniform(0.5, 2)
            rate = np.clip(rate + trend, min_rate, max_rate)
            trend_duration -= 1
            if trend_duration == 0:
                trend_active = False

        if flag_anomalies and np.random.random() < 0.05:
            anomaly = np.random.uniform(-50, 50)
            rate = np.clip(rate + anomaly, min_rate, max_rate)


        yield rate
        time_step += 1