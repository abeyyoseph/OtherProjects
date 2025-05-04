import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

def ReadGpsData(gps_data_path):
    df = pd.read_csv(gps_data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
    
    df['x'] = df['predicted_ego_pos_x']
    df['y'] = df['predicted_ego_pos_y']
    df['z'] = df['predicted_ego_pos_z']

    return df

def InitializeKalmanFilter(df):
    # Initialize Kalman Filter with 6 states (x, y, z, v_x, v_y, v_z)
    kf = KalmanFilter(dim_x=6, dim_z=3)
    dt = df['time_diff'].median()

    # State transition matrix
    kf.F = np.array([
        [1, 0, 0, dt,  0,  0],
        [0, 1, 0,  0, dt,  0],
        [0, 0, 1,  0,  0, dt],
        [0, 0, 0,  1,  0,  0],
        [0, 0, 0,  0,  1,  0],
        [0, 0, 0,  0,  0,  1]
    ])

    # Measurement function
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0],  # x
        [0, 1, 0, 0, 0, 0],  # y
        [0, 0, 1, 0, 0, 0]   # z
    ])

    # Covariances
    kf.R *= 5  # Measurement noise
    kf.P *= 500  # Initial estimate uncertainty
    kf.Q = Q_discrete_white_noise(dim=3, dt=dt, var=0.1, block_size=2)

    # Initial state with velocities set to 0
    kf.x = np.array([df['x'].iloc[0], 
                     df['y'].iloc[0], 
                     df['z'].iloc[0], 
                     0,
                     0,
                     0])

    return kf

def FilterData(gps_df, kalman_filter):
    # Apply filter to each measurement
    filtered = []
    for i in range(len(gps_df)):
        z = np.array([gps_df['x'].iloc[i], gps_df['y'].iloc[i], gps_df['z'].iloc[i]])
        kalman_filter.predict()
        kalman_filter.update(z)
        filtered.append(kalman_filter.x[:3])  # Only keep position

    return filtered

def PlotFilteredData(filtered_data, df):
    filtered_data_np = np.array(filtered_data)
    df['filtered_x'], df['filtered_y'], df['filtered_z'] = filtered_data_np[:, 0], filtered_data_np[:, 1], filtered_data_np[:, 2]

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(df['x'], df['y'], df['z'], label='Raw GPS', alpha=0.5)
    ax.plot(df['filtered_x'], df['filtered_y'], df['filtered_z'], label='Kalman Filtered', linewidth=2)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title("Kalman Filter Smoothing on GPS Data (3D)")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    gps_data_path = "test.csv"
    gps_df = ReadGpsData(gps_data_path)
    kalman_filter = InitializeKalmanFilter(gps_df)
    filtered_data = FilterData(gps_df, kalman_filter)
    PlotFilteredData(filtered_data, gps_df)