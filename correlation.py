import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from tools import read_data_table, write_result_table, read_weather_table
from forecast import build_dataset

def get_season(month):
    if month in [3, 4, 5]:
        return 1  # spring
    elif month in [6, 7, 8]:
        return 2  # summer
    elif month in [9, 10, 11]:
        return 3  # autumn
    else:
        return 4  # winter

def get_month_part(day):
    if day <= 10:
        return 1  # beginning
    elif day <= 20:
        return 2  # middle
    else:
        return 3  # end

def get_time_of_day(hour):
    if 6 <= hour < 9:
        return 1  # morning
    elif 9 <= hour < 12:
        return 2  # noon
    elif 12 <= hour < 18:
        return 3  # afternoon
    else:
        return 4  # evening

def calculate_correlation(order):
    """
    计算 [温度 湿度 风速] 与负荷的相关性
    
    参数:
        order(dict)
        
    返回:
        None
    """
    order_id = order.get('order_id')
    date_start = order.get('date_start')
    date_end = order.get('date_end')
    user_id = order.get('user_id')
    df, data_point_flag = read_data_table(user_id, date_start, date_end)

    num_points = 96
    if data_point_flag == 2:
        num_points = 48
    elif data_point_flag == 3:
        num_points = 24
    
    df = build_dataset(df, num_points)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    df = df.resample('D').apply(lambda x: x.loc[x.index.minute == 0])
    
    df_weather = read_weather_table(date_start, date_end)
    df['RHU'] = df_weather[df_weather['weather_type'] == 'RHU'].iloc[:, 1:].values.flatten()
    df['T'] = df_weather[df_weather['weather_type'] == 'T'].iloc[:, 1:].values.flatten()
    df['WS'] = df_weather[df_weather['weather_type'] == 'WS'].iloc[:, 1:].values.flatten()

    df_load = df[['Load']]
    df_factors = df[['T', 'RHU', 'WS']]
    cca = CCA(n_components=min(df_factors.shape[1], df_load.shape[1]))
    X_c, Y_c = cca.fit_transform(df_factors, df_load)
    correlations = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(X_c.shape[1])]
    x_weights = cca.x_weights_[:, 0]
    x_contributions = pd.Series(x_weights, index=df_factors.columns).abs().sort_values(ascending=False)
    
    column_mapping = {
        'T': '温度',
        'RHU': '湿度',
        'WS': '风速',
    }
    result = []
    for factor, contribution in zip(df_factors.columns, x_contributions):
        factor_name = column_mapping.get(factor, factor) 
        row = {
            'order_id': order_id,
            'factor': factor_name,
            'contribution': contribution,
        }
        result.append(row)
    
    write_result_table('correlation', result)