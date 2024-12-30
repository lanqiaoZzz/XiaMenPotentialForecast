import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from tools import read_data_table, write_result_table
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
    计算 [季节 月份阶段 是否为工作日 一天时段] 与负荷的相关性
    
    参数:
        order(dict)
        
    返回:
        None
    """
    order_id = order.get('order_id')
    user_id = order.get('user_id')
    date_start = order.get('date_start')
    date_end = order.get('date_end')
    df, data_point_flag = read_data_table(user_id, date_start, date_end)

    num_points = 96
    if data_point_flag == 2:
        num_points = 48
    elif data_point_flag == 3:
        num_points = 24
    
    df = build_dataset(df, num_points)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['Season'] = df['DateTime'].dt.month.apply(get_season)
    df['Month_Part'] = df['DateTime'].dt.day.apply(get_month_part)
    df['Is_Workday'] = df['DateTime'].dt.weekday < 5  
    df['Is_Workday'] = df['Is_Workday'].astype(int)  
    df['Time_of_Day'] = df['DateTime'].dt.hour.apply(get_time_of_day)

    df_load = df[['Load']]
    df_factors = df[['Season', 'Month_Part', 'Is_Workday', 'Time_of_Day']]

    cca = CCA(n_components=min(df_factors.shape[1], df_load.shape[1]))
    X_c, Y_c = cca.fit_transform(df_factors, df_load)
    correlations = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(X_c.shape[1])]
    x_weights = cca.x_weights_[:, 0]
    x_contributions = pd.Series(x_weights, index=df_factors.columns).abs().sort_values(ascending=False)

    result = []
    for factor, contribution in zip(df_factors.columns, x_contributions):
        row = {
            'order_id': order_id,
            'factor': factor,
            'contribution': contribution,
        }
        result.append(row)
    
    write_result_table('correlation', result)