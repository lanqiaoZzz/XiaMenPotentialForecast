import pandas as pd
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.metrics.pairwise import euclidean_distances
from datetime import timedelta
from tools import read_data_table, write_result_table


def calculate_mean_curve(df, end_index):
    """
    计算均值曲线
    
    参数:
        df (DataFrame)
        end_index (int)
        
    返回:
        ndarray
    """
    # 提取指定范围的子集
    subset = df.iloc[:end_index + 1, 1:]
    
    # 计算列的均值
    mean_curve = subset.mean(axis=0).values
    
    return mean_curve

def calculate_max_envelope(df, start_index, end_index):
    """
    计算最大值包络线
    
    参数:
        df (DataFrame)
        start_index (int)
        end_index (int)
        
    返回:
        ndarray
    """
    # 提取指定范围的子集
    subset = df.iloc[start_index:end_index + 1, 1:]
    
    # 计算列的最大值
    max_envelope = subset.max(axis=0).values
    
    return max_envelope

def calculate_min_envelope(df, start_index, end_index):
    """
    计算最小值包络线
    
    参数:
        df (DataFrame)
        end_index (int)
        
    返回:
        ndarray
    """
    # 提取指定范围的子集
    subset = df.iloc[start_index:end_index + 1, 1:]
    
    # 计算列的最小值
    min_envelope = subset.min(axis=0).values
    
    return min_envelope

def calculate_quantile_curve(df, end_index, quantile=0.3):
    """
    计算分位数曲线
    
    参数:
        df (DataFrame)
        end_index (int)
        quantile (float): 分位数，默认 0.3
        
    返回:
        ndarray
    """
    # 提取指定范围的子集aa
    subset = df.iloc[:end_index + 1, 1:]
    
    # 计算列的分位数
    quantile_curve = subset.quantile(quantile, axis=0).values
    
    return quantile_curve

def calculate_typical_curve(df, start_index, end_index, n_clusters=3):
    """
    计算典型曲线
    
    参数:
        df (DataFrame)
        start_index (int)
        end_index (int)
        n_clusters (int)
        
    返回:
        ndarray
    """
    # 将所有数据提取为数组
    data = df.iloc[:end_index + 1, 1:].values  
    
    # 使用 KMedoids 聚类
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=42)
    kmedoids.fit(data)
    
    # 获取簇中心（典型曲线）
    typical_curves = kmedoids.cluster_centers_
    
    # 提取指定行范围内的数据
    subset = data[start_index:end_index + 1]
    
    # 遍历每条典型曲线，计算欧式距离的平均值
    min_avg_distance = float('inf')
    selected_curve = None
    
    for typical_curve in typical_curves:
        # 计算 subset 中每条曲线到当前典型曲线的欧式距离
        distances = euclidean_distances(subset, [typical_curve])
        avg_distance = np.mean(distances)
        
        # 更新最小平均距离和对应的典型曲线
        if avg_distance < min_avg_distance:
            min_avg_distance = avg_distance
            selected_curve = typical_curve
    
    return selected_curve

def calculate_baseline(order, write_to_db=True):
    """
    计算 order.date_end 后一天的负荷基准线
    并 根据行业特点推荐最合适的负荷基准线
    
    参数:
        order(dict)
        write_to_db(bool): 是否写入数据库
        
    返回:
        df_result / None
    """
    order_id = order.get('order_id')
    # user_id = order.get('user_id')
    user_type = order.get('user_type')
    date_end = order.get('date_end')
    df, data_point_flag = read_data_table(order_id, None, date_end)

    df.ffill(axis=0, inplace=True)

    date_end = pd.to_datetime(date_end)
    end_index = df[df['data_date'] == date_end.date()].index[0]    
    start_index = end_index - 15
    if start_index < 0:
        start_index = 0

    cal_day = (date_end + timedelta(days = 1)).strftime('%Y-%m-%d')

    if write_to_db is False:
        num_points = 96
        freq = '15min'
        if data_point_flag == 2:
            num_points = 48
            freq = '30min'
        elif data_point_flag == 3:
            num_points = 24
            freq = '1h'
        
        time_points = pd.date_range(start=f"{cal_day} 00:00", periods=num_points, freq=freq)

        df_result = pd.DataFrame({
            'ds': time_points,
            'mean': calculate_mean_curve(df, end_index),
            'max': calculate_max_envelope(df, start_index, end_index),
            'min': calculate_min_envelope(df, start_index, end_index),
            'quantile': calculate_quantile_curve(df, end_index),    
            'typical': calculate_typical_curve(df, start_index, end_index)
        })

        return df_result
    
    result = []
    for baseline_type, _ in [
        ('mean', calculate_mean_curve),
        ('max', calculate_max_envelope),
        ('min', calculate_min_envelope),
        ('quantile', calculate_quantile_curve),
        ('typical', calculate_typical_curve),
    ]:
        # 基线值
        if baseline_type == 'mean':
            baseline_values = calculate_mean_curve(df, end_index)
        elif baseline_type == 'max':
            baseline_values = calculate_max_envelope(df, start_index, end_index)
        elif baseline_type == 'min':
            baseline_values = calculate_min_envelope(df, start_index, end_index)
        elif baseline_type == 'quantile':
            baseline_values = calculate_quantile_curve(df, end_index)
        elif baseline_type == 'typical':
            baseline_values = calculate_typical_curve(df, start_index, end_index)
  
        # 构造每行数据
        row = {
            'order_id': order_id,
            'type': {'mean': 1, 'max': 2, 'min': 3, 'quantile': 4, 'typical': 5}[baseline_type],
            'date': cal_day,  
            'data_point_flag': data_point_flag,
        }
        for i in range(96):
            if i < len(baseline_values):
                row[f'p{i+1}'] = baseline_values[i]
            else:
                row[f'p{i+1}'] = None       
        row['recommended'] = False
        
        result.append(row)
    
    if user_type in [2, 7, 8]:
        # 储能 电梯 其他: 最大值包络线
        result[1]['recommended'] = True
    elif user_type in [4, 5]:
        # 照明 充电桩: 30%分位数曲线
        result[3]['recommended'] = True
    else:
        # 空调 光伏 生产: 最小值包络线
        result[2]['recommended'] = True
    
    write_result_table('baseline', result)