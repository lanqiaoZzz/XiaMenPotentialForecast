from baseline import calculate_baseline
from forecast import short_term_load_forecast
import pandas as pd
import numpy as np
from datetime import timedelta
from tools import write_result_table

def calculate_potential(order):
    df_baseline = calculate_baseline(order, False)
    df_pred = short_term_load_forecast(order, False)

    df_baseline['pred'] = df_pred['pred_BP']

    df_potential = pd.DataFrame({
        'ds': df_baseline['ds'].values
    })

    for col in df_baseline.columns[1:-1]:
        df_potential[f'{col}'] = df_baseline['pred'] - df_baseline[col]
    
    return df_potential

def evaluate_potential(df_potential):
    evaluation_metrics = []
    baseline_sums = {}

    for col in df_potential.columns[1:]:
        potential_values = df_potential[col].values

        potential_sum = potential_values.sum()  
        potential_max = potential_values.max()   
        potential_min = potential_values.min()   
        potential_std = potential_values.std()   

        baseline_sums[col] = potential_sum

        evaluation_metrics.append({
            'baseline': col,
            'sum': potential_sum,
            'max': potential_max,
            'min': potential_min,
            'std': potential_std
        })

    df_metrics = pd.DataFrame(evaluation_metrics)

    normalized_metrics = df_metrics.copy()
    for col in ['sum', 'max', 'min', 'std']:
        normalized_metrics[col] = df_metrics[col] / np.sqrt((df_metrics[col]**2).sum())

    # 计算加权矩阵
    weights = {'sum': 0.25, 'max': 0.25, 'min': 0.25, 'std': 0.25}  # 等权法

    for col in weights:
        normalized_metrics[col] = normalized_metrics[col] * weights[col]

    # 计算理想解和负理想解
    positive_ideal = {
        'sum': normalized_metrics['sum'].max(),      # 收益型
        'max': normalized_metrics['max'].max(),      # 收益型
        'min': normalized_metrics['min'].max(),      # 成本型，取最大值（越大越好）
        'std': normalized_metrics['std'].min()       # 成本型
    }
    negative_ideal = {
        'sum': normalized_metrics['sum'].min(),      # 收益型
        'max': normalized_metrics['max'].min(),      # 收益型
        'min': normalized_metrics['min'].min(),      # 成本型
        'std': normalized_metrics['std'].max()       # 成本型
    }

    # 计算距离
    normalized_metrics['D+'] = np.sqrt(
        (normalized_metrics['sum'] - positive_ideal['sum'])**2 +
        (normalized_metrics['max'] - positive_ideal['max'])**2 +
        (normalized_metrics['min'] - positive_ideal['min'])**2 +
        (normalized_metrics['std'] - positive_ideal['std'])**2
    )
    normalized_metrics['D-'] = np.sqrt(
        (normalized_metrics['sum'] - negative_ideal['sum'])**2 +
        (normalized_metrics['max'] - negative_ideal['max'])**2 +
        (normalized_metrics['min'] - negative_ideal['min'])**2 +
        (normalized_metrics['std'] - negative_ideal['std'])**2
    )

    # 计算相对接近度
    normalized_metrics['C'] = normalized_metrics['D-'] / (normalized_metrics['D+'] + normalized_metrics['D-'])

    # 排序
    normalized_metrics.sort_values(by='C', ascending=False, inplace=True)

    df_result = pd.DataFrame({
        'baseline': normalized_metrics['baseline'],
        'sum': normalized_metrics['baseline'].map(baseline_sums),
        'score': normalized_metrics['C'],
    })

    df_result.reset_index(drop=True, inplace=True)

    return df_result
 
def calculate_and_evaluate_potential(order):
    order_id = order.get('order_id')
    date_end = order.get('date_end')
    
    df_potential = calculate_potential(order)
    df_result = evaluate_potential(df_potential)

    cal_day = (date_end + timedelta(days = 1)).strftime('%Y-%m-%d')

    result = []
    for i in range(len(df_result)):
        baseline = df_result.iloc[i]['baseline']
        sum = df_result.iloc[i]['sum']
        score = df_result.iloc[i]['score']
         
        row = {
            'order_id': order_id,
            'baseline_type': {'mean': 1, 'max': 2, 'min': 3, 'quantile': 4, 'typical': 5}[baseline],
            'date': cal_day,
            'sum': sum,
            'score': score,
        }

        result.append(row)
    
    write_result_table('potential', result)