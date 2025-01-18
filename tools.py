import pymysql
import pandas as pd
from decimal import Decimal
from config import *
from decimal import Decimal


def convert_decimal_to_float(df):
    return df.applymap(lambda x: float(x) if isinstance(x, Decimal) else x)

def read_data_table(user_id, date_start, date_end):
    """
    读 xia_men 数据库中存储数据的表
    
    """
    connection = pymysql.connect(**config_data)

    with connection.cursor() as cursor:
        if date_start is None:
            query = f"select * from dr_cons_curve where cons_id = '{user_id}' and data_date <= '{date_end}';"
        else:
            query = f"select * from dr_cons_curve where cons_id = '{user_id}' and data_date >= '{date_start}' and data_date <= '{date_end}';"
        
        cursor.execute(query)

        results = cursor.fetchall()

        if not results:
            return None, None

        columns = [desc[0] for desc in cursor.description]  # 字段名列表

        df = pd.DataFrame(results, columns=columns)

        columns_pi = []
        for i in range(1, 97):
            columns_pi.append('p' + str(i))
        df[columns_pi] = df[columns_pi].astype(float)

        columns_selected = ['data_date'] + columns_pi
        
        return df[columns_selected], int(df['data_point_flag'][0])

def check_user_id(user_id):
    connection = pymysql.connect(**config_data)

    with connection.cursor() as cursor:
        query = f"select * from dr_cons_curve where cons_id = '{user_id}';"
        
        cursor.execute(query)

        results = cursor.fetchall()

        if not results:
            return False
        else:
            return True

def read_weather_table(date_end):
    connection = pymysql.connect(**config_weather)

    with connection.cursor() as cursor:
        query = f"select * from vp_weather_real where DATEDIFF('{date_end}', data_date) < 10 and weather_type in ('WS', 'RHU', 'T');"
        
        cursor.execute(query)

        results = cursor.fetchall()

        if not results:
            return None

        columns = [desc[0] for desc in cursor.description]  
        df = pd.DataFrame(results, columns=columns)

        columns_pi = []
        for i in range(1, 25):
            columns_pi.append('p' + str(i))
        df[columns_pi] = df[columns_pi].astype(float)

        columns_selected = ['weather_type'] + columns_pi
        
        return df[columns_selected]

def read_orders_table(order_id):
    """
    读 xiamen_output 数据库中存储命令的表
    
    """
    connection = pymysql.connect(**config_result)

    with connection.cursor() as cursor:
        query = f"select * from orders where order_id = {order_id};"
        
        cursor.execute(query)

        result = cursor.fetchone()
        columns = [desc[0] for desc in cursor.description]  # 字段名列表

        order_dict = dict(zip(columns, result))

        return order_dict

def get_order_id(order):
    conditions = []
    params = []

    if order.func_type:
        conditions.append("func_type = %s")
        params.append(order.func_type)
    if order.user_id:
        conditions.append("user_id = %s")
        params.append(order.user_id)
    if order.date_start:
        conditions.append("date_start = %s")
        params.append(order.date_start)
    if order.date_end:
        conditions.append("date_end = %s")
        params.append(order.date_end)
    if order.time_forecast:
        conditions.append("time_forecast = %s")
        params.append(order.time_forecast)
    if order.cluster_num:
        conditions.append("cluster_num = %s")
        params.append(order.cluster_num)

    query = "SELECT order_id FROM orders WHERE " + " AND ".join(conditions)

    connection = pymysql.connect(**config_result)

    try:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            result = cursor.fetchone()
            if result:
                return result[0]
            else:
                return None
    except Exception as e:  
        print(f"Error selecting data: {e}")
    finally:
        connection.close()

def insert_order_table(order):
    fields = vars(order)

    columns = ', '.join(fields.keys())  
    values = ', '.join(['%s'] * len(fields))  
    sql_insert = f"INSERT INTO orders ({columns}) VALUES ({values})"

    connection = pymysql.connect(**config_result)

    try:
        with connection.cursor() as cursor:
            cursor.execute(sql_insert, tuple(fields.values()))
            connection.commit() 
            order_id = cursor.lastrowid
            return order_id
    except Exception as e:  
        print(f"Error inserting data: {e}")
        connection.rollback()
        return None
    finally:
        connection.close()

def read_result_table(table_name, order_id):
    query = f"SELECT * FROM {table_name} WHERE order_id = '{order_id}'"

    connection = pymysql.connect(**config_result)

    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchall()
            if result:
                columns = [desc[0] for desc in cursor.description]
                df_result = pd.DataFrame(result, columns=columns)
                return df_result
            else:
                return None         
    except Exception as e:  
        print(f"Error selecting data: {e}")
    finally:
        connection.close()

def write_result_table(table_name, result):
    """
    向 xiamen_output 数据库中的表写入数据
    
    """
    connection = pymysql.connect(**config_result)

    try:
        with connection.cursor() as cursor:
            values_placeholder = ', '.join(['%s'] * len(result[0]))
            insert_sql = f"""
                        insert into {table_name}
                        values ({values_placeholder})
                        """

            for row in result:           
                data = [Decimal(value) if isinstance(value, float) else value for value in row.values()]

                cursor.execute(insert_sql, data)

        connection.commit()
    except Exception as e:
        connection.rollback()  
        print(f"Error inserting data: {e}")
    finally:
        connection.close()  

def update_forecast_table(order_id, accuracy):
    """
    更新 xiamen_output 数据库中的 forecast 表
    
    """
    connection = pymysql.connect(**config_result)

    try:
        with connection.cursor() as cursor:            
            update_sql = f"""
                        update forecast
                        set accuracy = {accuracy}
                        where order_id = '{order_id}'
                        """

            cursor.execute(update_sql)

        connection.commit()
    except Exception as e:
        connection.rollback()  
        print(f"Error updating data: {e}")
    finally:
        connection.close() 

def build_cluster_curve_res(df_result):
    df_result = convert_decimal_to_float(df_result)

    data_point_flag = int(df_result['data_point_flag'].iloc[0])  
    curves = []

    for _, row in df_result.iterrows():
        curve = {
            'cluster_id': int(row['cluster_id']),
        }
        for col in df_result.columns:
            if col.startswith('p'):
                curve[col] = row[col]
        curves.append(curve)
    
    return {
        'success': True,
        'message': '',
        'data': {
            'data_point_flag': data_point_flag,
            'curves': curves
        }
    }

def build_cluster_evaluation_res(df_result):
    df_result = convert_decimal_to_float(df_result)
    
    data = []
    
    for index, row in df_result.iterrows():
        if 1 <= index <= 6:
            data.append({
                'cluster_num': int(row['cluster_num']),
                'silhouette': row['silhouette'],
                'calinski_harabasz': row['calinski_harabasz'],
                'davies_bouldin': row['davies_bouldin']
            })
        
    return {
        'success': True,
        'message': '',
        'data': data
    }

def build_factor_res(df_result):
    df_result = convert_decimal_to_float(df_result)
    
    data = [
        {
            'factor': 1, 
            'values': df_result[df_result['weather_type'] == 'WS'].iloc[:, 1:].values.flatten().tolist()
        },
        {
            'factor': 4, 
            'values': df_result[df_result['weather_type'] == 'RHU'].iloc[:, 1:].values.flatten().tolist()
        },
        {
            'factor': 6, 
            'values': df_result[df_result['weather_type'] == 'T'].iloc[:, 1:].values.flatten().tolist()
        }
    ]
        
    return {
        'success': True,
        'message': '',
        'data': data
    }