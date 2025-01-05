import pymysql
import pandas as pd
from decimal import Decimal
from config import config_data, config_result


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