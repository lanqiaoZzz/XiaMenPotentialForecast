import pymysql
from baseline import calculate_baseline
from cluster import evaluate_cluster, calculate_cluster_curve
from correlation import calculate_correlation
from forecast import forecast
from potential import calculate_and_evaluate_potential


def read_orders_table(order_id):
    """
    读 xiamen_output 数据库中存储命令的表
    
    """
    config = {
        'host': 'localhost',          # 数据库主机地址
        'user': 'root',               # 数据库用户名      
        'password': 'w20020309',      # 数据库密码
        'database': 'xiamen_output',  # 数据库名称
        'port': 3306,                 # 数据库端口，默认3306
        'charset': 'utf8mb4'          # 指定字符集
    }

    connection = pymysql.connect(**config)

    with connection.cursor() as cursor:
        query = f"select * from orders where order_id = {order_id};"
        
        cursor.execute(query)

        result = cursor.fetchone()
        columns = [desc[0] for desc in cursor.description]  # 字段名列表

        order_dict = dict(zip(columns, result))

        return order_dict


if __name__ == '__main__':
    order_id = 6
    order = read_orders_table(order_id)

    func_type = order.get('func_type')
    if func_type == 1:
        # 聚类指标评估与分析
        evaluate_cluster(order)
    elif func_type == 2:
        # 聚类与分析
        calculate_cluster_curve(order)
    elif func_type == 3:
        # 相关性分析
        calculate_correlation(order)
    elif func_type == 4:
        # 负荷基准线计算与选择
        calculate_baseline(order)
    elif func_type == 5 or func_type == 6:
        # 负荷预测
        forecast(order, True)
    elif func_type == 7:
        # 负荷潜力计算与评估
        calculate_and_evaluate_potential(order)