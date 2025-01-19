import pymysql
from baseline import calculate_baseline
from cluster import evaluate_cluster, calculate_cluster_curve
from correlation import calculate_correlation
from forecast import forecast
from potential import calculate_and_evaluate_potential

from pydantic import BaseModel
from datetime import date, datetime
from fastapi import FastAPI
from typing import Optional
from tools import *


class OrderRequest(BaseModel):
    func_type: Optional[int] = None
    user_id: Optional[str] = None
    date_start: Optional[date] = None
    date_end: Optional[date] = None
    time_forecast: Optional[datetime] = None
    cluster_num: Optional[int] = None

  
app = FastAPI()

@app.post("/cluster/curve")
async def cluster_curve(order_req: OrderRequest):
    """
    聚类负荷曲线展示
    """
    # 检查 user_id 是否存在
    if check_user_id(order_req.user_id) == False:
        return {
            "success": False,
            "message": "关联用户不存在",
            "data": None
        }
    # 检查时间范围是否合法
    if (order_req.date_end - order_req.date_start).days + 1 < order_req.cluster_num + 1:
        return {
            "success": False,
            "message": "时间范围太小，无法进行聚类",
            "data": None
        }

    # 查询是否执行过相同的 order
    order_req.func_type = 2
    if order_req.cluster_num is None:
        order_req.cluster_num = 2
    order_id = get_order_id(order_req)

    if order_id is None:
        # 未执行过：插入 order 并执行
        order_id = insert_order_table(order_req)
        if order_id:
            order = read_orders_table(order_id)
            calculate_cluster_curve(order)
        else:
            return {
                "success": False,
                "message": "执行失败",
                "data": None
            }
    
    # 查询数据库
    table_name = 'cluster_curve'
    df_result = read_result_table(table_name, order_id)
    if df_result is None:
        # 存在 order 但未执行成功：再次执行
        order = read_orders_table(order_id)
        evaluate_cluster(order)
        df_result = read_result_table(table_name, order_id)

    if not df_result.empty:
        res = build_cluster_curve_res(df_result)
        return res
    else:
        return {
            "success": False,
            "message": "执行失败",
            "data": None
        }
    
@app.post("/cluster/evaluation")
async def cluster_evaluation(order_req: OrderRequest):
    """
    获得典型聚类指标
    """
    # 检查 user_id 是否存在
    if check_user_id(order_req.user_id) == False:
        return {
            "success": False,
            "message": "关联用户不存在",
            "data": None
        }
    # 检查时间范围是否合法
    if (order_req.date_end - order_req.date_start).days + 1 < 8:
        return {
            "success": False,
            "message": "时间范围太小，无法进行聚类",
            "data": None
        }

    # 查询是否执行过相同的 order
    order_req.func_type = 1
    order_id = get_order_id(order_req)

    if order_id is None:
        # 未执行过：插入 order 并执行
        order_id = insert_order_table(order_req)
        if order_id:
            order = read_orders_table(order_id)
            evaluate_cluster(order)
        else:
            return {
                "success": False,
                "message": "执行失败",
                "data": None
            }
    
    # 查询数据库
    table_name = 'cluster_evaluation'
    df_result = read_result_table(table_name, order_id)
    if df_result is None:
        # 存在 order 但未执行成功：再次执行
        order = read_orders_table(order_id)
        evaluate_cluster(order)
        df_result = read_result_table(table_name, order_id)

    if not df_result.empty:
        res = build_cluster_evaluation_res(df_result)
        return res
    else:
        return {
            "success": False,
            "message": "执行失败",
            "data": None
        }

@app.post("/factor")
async def factor(order_req: OrderRequest):
    """
    获得各种因素状态监控结果（温度、湿度、风速）
    """
    date_start = order_req.date_start
    date_end = order_req.date_end
    # 检查时间范围是否合法
    if (date_end - date_start).days + 1 < 10:
        return {
            "success": False,
            "message": "时间范围太小，无法进行相关性分析",
            "data": None
        }

    df_result = read_weather_table(None, date_end)
    if df_result is None:
        return {
            "success": False,
            "message": "执行失败",
            "data": None
        }
    else:
        res = build_factor_res(df_result)
        return res

@app.post("/correlation")
async def correlation(order_req: OrderRequest):
    """
    获得外部因素相关性分析结果（温度、湿度、风速）
    """
    # 检查 user_id 是否存在
    if check_user_id(order_req.user_id) == False:
        return {
            "success": False,
            "message": "关联用户不存在",
            "data": None
        }
    # 检查时间范围是否合法
    if (order_req.date_end - order_req.date_start).days + 1 < 10:
        return {
            "success": False,
            "message": "时间范围太小，无法进行相关性分析",
            "data": None
        }

    # 查询是否执行过相同的 order
    order_req.func_type = 3
    order_id = get_order_id(order_req)

    if order_id is None:
        # 未执行过：插入 order 并执行
        order_id = insert_order_table(order_req)
        if order_id:
            order = read_orders_table(order_id)
            calculate_correlation(order)
        else:
            return {
                "success": False,
                "message": "执行失败",
                "data": None
            }
    
    # 查询数据库
    table_name = 'correlation'
    df_result = read_result_table(table_name, order_id)
    if df_result is None:
        # 存在 order 但未执行成功：再次执行
        order = read_orders_table(order_id)
        calculate_correlation(order)
        df_result = read_result_table(table_name, order_id)

    if not df_result.empty:
        res = build_correlation_res(df_result)
        return res
    else:
        return {
            "success": False,
            "message": "执行失败",
            "data": None
        }


# if __name__ == '__main__':
#     order_id = 7
#     order = read_orders_table(order_id)

#     func_type = order.get('func_type')
#     if func_type == 1:
#         # 聚类指标评估与分析
#         evaluate_cluster(order)
#     elif func_type == 2:
#         # 聚类与分析
#         calculate_cluster_curve(order)
#     elif func_type == 3:
#         # 相关性分析
#         calculate_correlation(order)
#     elif func_type == 4:
#         # 负荷基准线计算与选择
#         calculate_baseline(order)
#     elif func_type == 5 or func_type == 6:
#         # 负荷预测
#         forecast(order, True)
#     elif func_type == 7:
#         # 负荷潜力计算与评估
#         calculate_and_evaluate_potential(order)