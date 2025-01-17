from loguru import logger
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import pairwise_distances, silhouette_samples, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn_extra.cluster import KMedoids
from tools import read_data_table, write_result_table


class DataFrameUtil:
    @staticmethod
    def transform_date(df, date_column="date"):
        """
        将日期列转换为 datetime 格式，并添加季度、月份、每月的某天和星期列。

        Parameters:
            df (DataFrame): DataFrame。
            date_column (str, optional): 日期列的名称。默认为 "date"。

        Returns:
            DataFrame: 转换后的 DataFrame。
        """
        # 将日期对应列改为datatime格式
        df[date_column] = pd.to_datetime(df[date_column])
        # 添加季度列
        df["quarter"] = df[date_column].dt.quarter
        # 添加月份列
        df["month"] = df[date_column].dt.month
        # 添加每个月的某天列
        df["day"] = df[date_column].dt.day
        # 添加星期列
        df["weekday"] = df[date_column].dt.weekday

        return df

    @staticmethod
    def data_scaling(df, axis=1):
        """
        对 DataFrame 进行归一化。

        Parameters:
            df (DataFrame): DataFrame。
            axis (int, optional): 归一化的轴向。默认为 1（行归一化）。

        Returns:
            DataFrame: 归一化后的 DataFrame。
        """
        # 创建MinMaxScaler对象
        scaler = MinMaxScaler()

        # 对数据进行归一化
        if axis == 0:
            # 对列进行归一化
            scaled_data = scaler.fit_transform(df)
        else:
            # 对行进行归一化
            scaled_data = scaler.fit_transform(df.T).T

        # 将归一化后的数据转换为DataFrame对象并返回
        return pd.DataFrame(scaled_data, columns=df.columns, index=df.index)

    @staticmethod
    def data_preprocessing_algo(
        df, date_col_name, id_col_name=None, year=None, weekdays=0, begin=1, end=None
    ):
        """
        数据预处理，包括日期转换、数据切片、数据归一化。

        Parameters:
            df (DataFrame): DataFrame。
            date_col_name (str): 日期列的名称。
            id_col_name (str): ID 列的名称。
            year (int): 年份。
            weekdays (int): 工作日切片标志。0：不切片；1：工作日；2：周末。
            begin (str): 起始日期。
            end (str): 结束日期。

        Returns:
            DataFrame: 预处理后的 DataFrame。
        """
        df = df.dropna(axis=0, how="any")
        df = df[(df == 0).sum(axis=1) < 10]

        logger.info(
            "当前数据集含有%s行,%s列" % (df.shape[0], df.shape[1])
        )  # 查询数据集规模
        df = DataFrameUtil.transform_date(df, date_col_name)  # 转换日期
        logger.info(
            "数据集时间范围：%s - %s"
            % (df[date_col_name].min(), df[date_col_name].max())
        )  # 查询数据集时间

        if id_col_name != None:
            logger.info(
                "数据集共有%s个用户" % len(df[id_col_name].unique())
            )  # 查询数据集用户数
        else:
            logger.info("数据集只有1个用户")
        logger.info("数据集缺失值：\n%s" % (df.isnull().sum()))  # 查询数据集缺失值
        df = df.replace("", np.nan)

        if id_col_name != None:
            df = df.sort_values(
                by=[id_col_name, date_col_name], ascending=[True, True]
            )  # 按照id和日期进行重新排序
        else:
            df = df.sort_values(
                by=date_col_name, ascending=True
            )  # 按照日期进行重新排序

        if year:  # 按年份切片
            df = df[df[date_col_name].dt.year == year]
        if weekdays == 1:  # 按工作日切片
            df = df[df["weekday"] < 5]
        elif weekdays == 2:
            df = df[df["weekday"] >= 5]

        return df, df.iloc[:, begin:end]


def calculate_dunn_index(data, labels):
    """计算给定数据和标签的 Dunn Index。"""
    distances = squareform(pdist(data))
    unique_clusters = np.unique(labels)
    min_intercluster_distance = np.inf
    max_intracluster_distance = 0

    # 计算最小类间距离
    for i in unique_clusters:
        for j in unique_clusters:
            if i != j:
                inter_dist = distances[np.ix_(labels == i, labels == j)].min()
                if inter_dist < min_intercluster_distance:
                    min_intercluster_distance = inter_dist

    # 计算最大类内距离
    for k in unique_clusters:
        intra_dist = distances[labels == k, :][:, labels == k].max()
        if intra_dist > max_intracluster_distance:
            max_intracluster_distance = intra_dist

    # 计算 Dunn Index
    if max_intracluster_distance == 0:
        return np.inf  # 避免除以零
    return min_intercluster_distance / max_intracluster_distance

def correlation_distance(x, y, method='pearson'):
    """计算相关性距离，支持皮尔逊或斯皮尔曼方法"""
    if method == 'pearson':
        try:
            return 1 - pearsonr(x, y)[0]
        except ValueError:
            return 0
    elif method == 'spearman':
        return 1 - spearmanr(x, y)[0]

class PearsonKMedoids:
    def __init__(self, data, normalize=True, data_name="", folder_path="results"):
        """初始化方法，设置数据和归一化选项"""
        # 统计每个聚类簇的中心点
        self.means = []
        self.data = np.array(data)
        self.normalize = normalize
        self.fitdata = DataFrameUtil.data_scaling(data, int(normalize)) if normalize else self.data
        self.metrics = ['euclidean',
                         'manhattan',
                           lambda x, y: correlation_distance(x, y, 'pearson'),
                        lambda x, y: correlation_distance(x, y, 'spearman')]
        self.metrics_enum = ['euclidean', 'manhattan', 'pearson', 'spearman']
        self.data_name = data_name
        self.last_folder_path = f"{folder_path}/{self.data_name}/"

    def cluster_metrics(self, n_clusters, metric_label=0, multi_core=False):
        """
        计算聚类指标并保存为CSV文件。

        此方法针对不同聚类数计算以下指标，并将结果保存到CSV文件中：
        - 聚类数（从1到n_clusters-1）
        - 畸变程度（Distortion）：即聚类内样本与其质心的平均距离的平方和，较小的值表明聚类内部紧密。
        - 轮廓系数（Silhouette Score）：衡量聚类的紧密性与分离度。值范围在-1到1之间，接近1表示聚类效果较好。
        - Calinski-Harabasz 指数（Calinski-Harabasz Index）：该指数也称为方差比率准则，值越大表明聚类效果越好。
        - Davies-Bouldin 指数（Davies-Bouldin Index）：衡量类内距离与类间距离的比率。值越小，表示聚类效果越好。
        - Dunn 指数（Dunn Index）：最小的类间距离与最大的类内距离的比值。值越大，表示聚类效果越好。
        - 简化轮廓宽度准则（Simplified Silhouette Width Criterion, SWC）：基于轮廓系数，但计算平均轮廓系数与每个样本的轮廓系数差的绝对值的平均。值越小，聚类效果可能越好。

        参数：
        - n_clusters (int): 要测试的最大聚类数。
        - path (str): 结果CSV文件的保存路径。
        - metric_label (int or callable): 使用的距离度量标签或自定义函数。
        """
        metrics_df = pd.DataFrame()
        for i in range(1, n_clusters):
            # 计算距离矩阵，使用多核加速
            if multi_core:
                distance_matrix = pairwise_distances(self.fitdata, metric=self.metrics[metric_label], n_jobs=-1)
                km = KMedoids(n_clusters=i, init='k-medoids++', max_iter=300, random_state=0, metric='precomputed')
                km.fit(distance_matrix)
            else:
                km = KMedoids(n_clusters=i, init='k-medoids++', max_iter=300, random_state=0, metric=self.metrics[metric_label])
                km.fit(self.fitdata)
            labels = km.labels_
            metrics_df.loc[i, 'Distortion'] = km.inertia_
            metrics_df.loc[i, 'Silhouette'] = silhouette_score(
                self.fitdata, labels) if len(set(labels)) > 1 else None
            metrics_df.loc[i, 'Calinski-Harabasz'] = calinski_harabasz_score(
                self.fitdata, labels) if len(set(labels)) > 1 else None
            metrics_df.loc[i, 'Davies-Bouldin'] = davies_bouldin_score(
                self.fitdata, labels) if len(set(labels)) > 1 else None
            if len(set(labels)) > 1:
                metrics_df.loc[i, 'Dunn Index'] = calculate_dunn_index(self.fitdata, labels)
                metrics_df.loc[i, 'SWC'] = np.mean(
                    [silhouette_score(self.fitdata, labels) - abs(s) for s in silhouette_samples(self.fitdata, labels)])
            else:
                metrics_df.loc[i, 'Dunn Index'] = None
                metrics_df.loc[i, 'SWC'] = None
        # 空值填充0 所有数据保留三位小数
        metrics_df = metrics_df.fillna(0)
        metrics_df = metrics_df.round(3)
        return metrics_df

    def fit(self, n_clusters, metric_label, multi_core=False):
        """根据指定的聚类数和距离度量对数据进行聚类"""
        self.n_clusters = n_clusters
        if multi_core:
            distance_matrix = pairwise_distances(self.fitdata, metric=self.metrics[metric_label], n_jobs=-1)
            self.KMedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', init='k-medoids++', max_iter=300, random_state=0,)
            self.predictions = self.KMedoids.fit_predict(distance_matrix)
        # 计算距离矩阵，使用多核加速
        else:
            self.KMedoids = KMedoids(n_clusters=n_clusters, metric=self.metrics[metric_label], init='k-medoids++', max_iter=300, random_state=0)
            self.predictions = self.KMedoids.fit_predict(self.fitdata) + 1

    def get_cluster_counts(self):  # 统计聚类簇和每个簇中样本的数量
        return pd.Series(self.predictions).value_counts()

    def get_cluster(self):
        return self.predictions


def evaluate_cluster(order):
    """
    评估聚类数为 1-7 时聚类的效果

    参数:
        order(dict)
        
    返回:
        None
    """
    order_id = order.get('order_id')
    date_start = order.get('date_start')
    date_end = order.get('date_end')
    user_id = order.get('user_id')
    df_resource, _ = read_data_table(user_id, date_start, date_end)

    # df_resource, _ = read_data_table(order_id, date_start, date_end)
    
    df_resource, df_data = DataFrameUtil.data_preprocessing_algo(df_resource, "data_date", None, None, 0, 1, 97)  

    evaluator = PearsonKMedoids(df_data)
    metrics = evaluator.cluster_metrics(n_clusters=8, metric_label=1)

    result = []
    for i in range(len(metrics)):
        metric = metrics.iloc[i]  
        row = {
            'order_id': order_id,
            'cluster_num': i + 1,
            'distortion': metric.values[0],
            'silhouette': metric.values[1],
            'calinski_harabasz': metric.values[2],
            'davies_bouldin': metric.values[3],
            'dunn_index': metric.values[4],
            'swc': metric.values[5]
        }
        result.append(row)

    write_result_table('cluster_evaluation', result)


def calculate_cluster_curve(order):   
    """
    计算在指定聚类簇数目时各簇的平均负荷曲线

    参数:
        order(dict)
        
    返回:
        None
    """
    order_id = order.get('order_id')
    date_start = order.get('date_start')
    date_end = order.get('date_end')
    cluster_num = order.get('cluster_num')
    if cluster_num is None:
        cluster_num = 2
    user_id = order.get('user_id')
    df_resource, data_point_flag = read_data_table(user_id, date_start, date_end)

    # df_resource, data_point_flag = read_data_table(order_id, date_start, date_end)
    
    df_resource, df_data = DataFrameUtil.data_preprocessing_algo(df_resource, "data_date", None, None, 0, 1, 97)  

    evaluator = PearsonKMedoids(df_data)
    evaluator.fit(n_clusters=cluster_num, metric_label=1)

    cluster_data = {i: evaluator.data[evaluator.predictions == i] for i in range(1, evaluator.n_clusters + 1)}
    result = []
    for i, data in cluster_data.items():
        mean_profile = data.mean(axis=0)

        row = {
            'order_id': order_id,
            'cluster_id': i,
            'data_point_flag': data_point_flag,
        }
        for i in range(96):
            if i < len(mean_profile):
                row[f'p{i+1}'] = mean_profile[i]
            else:
                row[f'p{i+1}'] = None  

        result.append(row)
    
    write_result_table('cluster_curve', result)