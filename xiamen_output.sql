CREATE DATABASE xiamen_output;
USE xiamen_output;

-- orders表: 命令表，存储执行功能的命令
CREATE TABLE orders (
    order_id INT AUTO_INCREMENT PRIMARY KEY,    -- 命令ID
    func_type INT NOT NULL,                     -- 功能类型: 1 – 聚类指标评估与分析，2 – 聚类与分析，3 - 相关性分析，4 – 负荷基准线计算与选择
    user_id VARCHAR(50) NOT NULL,               -- 用户ID，与 dr_cons_curve 表中的 cons_id 对应
    date_start DATE,                            -- 数据待分析的起始日期，YYYY-MM-DD
    date_end DATE,                              -- 数据待分析的结束日期，YYYY-MM-DD
    time_forecast DATETIME,                     
    user_type INT,                              -- 用户类型: 1 - 空调，2 - 储能，3 - 光伏，4 - 照明，5 - 充电桩，6 - 生产，7 - 电梯，8 - 其他
    cluster_num INT,                            -- 聚类簇数，默认为2
    window_size INT                             -- 预测采用的历史数据长度，超短期预测默认为 1 * 数据日点数，短期预测默认为 30 * 数据日点数
);

-- INSERT INTO orders VALUES (1, 1, '3500520081112', '2023-06-16', '2024-10-29', NULL, NULL, NULL);
INSERT INTO orders VALUES (2, 2, '3500520081112', '2023-06-16', '2024-10-29', NULL, NULL, 3, NULL);
-- INSERT INTO orders VALUES (3, 3, '3500520081112', '2023-06-16', '2024-10-29', NULL, 3, NULL);
-- INSERT INTO orders VALUES (4, 4, '3500520081112', '2023-06-16', '2024-10-29', 5, NULL, NULL);
-- INSERT INTO orders VALUES (5, 5, '3500520081112', '2024-09-16', '2024-10-29', 5, NULL, 96);
INSERT INTO orders VALUES (6, 6, '3500520081112', NULL, NULL '2024-10-29 00:00:00', 5, NULL, 96);
-- INSERT INTO orders VALUES (7, 7, '3500520081112', '2024-09-16', '2024-10-29', 5, NULL, 96);

-- cluster_evaluation表: 结果表，存储聚类指标评估与分析的结果
CREATE TABLE cluster_evaluation (
    order_id INT NOT NULL,                         
    cluster_num INT NOT NULL,                      -- 聚类数
    distortion DECIMAL(10, 2),                     -- 畸变程度
    silhouette DECIMAL(10, 2),                     -- 轮廓系数
    calinski_harabasz DECIMAL(10, 2),              -- Calinski-Harabasz指数
    davies_bouldin DECIMAL(10, 2),                 -- Davies-Bouldin指数
    dunn_index DECIMAL(10, 2),                     -- Dunn指数
    swc DECIMAL(10, 2),                            -- 简化轮廓宽度准则
    PRIMARY KEY (order_id, cluster_num)            -- 联合主键，确保同一订单和聚类数的唯一性
);

-- cluster_curve表: 结果表，存储聚类与分析的结果（各簇平均负荷曲线）
CREATE TABLE cluster_curve (
    order_id INT NOT NULL,           
    cluster_id INT NOT NULL,            -- 聚类簇id
    data_point_flag INT,                -- 数据点数标志：1 - 96点，2 - 48点，3 - 24点
    p1 DECIMAL(10, 2),                  -- 第1个时间点簇平均负荷曲线的值
    p2 DECIMAL(10, 2),                  
    p3 DECIMAL(10, 2),                  
    p4 DECIMAL(10, 2),                  
    p5 DECIMAL(10, 2),                  
    p6 DECIMAL(10, 2),
    p7 DECIMAL(10, 2),
    p8 DECIMAL(10, 2),
    p9 DECIMAL(10, 2),
    p10 DECIMAL(10, 2),
    p11 DECIMAL(10, 2),
    p12 DECIMAL(10, 2),
    p13 DECIMAL(10, 2),
    p14 DECIMAL(10, 2),
    p15 DECIMAL(10, 2),
    p16 DECIMAL(10, 2),
    p17 DECIMAL(10, 2),
    p18 DECIMAL(10, 2),
    p19 DECIMAL(10, 2),
    p20 DECIMAL(10, 2),
    p21 DECIMAL(10, 2),
    p22 DECIMAL(10, 2),
    p23 DECIMAL(10, 2),
    p24 DECIMAL(10, 2),
    p25 DECIMAL(10, 2),
    p26 DECIMAL(10, 2),
    p27 DECIMAL(10, 2),
    p28 DECIMAL(10, 2),
    p29 DECIMAL(10, 2),
    p30 DECIMAL(10, 2),
    p31 DECIMAL(10, 2),
    p32 DECIMAL(10, 2),
    p33 DECIMAL(10, 2),
    p34 DECIMAL(10, 2),
    p35 DECIMAL(10, 2),
    p36 DECIMAL(10, 2),
    p37 DECIMAL(10, 2),
    p38 DECIMAL(10, 2),
    p39 DECIMAL(10, 2),
    p40 DECIMAL(10, 2),
    p41 DECIMAL(10, 2),
    p42 DECIMAL(10, 2),
    p43 DECIMAL(10, 2),
    p44 DECIMAL(10, 2),
    p45 DECIMAL(10, 2),
    p46 DECIMAL(10, 2),
    p47 DECIMAL(10, 2),
    p48 DECIMAL(10, 2),
    p49 DECIMAL(10, 2),
    p50 DECIMAL(10, 2),
    p51 DECIMAL(10, 2),
    p52 DECIMAL(10, 2),
    p53 DECIMAL(10, 2),
    p54 DECIMAL(10, 2),
    p55 DECIMAL(10, 2),
    p56 DECIMAL(10, 2),
    p57 DECIMAL(10, 2),
    p58 DECIMAL(10, 2),
    p59 DECIMAL(10, 2),
    p60 DECIMAL(10, 2),
    p61 DECIMAL(10, 2),
    p62 DECIMAL(10, 2),
    p63 DECIMAL(10, 2),
    p64 DECIMAL(10, 2),
    p65 DECIMAL(10, 2),
    p66 DECIMAL(10, 2),
    p67 DECIMAL(10, 2),
    p68 DECIMAL(10, 2),
    p69 DECIMAL(10, 2),
    p70 DECIMAL(10, 2),
    p71 DECIMAL(10, 2),
    p72 DECIMAL(10, 2),
    p73 DECIMAL(10, 2),
    p74 DECIMAL(10, 2),
    p75 DECIMAL(10, 2),
    p76 DECIMAL(10, 2),
    p77 DECIMAL(10, 2),
    p78 DECIMAL(10, 2),
    p79 DECIMAL(10, 2),
    p80 DECIMAL(10, 2),
    p81 DECIMAL(10, 2),
    p82 DECIMAL(10, 2),
    p83 DECIMAL(10, 2),
    p84 DECIMAL(10, 2),
    p85 DECIMAL(10, 2),
    p86 DECIMAL(10, 2),
    p87 DECIMAL(10, 2),
    p88 DECIMAL(10, 2),
    p89 DECIMAL(10, 2),
    p90 DECIMAL(10, 2),
    p91 DECIMAL(10, 2),
    p92 DECIMAL(10, 2),
    p93 DECIMAL(10, 2),
    p94 DECIMAL(10, 2),
    p95 DECIMAL(10, 2),
    p96 DECIMAL(10, 2),                 -- 第96个时间点基线的值
    PRIMARY KEY (order_id, cluster_id) 
);

-- correlation: 结果表，存储相关性分析的结果
CREATE TABLE correlation (
    order_id INT NOT NULL,                         
    factor VARCHAR(50) NOT NULL,                  -- 因素名称
    contribution DECIMAL(10, 2),                  -- 因素贡献即相关性，越接近1表示相关性越强
    PRIMARY KEY (order_id, factor)                 
);

-- baseline表: 结果表，存储基准线计算与选择的结果
CREATE TABLE baseline (
    order_id INT NOT NULL,           
    type INT NOT NULL,                  -- 基线类型 (1 - mean, 2 - max, 3 - min, 4 - quantile, 5 - typical)
    date DATE NOT NULL,                 -- 基线日期
    data_point_flag INT,                -- 数据点数标志：1 - 96点，2 - 48点，3 - 24点
    p1 DECIMAL(10, 2),                  -- 第一个时间点基线的值
    p2 DECIMAL(10, 2),                  
    p3 DECIMAL(10, 2),                  
    p4 DECIMAL(10, 2),                  
    p5 DECIMAL(10, 2),                  
    p6 DECIMAL(10, 2),
    p7 DECIMAL(10, 2),
    p8 DECIMAL(10, 2),
    p9 DECIMAL(10, 2),
    p10 DECIMAL(10, 2),
    p11 DECIMAL(10, 2),
    p12 DECIMAL(10, 2),
    p13 DECIMAL(10, 2),
    p14 DECIMAL(10, 2),
    p15 DECIMAL(10, 2),
    p16 DECIMAL(10, 2),
    p17 DECIMAL(10, 2),
    p18 DECIMAL(10, 2),
    p19 DECIMAL(10, 2),
    p20 DECIMAL(10, 2),
    p21 DECIMAL(10, 2),
    p22 DECIMAL(10, 2),
    p23 DECIMAL(10, 2),
    p24 DECIMAL(10, 2),
    p25 DECIMAL(10, 2),
    p26 DECIMAL(10, 2),
    p27 DECIMAL(10, 2),
    p28 DECIMAL(10, 2),
    p29 DECIMAL(10, 2),
    p30 DECIMAL(10, 2),
    p31 DECIMAL(10, 2),
    p32 DECIMAL(10, 2),
    p33 DECIMAL(10, 2),
    p34 DECIMAL(10, 2),
    p35 DECIMAL(10, 2),
    p36 DECIMAL(10, 2),
    p37 DECIMAL(10, 2),
    p38 DECIMAL(10, 2),
    p39 DECIMAL(10, 2),
    p40 DECIMAL(10, 2),
    p41 DECIMAL(10, 2),
    p42 DECIMAL(10, 2),
    p43 DECIMAL(10, 2),
    p44 DECIMAL(10, 2),
    p45 DECIMAL(10, 2),
    p46 DECIMAL(10, 2),
    p47 DECIMAL(10, 2),
    p48 DECIMAL(10, 2),
    p49 DECIMAL(10, 2),
    p50 DECIMAL(10, 2),
    p51 DECIMAL(10, 2),
    p52 DECIMAL(10, 2),
    p53 DECIMAL(10, 2),
    p54 DECIMAL(10, 2),
    p55 DECIMAL(10, 2),
    p56 DECIMAL(10, 2),
    p57 DECIMAL(10, 2),
    p58 DECIMAL(10, 2),
    p59 DECIMAL(10, 2),
    p60 DECIMAL(10, 2),
    p61 DECIMAL(10, 2),
    p62 DECIMAL(10, 2),
    p63 DECIMAL(10, 2),
    p64 DECIMAL(10, 2),
    p65 DECIMAL(10, 2),
    p66 DECIMAL(10, 2),
    p67 DECIMAL(10, 2),
    p68 DECIMAL(10, 2),
    p69 DECIMAL(10, 2),
    p70 DECIMAL(10, 2),
    p71 DECIMAL(10, 2),
    p72 DECIMAL(10, 2),
    p73 DECIMAL(10, 2),
    p74 DECIMAL(10, 2),
    p75 DECIMAL(10, 2),
    p76 DECIMAL(10, 2),
    p77 DECIMAL(10, 2),
    p78 DECIMAL(10, 2),
    p79 DECIMAL(10, 2),
    p80 DECIMAL(10, 2),
    p81 DECIMAL(10, 2),
    p82 DECIMAL(10, 2),
    p83 DECIMAL(10, 2),
    p84 DECIMAL(10, 2),
    p85 DECIMAL(10, 2),
    p86 DECIMAL(10, 2),
    p87 DECIMAL(10, 2),
    p88 DECIMAL(10, 2),
    p89 DECIMAL(10, 2),
    p90 DECIMAL(10, 2),
    p91 DECIMAL(10, 2),
    p92 DECIMAL(10, 2),
    p93 DECIMAL(10, 2),
    p94 DECIMAL(10, 2),
    p95 DECIMAL(10, 2),
    p96 DECIMAL(10, 2),                 -- 第96个时间点基线的值
    recommended BOOLEAN NOT NULL,       -- 是否被推荐作为最终的负荷基准线（根据行业特点）
    PRIMARY KEY (order_id, type)
);

-- forecast表: 结果表，存储超短期负荷预测和短期负荷预测的结果
CREATE TABLE forecast (
    order_id INT NOT NULL,           
    type INT NOT NULL,                  -- 预测类型：1 – 超短期，2 – 短期
    time_start DATETIME NOT NULL,               
    data_point_flag INT,                -- 数据点数标志：1 - 96点，2 - 48点，3 - 24点
    p1 DECIMAL(10, 2),                  -- 第1个时间点的预测结果
    p2 DECIMAL(10, 2),                  
    p3 DECIMAL(10, 2),                  
    p4 DECIMAL(10, 2),                  
    p5 DECIMAL(10, 2),                  
    p6 DECIMAL(10, 2),
    p7 DECIMAL(10, 2),
    p8 DECIMAL(10, 2),
    p9 DECIMAL(10, 2),
    p10 DECIMAL(10, 2),
    p11 DECIMAL(10, 2),
    p12 DECIMAL(10, 2),
    p13 DECIMAL(10, 2),
    p14 DECIMAL(10, 2),
    p15 DECIMAL(10, 2),
    p16 DECIMAL(10, 2),
    p17 DECIMAL(10, 2),
    p18 DECIMAL(10, 2),
    p19 DECIMAL(10, 2),
    p20 DECIMAL(10, 2),
    p21 DECIMAL(10, 2),
    p22 DECIMAL(10, 2),
    p23 DECIMAL(10, 2),
    p24 DECIMAL(10, 2),
    p25 DECIMAL(10, 2),
    p26 DECIMAL(10, 2),
    p27 DECIMAL(10, 2),
    p28 DECIMAL(10, 2),
    p29 DECIMAL(10, 2),
    p30 DECIMAL(10, 2),
    p31 DECIMAL(10, 2),
    p32 DECIMAL(10, 2),
    p33 DECIMAL(10, 2),
    p34 DECIMAL(10, 2),
    p35 DECIMAL(10, 2),
    p36 DECIMAL(10, 2),
    p37 DECIMAL(10, 2),
    p38 DECIMAL(10, 2),
    p39 DECIMAL(10, 2),
    p40 DECIMAL(10, 2),
    p41 DECIMAL(10, 2),
    p42 DECIMAL(10, 2),
    p43 DECIMAL(10, 2),
    p44 DECIMAL(10, 2),
    p45 DECIMAL(10, 2),
    p46 DECIMAL(10, 2),
    p47 DECIMAL(10, 2),
    p48 DECIMAL(10, 2),
    p49 DECIMAL(10, 2),
    p50 DECIMAL(10, 2),
    p51 DECIMAL(10, 2),
    p52 DECIMAL(10, 2),
    p53 DECIMAL(10, 2),
    p54 DECIMAL(10, 2),
    p55 DECIMAL(10, 2),
    p56 DECIMAL(10, 2),
    p57 DECIMAL(10, 2),
    p58 DECIMAL(10, 2),
    p59 DECIMAL(10, 2),
    p60 DECIMAL(10, 2),
    p61 DECIMAL(10, 2),
    p62 DECIMAL(10, 2),
    p63 DECIMAL(10, 2),
    p64 DECIMAL(10, 2),
    p65 DECIMAL(10, 2),
    p66 DECIMAL(10, 2),
    p67 DECIMAL(10, 2),
    p68 DECIMAL(10, 2),
    p69 DECIMAL(10, 2),
    p70 DECIMAL(10, 2),
    p71 DECIMAL(10, 2),
    p72 DECIMAL(10, 2),
    p73 DECIMAL(10, 2),
    p74 DECIMAL(10, 2),
    p75 DECIMAL(10, 2),
    p76 DECIMAL(10, 2),
    p77 DECIMAL(10, 2),
    p78 DECIMAL(10, 2),
    p79 DECIMAL(10, 2),
    p80 DECIMAL(10, 2),
    p81 DECIMAL(10, 2),
    p82 DECIMAL(10, 2),
    p83 DECIMAL(10, 2),
    p84 DECIMAL(10, 2),
    p85 DECIMAL(10, 2),
    p86 DECIMAL(10, 2),
    p87 DECIMAL(10, 2),
    p88 DECIMAL(10, 2),
    p89 DECIMAL(10, 2),
    p90 DECIMAL(10, 2),
    p91 DECIMAL(10, 2),
    p92 DECIMAL(10, 2),
    p93 DECIMAL(10, 2),
    p94 DECIMAL(10, 2),
    p95 DECIMAL(10, 2),
    p96 DECIMAL(10, 2),                 -- 第96个时间点的预测结果
    accuracy DECIMAL(10, 2),
    PRIMARY KEY (order_id, time_start)          
);

-- forecast_actuals表: 结果表，存储超短期负荷预测和短期负荷预测所对应的实际负荷
CREATE TABLE forecast_actuals (
    order_id INT NOT NULL PRIMARY KEY,           
    type INT NOT NULL,                  -- 预测类型：1 – 超短期，2 – 短期
    time_start DATETIME NOT NULL,                 
    data_point_flag INT,                -- 数据点数标志：1 - 96点，2 - 48点，3 - 24点
    p1 DECIMAL(10, 2),                  -- 第1个时间点的实际负荷
    p2 DECIMAL(10, 2),                  
    p3 DECIMAL(10, 2),                  
    p4 DECIMAL(10, 2),                  
    p5 DECIMAL(10, 2),                  
    p6 DECIMAL(10, 2),
    p7 DECIMAL(10, 2),
    p8 DECIMAL(10, 2),
    p9 DECIMAL(10, 2),
    p10 DECIMAL(10, 2),
    p11 DECIMAL(10, 2),
    p12 DECIMAL(10, 2),
    p13 DECIMAL(10, 2),
    p14 DECIMAL(10, 2),
    p15 DECIMAL(10, 2),
    p16 DECIMAL(10, 2),
    p17 DECIMAL(10, 2),
    p18 DECIMAL(10, 2),
    p19 DECIMAL(10, 2),
    p20 DECIMAL(10, 2),
    p21 DECIMAL(10, 2),
    p22 DECIMAL(10, 2),
    p23 DECIMAL(10, 2),
    p24 DECIMAL(10, 2),
    p25 DECIMAL(10, 2),
    p26 DECIMAL(10, 2),
    p27 DECIMAL(10, 2),
    p28 DECIMAL(10, 2),
    p29 DECIMAL(10, 2),
    p30 DECIMAL(10, 2),
    p31 DECIMAL(10, 2),
    p32 DECIMAL(10, 2),
    p33 DECIMAL(10, 2),
    p34 DECIMAL(10, 2),
    p35 DECIMAL(10, 2),
    p36 DECIMAL(10, 2),
    p37 DECIMAL(10, 2),
    p38 DECIMAL(10, 2),
    p39 DECIMAL(10, 2),
    p40 DECIMAL(10, 2),
    p41 DECIMAL(10, 2),
    p42 DECIMAL(10, 2),
    p43 DECIMAL(10, 2),
    p44 DECIMAL(10, 2),
    p45 DECIMAL(10, 2),
    p46 DECIMAL(10, 2),
    p47 DECIMAL(10, 2),
    p48 DECIMAL(10, 2),
    p49 DECIMAL(10, 2),
    p50 DECIMAL(10, 2),
    p51 DECIMAL(10, 2),
    p52 DECIMAL(10, 2),
    p53 DECIMAL(10, 2),
    p54 DECIMAL(10, 2),
    p55 DECIMAL(10, 2),
    p56 DECIMAL(10, 2),
    p57 DECIMAL(10, 2),
    p58 DECIMAL(10, 2),
    p59 DECIMAL(10, 2),
    p60 DECIMAL(10, 2),
    p61 DECIMAL(10, 2),
    p62 DECIMAL(10, 2),
    p63 DECIMAL(10, 2),
    p64 DECIMAL(10, 2),
    p65 DECIMAL(10, 2),
    p66 DECIMAL(10, 2),
    p67 DECIMAL(10, 2),
    p68 DECIMAL(10, 2),
    p69 DECIMAL(10, 2),
    p70 DECIMAL(10, 2),
    p71 DECIMAL(10, 2),
    p72 DECIMAL(10, 2),
    p73 DECIMAL(10, 2),
    p74 DECIMAL(10, 2),
    p75 DECIMAL(10, 2),
    p76 DECIMAL(10, 2),
    p77 DECIMAL(10, 2),
    p78 DECIMAL(10, 2),
    p79 DECIMAL(10, 2),
    p80 DECIMAL(10, 2),
    p81 DECIMAL(10, 2),
    p82 DECIMAL(10, 2),
    p83 DECIMAL(10, 2),
    p84 DECIMAL(10, 2),
    p85 DECIMAL(10, 2),
    p86 DECIMAL(10, 2),
    p87 DECIMAL(10, 2),
    p88 DECIMAL(10, 2),
    p89 DECIMAL(10, 2),
    p90 DECIMAL(10, 2),
    p91 DECIMAL(10, 2),
    p92 DECIMAL(10, 2),
    p93 DECIMAL(10, 2),
    p94 DECIMAL(10, 2),
    p95 DECIMAL(10, 2),
    p96 DECIMAL(10, 2)                  -- 第96个时间点的实际负荷
);

-- potential表: 结果表，存储负荷潜力计算与评估的结果
CREATE TABLE potential (
    order_id INT NOT NULL,                         
    baseline_type INT NOT NULL,             -- 基线类型 (1 - mean, 2 - max, 3 - min, 4 - quantile, 5 - typical)
    time DATETIME NOT NULL,                     
    potential DECIMAL(10, 2),                     
    PRIMARY KEY (order_id, baseline_type, time)           
);

-- potential_evaluation表: 结果表，存储负荷潜力计算与评估的结果
CREATE TABLE potential_evaluation (
    order_id INT NOT NULL,                         
    baseline_type INT NOT NULL,             -- 基线类型 (1 - mean, 2 - max, 3 - min, 4 - quantile, 5 - typical)
    score DECIMAL(10, 2) NOT NULL,                    
    PRIMARY KEY (order_id, baseline_type)           
);