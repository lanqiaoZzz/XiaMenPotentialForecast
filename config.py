# 存储电力数据的数据库
config_data = {
    'host': 'localhost',          # 数据库主机地址
    'user': 'root',               # 数据库用户名      
    'password': 'w20020309',      # 数据库密码
    'database': 'xia_men',        # 数据库名称
    'port': 3306,                 # 数据库端口，默认3306
    'charset': 'utf8mb4'          # 指定字符集
}

# 存储天气数据的数据库
config_weather = {
    'host': 'localhost',          
    'user': 'root',                
    'password': 'w20020309',      
    'database': 'weather',        
    'port': 3306,                
    'charset': 'utf8mb4'          
}

# 存储程序运行结果的数据库
config_result = {
    'host': 'localhost',          
    'user': 'root',               
    'password': 'w20020309',      
    'database': 'xiamen_output',        
    'port': 3306,                 
    'charset': 'utf8mb4'          
}