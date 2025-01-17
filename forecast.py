import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from prophet import Prophet
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor
from datetime import timedelta
from tools import read_data_table, write_result_table, update_forecast_table

num_epochs = 1  # LSTM 训练轮数

def build_dataset(df_csv, num_points):
    freq = '15min'
    if num_points == 48:
        freq = '30min'
    elif num_points == 24:
        freq = '1h'
    
    df = pd.DataFrame()

    for _, row in df_csv.iterrows():
        date = row['data_date']
        load_data = row[1:].values
        time_points = pd.date_range(start=f"{date} 00:00", periods=num_points, freq=freq)

        df_temp = pd.DataFrame({
            'DateTime': time_points,
            'Load': load_data
        })

        df = pd.concat([df, df_temp], ignore_index=True)
    
    df.ffill(inplace=True)
    
    return df

def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_indices = y_true != 0
    y_true = y_true[non_zero_indices]
    y_pred = y_pred[non_zero_indices]
    mape = np.mean(np.abs((y_true - y_pred) / y_true))
    return 1 - mape;

class MyLSTM(nn.Module):
    def __init__(self, predict_size):
        super(MyLSTM, self).__init__()

        self.lstm1 = nn.LSTM(input_size=1, hidden_size=128, batch_first=True, dropout=0.2)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, dropout=0.2)
        # self.lstm3 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64, predict_size)
    
    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        # out, _ = self.lstm3(out)
        out = self.fc(out[:, -1, :])  
        return out

def predict_LSTM(df, window_size, predict_size):
    train_x = []
    train_y = []
    future_x = []

    for i in range(len(df) - window_size - predict_size + 1):
        train_x.append(df['Load'][i:(i + window_size)].values)
        train_y.append(df['Load'][(i + window_size):(i + window_size + predict_size)].values)
    
    future_x.append(df['Load'][-window_size:].values)

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    future_x = np.array(future_x)

    train_x_LSTM = torch.from_numpy(train_x.reshape(-1, window_size, 1)).float()
    train_y_LSTM = torch.from_numpy(train_y).float()
    future_x_LSTM = torch.from_numpy(future_x.reshape(-1, window_size, 1)).float()

    train_dataset = TensorDataset(train_x_LSTM, train_y_LSTM)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_LSTM = MyLSTM(predict_size).to(device)  
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model_LSTM.parameters())

    model_LSTM.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for _, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model_LSTM(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch: {epoch + 1}, Loss: {epoch_loss / len(train_loader)}")

    model_LSTM.eval()

    train_pred_LSTM = []
    future_pred_LSTM = []

    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)
            output = model_LSTM(x)
            train_pred_LSTM.append(output[:, 0].cpu())
    
        train_pred_LSTM.append(output[-1, 1:].cpu())

    with torch.no_grad():
        future_x_LSTM = future_x_LSTM.to(device)
        output = model_LSTM(future_x_LSTM)
        future_pred_LSTM.append(output.cpu())

    train_pred_LSTM = torch.cat(train_pred_LSTM, dim=0).numpy()   
    future_pred_LSTM = torch.cat(future_pred_LSTM, dim=0).numpy().flatten()

    return train_pred_LSTM, future_pred_LSTM, train_x, train_y, future_x

def predict_Prophet(df, window_size, predict_size, freq):
    train_start = window_size

    df_train = df.iloc[train_start:]
    df_train.rename(columns={"DateTime": "ds"}, inplace=True)
    df_train.rename(columns={"Load": "y"}, inplace=True)

    model_Prophet = Prophet(weekly_seasonality=True, yearly_seasonality=False)
    model_Prophet.fit(df_train)

    df_future = model_Prophet.make_future_dataframe(periods=predict_size, freq=freq)   
    pred_Prophet = model_Prophet.predict(df_future)[['ds', 'yhat']]

    train_pred_Prophet = pred_Prophet.iloc[:-predict_size].yhat.values
    future_pred_Prophet = pred_Prophet.iloc[-predict_size:]

    return train_pred_Prophet, future_pred_Prophet

def predict_BP(df, window_size, train_pred_LSTM, train_pred_Prophet, future_pred_LSTM, future_pred_Prophet):
    train_start = window_size

    df_train = df.iloc[train_start:]
    df_train.rename(columns={"DateTime": "ds"}, inplace=True)
    df_train.rename(columns={"Load": "y"}, inplace=True)
    df_train['pred_LSTM'] = train_pred_LSTM
    df_train['pred_Prophet'] = train_pred_Prophet

    model_BP = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000)
    model_BP.fit(df_train[['pred_LSTM', 'pred_Prophet']], df_train['y'])

    df_future = future_pred_Prophet
    df_future.rename(columns={"yhat": "pred_Prophet"}, inplace=True)
    df_future['pred_LSTM'] = future_pred_LSTM

    future_pred_BP = model_BP.predict(df_future[['pred_LSTM', 'pred_Prophet']])
    
    train_pred_BP = model_BP.predict(df_train[['pred_LSTM', 'pred_Prophet']])
    mape = MAPE(df_train['y'], train_pred_BP)

    return future_pred_BP, mape

def ultra_short_term_load_forecast(order):
    order_id = order.get('order_id')
    user_id = order.get('user_id')
    date_start = order.get('date_start')
    date_end = order.get('date_end')

    df_csv, data_point_flag = read_data_table(user_id, date_start, date_end)
    
    freq = '15min'
    num_points = 96
    if data_point_flag == 2:
        freq = '30min'
        num_points = 48
    elif data_point_flag == 3:
        freq = '1h'
        num_points = 24
    
    window_size = order.get('window_size')
    if window_size is None:
        window_size = 1 * num_points
    predict_size = 1
    
    df = build_dataset(df_csv, num_points)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['Load'] = scaler.fit_transform(df['Load'].values.reshape(-1, 1))
  
    train_pred_LSTM, future_pred_LSTM, train_x, train_y, future_x = predict_LSTM(df, window_size, predict_size)

    model_XGBoost = LGBMRegressor(boosting_type='gbdt', learning_rate=0.01, n_estimators=5000, max_depth=13)
    model_XGBoost.fit(train_x, train_y)

    train_new_feature = train_pred_LSTM.reshape(-1, 1)
    future_new_feature = future_pred_LSTM.reshape(-1, 1)
    train_x_fusion = np.concatenate((train_x, train_new_feature), axis=1)
    future_x_fusion = np.concatenate((future_x, future_new_feature), axis=1)

    model_XGBoost.fit(train_x_fusion, train_y)
    future_pred_fusion = model_XGBoost.predict(future_x_fusion)

    df_result = pd.DataFrame({
        'ds': df.iloc[-1].DateTime + pd.to_timedelta(freq),
        'pred_LSTM': future_pred_LSTM,
        'pred_XGBoost': future_pred_fusion,
        'pred_Fusion': future_pred_fusion
    })
    df_result.iloc[:, 1:] = scaler.inverse_transform(df_result.iloc[:, 1:])
    
    result = []
    date = (df.iloc[-1].DateTime + pd.to_timedelta(freq)).date()
    row = {
        'order_id': order_id,
        'type': 1,
        'date': date,
        'data_point_flag': data_point_flag,
    }

    for i in range(96):
        if i < len(df_result['pred_Fusion']):
            row[f'p{i+1}'] = df_result['pred_Fusion'][i]
        else:
            row[f'p{i+1}'] = None
    row['accuracy'] = None
    result.append(row)

    write_result_table('forecast', result)

def short_term_load_forecast(order, write_to_db):
    order_id = order.get('order_id')
    date_start = order.get('date_start')
    date_end = order.get('date_end')

    user_id = order.get('user_id')
    df_csv, data_point_flag = read_data_table(user_id, date_start, date_end)

    # df_csv, data_point_flag = read_data_table(order_id, date_start, date_end)
    
    freq = '15min'
    num_points = 96
    if data_point_flag == 2:
        freq = '30min'
        num_points = 48
    elif data_point_flag == 3:
        freq = '1h'
        num_points = 24
    
    window_size = order.get('window_size')
    if window_size is None:
        window_size = 30 * num_points
    predict_size = num_points
    
    freq = '15min'
    
    df = build_dataset(df_csv, num_points)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['Load'] = scaler.fit_transform(df['Load'].values.reshape(-1, 1))

    train_pred_LSTM, future_pred_LSTM, _, _, _ = predict_LSTM(df, window_size, predict_size)
    train_pred_Prophet, future_pred_Prophet = predict_Prophet(df, window_size, predict_size, freq)
    future_pred_BP, acc_train = predict_BP(df, window_size, train_pred_LSTM, train_pred_Prophet, future_pred_LSTM, future_pred_Prophet)

    df_result = future_pred_Prophet
    df_result.rename(columns={"yhat": "pred_Prophet"}, inplace=True)
    df_result['pred_LSTM'] = future_pred_LSTM
    df_result['pred_BP'] = future_pred_BP
    df_result.reset_index(drop=True, inplace=True)

    df_result.iloc[:, 1:] = scaler.inverse_transform(df_result.iloc[:, 1:])

    if write_to_db is False:
        return df_result

    result = []
    date = (df.iloc[-1].DateTime + pd.to_timedelta(freq)).date()
    row = {
        'order_id': order_id,
        'type': 1,
        'date': date,
        'data_point_flag': data_point_flag,
    }
    for i in range(96):
        if i < len(df_result['pred_BP']):
            row[f'p{i+1}'] = df_result['pred_BP'][i]
        else:
            row[f'p{i+1}'] = None
    row['accuracy'] = None
    result.append(row)

    write_result_table('forecast', result)

    return df_result['pred_BP'], acc_train

def forecast(order, write_to_db=False):
    order_id = order.get('order_id')
    func_type = order.get('func_type')
    date_start = order.get('date_start')
    date_end = order.get('date_end')
    time_start = order.get('time_start')

    user_id = order.get('user_id')
    df, data_point_flag = read_data_table(user_id, date_start, date_end)

    # df, data_point_flag = read_data_table(order_id, date_start, date_end)
        
    date_forecast = date_end + timedelta(days=1)    # 预测日期
    df_forecast, _ = read_data_table(user_id, date_forecast, date_forecast)    # 预测日期的实际负荷（可能不存在）
    
    # df_forecast, _ = read_data_table(order_id, date_forecast, date_forecast)    # 预测日期的实际负荷（可能不存在）

    result = []
    row = {
        'order_id': order_id,
        'type': func_type - 4,
        'date': date_forecast,
        'data_point_flag': data_point_flag,
    }
    if func_type == 5:
        ultra_short_term_load_forecast(order)

        for i in range(96):
            if i < len(df.iloc[-1]) - 2:
                row[f'p{i+1}'] = df.iloc[-1].iloc[i + 2]    # 预测步前(num_points - 1)步
            elif i == len(df.iloc[-1]) - 2 and df_forecast is not None:
                row[f'p{i+1}'] = df_forecast.iloc[-1].iloc[1]   # 预测步
            else:
                row[f'p{i+1}'] = None
        result.append(row)
    else:
        future_pred, acc_train = short_term_load_forecast(order, write_to_db)
        
        if df_forecast is not None:
            acc_future = MAPE(df_forecast.iloc[-1].iloc[1:], future_pred)
            update_forecast_table(order_id, acc_future)

            for i in range(96):
                if i < len(df_forecast.iloc[-1]) - 1:
                    row[f'p{i+1}'] = df_forecast.iloc[-1].iloc[i + 1]
                else:
                    row[f'p{i+1}'] = None
            result.append(row)
        else:
            update_forecast_table(order_id, acc_train)
         
    if result:
        write_result_table('forecast_actuals', result)