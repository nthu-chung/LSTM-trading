import math
import numpy as np
import pandas as pd
import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.tensorboard import SummaryWriter
import talib
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'n_epochs': 50,
    'batch_size': 256,
    'look_back': 15,
    'early_stop': 25,
    'learning_rate': 0.001,
    'header_list': ["buy_price_lv1", "buy_size_lv1", "sell_price_lv1", "sell_size_lv1", "buy_price_lv2",
                    "buy_size_lv2", "sell_price_lv2", "sell_size_lv2", "buy_price_lv3", "buy_size_lv3",
                    "sell_price_lv3", "sell_size_lv3", "buy_price_lv4", "buy_size_lv4", "sell_price_lv4",
                    "sell_size_lv4", "buy_price_lv5", "buy_size_lv5", "sell_price_lv5", "sell_size_lv5"],
    'features': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 26, 27,
                 29, 30, 32, 33, 35, 36, 37, 38, 39, 40, 41],
    'save_path': './models/model.ckpt'  # Model will be saved here.
}
Input_dim = 35
Hidden_dim = 32
Num_layers = 3
Output_dim = 3
dropout_p = 0.5

list1 = []
# weight = torch.Tensor([1, 1, 1])
# weight = weight.to(device)


def select_feat(train_data, test_data):
    """Selects useful features to perform regression"""
    available_train_data = []
    available_test_data = []
    # train_data_raw = train_data[100:500, -1]
    # test_data_raw = test_data[100:500, -1]
    for index in range(len(train_data) - config['look_back']):
        available_train_data.append(train_data[index: index + config['look_back']])
    for index in range(len(test_data) - config['look_back']):
        available_test_data.append(test_data[index: index + config['look_back']])
    data_1 = np.array(available_train_data)
    data_2 = np.array(available_test_data)
    invalid_set_size_1 = int(np.round(0.1 * data_1.shape[0]))
    invalid_set_size_2 = int(np.round(0.1 * data_2.shape[0]))
    valid_set_size = int(np.round(0.2 * (data_1.shape[0] - invalid_set_size_1)))
    train_set_size = data_1.shape[0] - valid_set_size
    data_1 = torch.from_numpy(data_1).type(torch.FloatTensor)
    data_2 = torch.from_numpy(data_2).type(torch.FloatTensor)
    x_train = data_1[invalid_set_size_1:train_set_size + invalid_set_size_1, :-1, :-1]
    x_train = np.take(x_train, config['features'], axis=2)
    y_train = data_1[invalid_set_size_1:train_set_size + invalid_set_size_1, -1, -4:-1]
    x_valid = data_1[train_set_size + invalid_set_size_1:, :-1, :-1]
    x_valid = np.take(x_valid, config['features'], axis=2)
    y_valid = data_1[train_set_size + invalid_set_size_1:, -1, -4:-1]
    x_test = data_2[invalid_set_size_2:, :-1, :-1]
    print(x_test[0:40,-1,0])
    print(data_2[invalid_set_size_2:invalid_set_size_2+40,-1,0])
    x_test = np.take(x_test, config['features'], axis=2)
    
    y_test = data_2[invalid_set_size_2:, -1, -4:-1]
    class_sample_count = np.array(
        [len(np.where(train_Data[:, -1] == t)[0]) for t in np.unique(train_Data[:, -1])])
    data_weight = 1. / class_sample_count
    print(data_weight)
    samples_weight = data_weight[train_Data[:, -1].astype(int) - 1]
    samples_weight_train = samples_weight[config['look_back'] + invalid_set_size_1 - 2 
                                          :config['look_back'] + train_set_size + invalid_set_size_1 - 2]
    sampler_train = WeightedRandomSampler(samples_weight_train, len(samples_weight_train), replacement=True)
    # batch_sampler_train = BatchSampler(sampler_train, config['batch_size'], drop_last=False)
    # samples_weight_valid = samples_weight[config['look_back'] + train_set_size + invalid_set_size_1 - 1:]
    # sampler_valid = WeightedRandomSampler(samples_weight_valid, len(samples_weight_valid), replacement=True)
    # print(data_1.shape)
    # print(samples_weight.shape)
    # batch_sampler_valid = BatchSampler(sampler_valid, config['batch_size'], drop_last=False)
    # sampler_test = WeightedRandomSampler(samples_weight_test, len(samples_weight_test), replacement=True)
    # batch_sampler_test = BatchSampler(sampler_test, config['batch_size'], drop_last=False)
    shift = invalid_set_size_2 + config['look_back'] - 2
    # print(data_2[invalid_set_size_2:invalid_set_size_2+40, -1, 0])
    # print(x_test[0:40, -1, 0])
    # print(test_data[invalid_set_size_2 + config['look_back'] - 2:invalid_set_size_2 + config['look_back'] + 38, 0])
    return x_train, x_valid, x_test, y_train, y_valid, y_test, sampler_train, data_weight, shift



class TXFC2(Dataset):
    """
    x: Features.
    y: Targets, if none, do prediction.
    """

    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


# Build model
#####################

# Here we define our model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
                            dropout=dropout)
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        data = data.to(device)
        # Initialize hidden state with zeros
        h_0 = torch.zeros(self.num_layers, data.size(0), self.hidden_dim).requires_grad_()
        h_0 = h_0.to(device)
        # Initialize cell state
        c_0 = torch.zeros(self.num_layers, data.size(0), self.hidden_dim).requires_grad_()
        c_0 = c_0.to(device)
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (h_n, c_n) = self.lstm(data, (h_0.detach(), c_0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        lstm_out = self.fc(out[:, -1, :])
        out = torch.sigmoid(lstm_out)
        # out = torch.sigmoid(lstm_out)
        # out.size() --> 100, 10
        return lstm_out


model = LSTM(Input_dim, Hidden_dim, Output_dim, Num_layers, dropout_p)
model = model.to(device)

optimiser = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
df_train = pd.read_csv('training_data_4.csv', names=config['header_list'])
df_test = pd.read_csv('testing_data_4.csv', names=config['header_list'])

def dftrainandtest() :

   
    rise_train1 = df_train['buy_price_lv1'] > df_train['buy_price_lv1'].shift(1)
    fall_train1 = df_train['buy_price_lv1'] < df_train['buy_price_lv1'].shift(1)
    same_train1 = df_train['buy_price_lv1'] == df_train['buy_price_lv1'].shift(1)
    df_train.loc[rise_train1, 'bOF_lv1'] = df_train['buy_size_lv1']
    df_train.loc[same_train1, 'bOF_lv1'] = df_train['buy_size_lv1'] - df_train['buy_size_lv1'].shift(1)
    df_train.loc[fall_train1, 'bOF_lv1'] = -1 * df_train['buy_size_lv1']
    df_train.loc[rise_train1, 'aOF_lv1'] = -1 * df_train['sell_size_lv1']
    df_train.loc[same_train1, 'aOF_lv1'] = df_train['sell_size_lv1'] - df_train['sell_size_lv1'].shift(1)
    df_train.loc[fall_train1, 'aOF_lv1'] = df_train['sell_size_lv1']
    df_train['OFI_lv1'] = df_train['bOF_lv1'] - df_train['aOF_lv1']
    rise_test1 = df_test['buy_price_lv1'] > df_test['buy_price_lv1'].shift(1)
    fall_test1 = df_test['buy_price_lv1'] < df_test['buy_price_lv1'].shift(1)
    same_test1 = df_test['buy_price_lv1'] == df_test['buy_price_lv1'].shift(1)
    df_test.loc[rise_test1, 'bOF_lv1'] = df_test['buy_size_lv1']
    df_test.loc[same_test1, 'bOF_lv1'] = df_test['buy_size_lv1'] - df_test['buy_size_lv1'].shift(1)
    df_test.loc[fall_test1, 'bOF_lv1'] = -1 * df_test['buy_size_lv1']
    df_test.loc[rise_test1, 'aOF_lv1'] = -1 * df_test['sell_size_lv1']
    df_test.loc[same_test1, 'aOF_lv1'] = df_test['sell_size_lv1'] - df_test['sell_size_lv1'].shift(1)
    df_test.loc[fall_test1, 'aOF_lv1'] = df_test['sell_size_lv1']
    df_test['OFI_lv1'] = df_test['bOF_lv1'] - df_test['aOF_lv1']
    rise_train2 = df_train['buy_price_lv2'] > df_train['buy_price_lv2'].shift(1)
    fall_train2 = df_train['buy_price_lv2'] < df_train['buy_price_lv2'].shift(1)
    same_train2 = df_train['buy_price_lv2'] == df_train['buy_price_lv2'].shift(1)
    df_train.loc[rise_train2, 'bOF_lv2'] = df_train['buy_size_lv2']
    df_train.loc[same_train2, 'bOF_lv2'] = df_train['buy_size_lv2'] - df_train['buy_size_lv2'].shift(1)
    df_train.loc[fall_train2, 'bOF_lv2'] = -1 * df_train['buy_size_lv2']
    df_train.loc[rise_train2, 'aOF_lv2'] = -1 * df_train['sell_size_lv2']
    df_train.loc[same_train2, 'aOF_lv2'] = df_train['sell_size_lv2'] - df_train['sell_size_lv2'].shift(1)
    df_train.loc[fall_train2, 'aOF_lv2'] = df_train['sell_size_lv2']
    df_train['OFI_lv2'] = df_train['bOF_lv2'] - df_train['aOF_lv2']
    rise_test2 = df_test['buy_price_lv2'] > df_test['buy_price_lv2'].shift(1)
    fall_test2 = df_test['buy_price_lv2'] < df_test['buy_price_lv2'].shift(1)
    same_test2 = df_test['buy_price_lv2'] == df_test['buy_price_lv2'].shift(1)
    df_test.loc[rise_test2, 'bOF_lv2'] = df_test['buy_size_lv2']
    df_test.loc[same_test2, 'bOF_lv2'] = df_test['buy_size_lv2'] - df_test['buy_size_lv2'].shift(1)
    df_test.loc[fall_test2, 'bOF_lv2'] = -1 * df_test['buy_size_lv2']
    df_test.loc[rise_test2, 'aOF_lv2'] = -1 * df_test['sell_size_lv2']
    df_test.loc[same_test2, 'aOF_lv2'] = df_test['sell_size_lv2'] - df_test['sell_size_lv2'].shift(1)
    df_test.loc[fall_test2, 'aOF_lv2'] = df_test['sell_size_lv2']
    df_test['OFI_lv2'] = df_test['bOF_lv2'] - df_test['aOF_lv2']
    rise_train3 = df_train['buy_price_lv3'] > df_train['buy_price_lv3'].shift(1)
    fall_train3 = df_train['buy_price_lv3'] < df_train['buy_price_lv3'].shift(1)
    same_train3 = df_train['buy_price_lv3'] == df_train['buy_price_lv3'].shift(1)
    df_train.loc[rise_train3, 'bOF_lv3'] = df_train['buy_size_lv3']
    df_train.loc[same_train3, 'bOF_lv3'] = df_train['buy_size_lv3'] - df_train['buy_size_lv3'].shift(1)
    df_train.loc[fall_train3, 'bOF_lv3'] = -1 * df_train['buy_size_lv3']
    df_train.loc[rise_train3, 'aOF_lv3'] = -1 * df_train['sell_size_lv3']
    df_train.loc[same_train3, 'aOF_lv3'] = df_train['sell_size_lv3'] - df_train['sell_size_lv3'].shift(1)
    df_train.loc[fall_train3, 'aOF_lv3'] = df_train['sell_size_lv3']
    df_train['OFI_lv3'] = df_train['bOF_lv3'] - df_train['aOF_lv3']
    rise_test3 = df_test['buy_price_lv3'] > df_test['buy_price_lv3'].shift(1)
    fall_test3 = df_test['buy_price_lv3'] < df_test['buy_price_lv3'].shift(1)
    same_test3 = df_test['buy_price_lv3'] == df_test['buy_price_lv3'].shift(1)
    df_test.loc[rise_test3, 'bOF_lv3'] = df_test['buy_size_lv3']
    df_test.loc[same_test3, 'bOF_lv3'] = df_test['buy_size_lv3'] - df_test['buy_size_lv3'].shift(1)
    df_test.loc[fall_test3, 'bOF_lv3'] = -1 * df_test['buy_size_lv3']
    df_test.loc[rise_test3, 'aOF_lv3'] = -1 * df_test['sell_size_lv3']
    df_test.loc[same_test3, 'aOF_lv3'] = df_test['sell_size_lv3'] - df_test['sell_size_lv3'].shift(1)
    df_test.loc[fall_test3, 'aOF_lv3'] = df_test['sell_size_lv3']
    df_test['OFI_lv3'] = df_test['bOF_lv3'] - df_test['aOF_lv3']
    rise_train4 = df_train['buy_price_lv4'] > df_train['buy_price_lv4'].shift(1)
    fall_train4 = df_train['buy_price_lv4'] < df_train['buy_price_lv4'].shift(1)
    same_train4 = df_train['buy_price_lv4'] == df_train['buy_price_lv4'].shift(1)
    df_train.loc[rise_train4, 'bOF_lv4'] = df_train['buy_size_lv4']
    df_train.loc[same_train4, 'bOF_lv4'] = df_train['buy_size_lv4'] - df_train['buy_size_lv4'].shift(1)
    df_train.loc[fall_train4, 'bOF_lv4'] = -1 * df_train['buy_size_lv4']
    df_train.loc[rise_train4, 'aOF_lv4'] = -1 * df_train['sell_size_lv4']
    df_train.loc[same_train4, 'aOF_lv4'] = df_train['sell_size_lv4'] - df_train['sell_size_lv4'].shift(1)
    df_train.loc[fall_train4, 'aOF_lv4'] = df_train['sell_size_lv4']
    df_train['OFI_lv4'] = df_train['bOF_lv4'] - df_train['aOF_lv4']
    rise_test4 = df_test['buy_price_lv4'] > df_test['buy_price_lv4'].shift(1)
    fall_test4 = df_test['buy_price_lv4'] < df_test['buy_price_lv4'].shift(1)
    same_test4 = df_test['buy_price_lv4'] == df_test['buy_price_lv4'].shift(1)
    df_test.loc[rise_test4, 'bOF_lv4'] = df_test['buy_size_lv4']
    df_test.loc[same_test4, 'bOF_lv4'] = df_test['buy_size_lv4'] - df_test['buy_size_lv4'].shift(1)
    df_test.loc[fall_test4, 'bOF_lv4'] = -1 * df_test['buy_size_lv4']
    df_test.loc[rise_test4, 'aOF_lv4'] = -1 * df_test['sell_size_lv4']
    df_test.loc[same_test4, 'aOF_lv4'] = df_test['sell_size_lv4'] - df_test['sell_size_lv4'].shift(1)
    df_test.loc[fall_test4, 'aOF_lv4'] = df_test['sell_size_lv4']
    df_test['OFI_lv4'] = df_test['bOF_lv4'] - df_test['aOF_lv4']
    rise_train5 = df_train['buy_price_lv5'] > df_train['buy_price_lv5'].shift(1)
    fall_train5 = df_train['buy_price_lv5'] < df_train['buy_price_lv5'].shift(1)
    same_train5 = df_train['buy_price_lv5'] == df_train['buy_price_lv5'].shift(1)
    df_train.loc[rise_train5, 'bOF_lv5'] = df_train['buy_size_lv5']
    df_train.loc[same_train5, 'bOF_lv5'] = df_train['buy_size_lv5'] - df_train['buy_size_lv5'].shift(1)
    df_train.loc[fall_train5, 'bOF_lv5'] = -1 * df_train['buy_size_lv5']
    df_train.loc[rise_train5, 'aOF_lv5'] = -1 * df_train['sell_size_lv5']
    df_train.loc[same_train5, 'aOF_lv5'] = df_train['sell_size_lv5'] - df_train['sell_size_lv5'].shift(1)
    df_train.loc[fall_train5, 'aOF_lv5'] = df_train['sell_size_lv5']
    df_train['OFI_lv5'] = df_train['bOF_lv5'] - df_train['aOF_lv5']
    rise_test5 = df_test['buy_price_lv5'] > df_test['buy_price_lv5'].shift(1)
    fall_test5 = df_test['buy_price_lv5'] < df_test['buy_price_lv5'].shift(1)
    same_test5 = df_test['buy_price_lv5'] == df_test['buy_price_lv5'].shift(1)
    df_test.loc[rise_test5, 'bOF_lv5'] = df_test['buy_size_lv5']
    df_test.loc[same_test5, 'bOF_lv5'] = df_test['buy_size_lv5'] - df_test['buy_size_lv5'].shift(1)
    df_test.loc[fall_test5, 'bOF_lv5'] = -1 * df_test['buy_size_lv5']
    df_test.loc[rise_test5, 'aOF_lv5'] = -1 * df_test['sell_size_lv5']
    df_test.loc[same_test5, 'aOF_lv5'] = df_test['sell_size_lv5'] - df_test['sell_size_lv5'].shift(1)
    df_test.loc[fall_test5, 'aOF_lv5'] = df_test['sell_size_lv5']
    df_test['OFI_lv5'] = df_test['bOF_lv5'] - df_test['aOF_lv5']
    df_train['MACD'] = talib.MACD((df_train['buy_price_lv1'] + df_train['sell_price_lv1']) / 2, fastperiod=12,
                                slowperiod=26, signalperiod=9)[0]
    df_test['MACD'] = talib.MACD((df_test['buy_price_lv1'] + df_test['sell_price_lv1']) / 2, fastperiod=12,
                                slowperiod=26, signalperiod=9)[0]
    df_test['MOM'] = talib.MOM((df_test['buy_price_lv1'] + df_test['sell_price_lv1']) / 2,timeperiod=10)
    df_train['MOM'] = talib.MOM((df_train['buy_price_lv1'] + df_train['sell_price_lv1']) / 2,timeperiod=10)
    df_train['EMA'] = talib.EMA((df_train['buy_price_lv1'] + df_train['sell_price_lv1']) / 2, timeperiod=30)
    df_test['EMA'] = talib.EMA((df_test['buy_price_lv1'] + df_test['sell_price_lv1']) / 2, timeperiod=30)
    df_test['CCI'] = talib.CCI(df_test['sell_price_lv5'],df_test['buy_price_lv5'],(df_test['buy_price_lv1'] + df_test['sell_price_lv1']) / 2,timeperiod=14)
    df_train['CCI'] = talib.CCI(df_train['sell_price_lv5'],df_train['buy_price_lv5'],(df_train['buy_price_lv1'] + df_train['sell_price_lv1']) / 2,timeperiod=14)
    
    df_train['up'] = (df_train['buy_price_lv1'] + df_train['sell_price_lv1']) / 2 > (df_train['buy_price_lv1'].shift(1) +
                                                                                    df_train['sell_price_lv1'].shift(
                                                                                        1)) / 2
    df_train['up'] = df_train['up'].astype(int)
    df_train['same'] = (df_train['buy_price_lv1'] + df_train['sell_price_lv1']) / 2 == (
            df_train['buy_price_lv1'].shift(1) + df_train['sell_price_lv1'].shift(1)) / 2
    df_train['same'] = df_train['same'].astype(int)
    df_train['down'] = (df_train['buy_price_lv1'] + df_train['sell_price_lv1']) / 2 < (df_train['buy_price_lv1'].shift(1) +
                                                                                    df_train[
                                                                                        'sell_price_lv1'].shift(1)) / 2
    df_train['down'] = df_train['down'].astype(int)
    df_test['up'] = (df_test['buy_price_lv1'] + df_test['sell_price_lv1']) / 2 > (df_test['buy_price_lv1'].shift(1) +
                                                                                df_test['sell_price_lv1'].shift(1)) / 2
    df_test['up'] = df_test['up'].astype(int)
    df_test['same'] = (df_test['buy_price_lv1'] + df_test['sell_price_lv1']) / 2 == (df_test['buy_price_lv1'].shift(1) +
                                                                                    df_test['sell_price_lv1'].shift(
                                                                                        1)) / 2
    df_test['same'] = df_test['same'].astype(int)
    df_test['down'] = (df_test['buy_price_lv1'] + df_test['sell_price_lv1']) / 2 < (df_test['buy_price_lv1'].shift(1) +
                                                                                    df_test['sell_price_lv1'].shift(1)) / 2
    df_test['down'] = df_test['down'].astype(int)


    df_train['label_type'] = df_train['up'] * 1 + df_train['same'] * 2 + df_train['down'] * 3
    df_test['label_type'] = df_test['up'] * 1 + df_test['same'] * 2 + df_test['down'] * 3
    df_train.fillna(0, inplace=True)
    df_test.fillna(0, inplace=True)
    df_train.drop(df_train.head(1).index, inplace=True)
    df_test.drop(df_test.head(1).index, inplace=True)
    
dftrainandtest()
    
train_Data, test_Data = df_train.values, df_test.values


scaler = MinMaxScaler(feature_range=(0, 1))

train_Data[:, :-1] = scaler.fit_transform(train_Data[:, :-1])
test_Data[:, :-1] = scaler.fit_transform(train_Data[:, :-1])




# Select features
x_Train, x_Valid, x_Test, y_Train, y_Valid, y_Test, sampler_Train, data_weight, data_shift = select_feat(train_Data, test_Data)
weight = torch.Tensor([1, 1, 1])
weight = weight.to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
print(f"""x_train size: {x_Train.shape} 
y_train size: {x_Train.shape}
x_valid size: {x_Valid.shape}
y_valid size: {y_Valid.shape}  
x_test size: {x_Test.shape}
y_test size: {y_Test.shape}""")
train_dataset, valid_dataset, test_dataset = TXFC2(x_Train, y_Train), \
                                             TXFC2(x_Valid, y_Valid), \
                                             TXFC2(x_Test, y_Test)

# Pytorch data loader loads pytorch dataset into batches.
# train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
# valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
# test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler_Train, pin_memory=True)
# valid_loader = DataLoader(valid_dataset, batch_sampler=batch_sampler_Valid, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

# Train model
#####################
writer = SummaryWriter()
if not os.path.isdir('./models'):
    os.mkdir('./models')  # Create directory of saving models.
best_loss, step = math.inf, 0
no_improve = 0
for t in range(config['n_epochs']):
    model.train()
    loss_record = []
    # Initialise hidden state
    # Don't do this if you want your LSTM to be stateful
    # model.hidden = model.init_hidden()
    for i in train_loader:
        optimiser.zero_grad()
        # i[0] = torch.reshape(i[0], (512, 2, 1))
        # i[1] = torch.reshape(i[1], (512, 1))
        i[0] = i[0].to(device)
        # print(i[0])
        # print(i[0].shape)
        i[1] = i[1].to(device)
        y_train_pred = model(i[0])
        y_train_pred = torch.squeeze(y_train_pred)
        loss = loss_fn(y_train_pred, i[1])
        loss.backward()
        optimiser.step()
        step += 1
        loss_record.append(loss.detach().item())
    mean_train_loss = sum(loss_record) / len(loss_record)
    writer.add_scalar('Loss/train', mean_train_loss, step)
    if t % 10 == 0:
        print("Epoch ", t, "cross entropy(train): ", mean_train_loss)
    loss_record = []
    model.eval()
    for i in valid_loader:
        i[0] = i[0].to(device)
        i[1] = i[1].to(device)
        with torch.no_grad():
            y_valid_pred = model(i[0])
            # print(f'y_valid_pred.shape:{y_valid_pred.shape}')
            # print(f'i[1].shape:{i[1].shape}')
            y_valid_pred = torch.squeeze(y_valid_pred)
            loss = loss_fn(y_valid_pred, i[1])
        loss_record.append(loss.detach().item())
        # print("i[1]:", i[1])
        # print("y_valid_pred:", y_valid_pred)
    # print("sum(loss_record):", sum(loss_record))
    # print("len(loss_record):", len(loss_record))
    mean_valid_loss = sum(loss_record) / len(loss_record)
    writer.add_scalar('Loss/valid', mean_valid_loss, step)
    if t % 10 == 0:
        print("Epoch ", t, "cross entropy(valid): ", mean_valid_loss)
    print("epoch", t, "is completed")
    if mean_valid_loss < best_loss:
        no_improve = 0
        best_loss = mean_valid_loss
        torch.save(model.state_dict(), config['save_path'])  # Save your best model
        print('Saving model with loss {:.3f}...'.format(best_loss))
    else:
        no_improve += 1
    if no_improve >= config['early_stop']:
        print("Stop training because there is no further improvement in the model")
        break


# Testing
def save_pred(predicts, file):
    """ Save predictions to specified file """
    with open(file, 'w') as fp:
        save_writer = csv.writer(fp)
        save_writer.writerow(['id', 'predicted price'])
        for K, p in enumerate(predicts):
            save_writer.writerow([K, p[0]])


model = LSTM(Input_dim, Hidden_dim, Output_dim, Num_layers, dropout_p)
model.load_state_dict(torch.load(config['save_path']))
model = model.to(device)
model.eval()  # Set your model to evaluation mode.
preds = []
for i in test_loader:
    i[0] = i[0].to(device)
    with torch.no_grad():
        pred = model(i[0])
        preds.append(pred.detach().cpu())
preds = torch.cat(preds, dim=0)
df_preds = pd.DataFrame(preds.numpy(), columns = ['up','same','down']) 

def trade() :
    buy = 0
    sell = 0
    count = 0
    balance = 0

    global list1
    global data_shift
    row = df_preds.shape[0]
    for i in range(data_shift,row,1000) :
        
        a = df_preds.idxmax(1).iloc[i]
        print(a)
        if (a == "up") :
            buy = df_test['sell_price_lv1'].iloc[i]
            balance = balance - buy
            count += 1
        if (a == "down") :
            sell = df_test['buy_price_lv1'].iloc[i]
            balance = balance + sell
            count -= 1
        list1.append(balance)
        if (i + 1000 >= row - 1) :
            if (count > 0) :
                balance = balance + df_test['sell_price_lv1'].iloc[row - 1]*count
                list1.append(balance)
            if (count < 0) :
                balance = balance - df_test['buy_price_lv1'].iloc[row - 1]*count
                list1.append(balance)
    plt.plot(list1)
    plt.show()
    
    
trade()
writer.close()
# save_pred(preds, 'pred.csv')
result_pred = [0]
sum1_pred = 0
sum2_pred = 0
sum3_pred = 0
for i in range(preds.shape[0]):
    if preds[i][0] > preds[i][1] and preds[i][0] > preds[i][2]:
        result_pred.append(1)
        sum1_pred += 1
    elif preds[i][1] > preds[i][0] and preds[i][1] > preds[i][2]:
        result_pred.append(0)
        sum3_pred += 1
    elif preds[i][2] > preds[i][0] and preds[i][2] > preds[i][1]:
        result_pred.append(-1)
        sum2_pred += 1
    else:
        sum2_pred += 1
result_real = [0]
sum1_real = 0
sum2_real = 0
sum3_real = 0
for i in range(y_Test.shape[0]):
    if y_Test[i][0] > y_Test[i][1] and y_Test[i][0] > y_Test[i][2]:
        result_real.append(1)
        sum1_real += 1
    elif y_Test[i][1] > y_Test[i][0] and y_Test[i][1] > y_Test[i][2]:
        result_real.append(0)
        sum3_real += 1
    elif y_Test[i][2] > y_Test[i][0] and y_Test[i][2] > y_Test[i][1]:
        result_real.append(-1)
        sum2_real += 1
    else:
        sum2_real += 1

accuracy_total = sum(1 for x, y in zip(result_pred, result_real) if x == y) / len(result_real)
print('accuracy_total: %.4f' % accuracy_total)
print(sum1_pred, sum2_pred, sum3_pred)
print(sum1_real, sum2_real, sum3_real)
matrix = confusion_matrix(np.argmax(y_Test, axis=1), np.argmax(preds, axis=1))
disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=np.array(['up', 'same', 'down']))
disp.plot()
plt.show()

