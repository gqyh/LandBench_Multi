

import torch
import torch.nn as nn
import numpy as np
from convlstm import ConvLSTM
# ------------------------------------------------------------------------------------------------------------------------------
# simple lstm model with fully-connect layer
class LSTMModel(nn.Module):
    """single task model"""

    def __init__(self, cfg,lstmmodel_cfg):
        super(LSTMModel,self).__init__()
        self.lstm = nn.LSTM(lstmmodel_cfg["input_size"], lstmmodel_cfg["hidden_size"],batch_first=True)
        self.drop = nn.Dropout(p=cfg["dropout_rate"])
        self.dense = nn.Linear(lstmmodel_cfg["hidden_size"],lstmmodel_cfg["out_size"])

    def forward(self, inputs,aux):
        inputs_new = inputs
        x, _ = self.lstm(inputs_new.float())
        x = self.drop(x)
        # we only predict the last step
        x = self.dense(x[:,-1,:]) 
        return x

# simple CNN model with fully-connect layer
class CNN(nn.Module):
    """single task model"""

    def __init__(self, cfg):
        super(CNN,self).__init__()
        self.latn = ((2*cfg["spatial_offset"]+1)-cfg["kernel_size"])//cfg["stride_cnn"]+1
        self.lonn = ((2*cfg["spatial_offset"]+1)-cfg["kernel_size"])//cfg["stride_cnn"]+1
        self.cnn = nn.Conv2d(in_channels=cfg["input_size_cnn"],out_channels=cfg["hidden_size"],kernel_size=cfg["kernel_size"],stride=cfg["stride_cnn"])
        self.drop = nn.Dropout(p=cfg["dropout_rate"])
        self.dense = nn.Linear(int(cfg["hidden_size"])*int(self.latn)*int(self.lonn),1)

    def forward(self, inputs,aux):
        x = self.cnn(inputs.float())
        x = self.drop(x)
        x = x.reshape(x.shape[0],-1)
        # we only predict the last step
        x = self.dense(x) 
        return x
# ------------------------------------------------------------------------------------------------------------------------------
# simple convlstm model with fully-connect layer
class ConvLSTMModel(nn.Module):
    """single task model"""

    def __init__(self, cfg):
        super(ConvLSTMModel,self).__init__()
        self.ConvLSTM_net = ConvLSTM(input_size=(int(2*cfg["spatial_offset"]+1),int(2*cfg["spatial_offset"]+1)),
                       input_dim=int(cfg["input_size"]),
                       hidden_dim=[int(cfg["hidden_size"]), int(cfg["hidden_size"]/2)],
                       kernel_size=(int(cfg["kernel_size"]), int(cfg["kernel_size"])),
                       num_layers=2,cfg=cfg,batch_first=True)
        self.drop = nn.Dropout(p=cfg["dropout_rate"])
        self.dense = nn.Linear(int(cfg["hidden_size"]/2)*int(2*cfg["spatial_offset"]+1)*int(2*cfg["spatial_offset"]+1),1)
        #self.batchnorm = nn.BatchNorm1d(int(cfg["hidden_size"]/2))

    def forward(self, inputs,aux,cfg):
        threshold = torch.nn.Threshold(0., 0.0)
        inputs_new = torch.cat([inputs, aux], 2).float()
        #inputs_new = inputs.float()
        hidden =  self.ConvLSTM_net.get_init_states(inputs_new.shape[0])
        last_state, encoder_state =  self.ConvLSTM_net(inputs_new.clone(), hidden)
        last_state = self.drop(last_state)
        #Convout = last_state[:,-1,:,cfg["spatial_offset"],cfg["spatial_offset"]]
        Convout = last_state[:,-1,:,:,:]
        #Convout = self.batchnorm(Convout)
        shape=Convout.shape[0]
        #print('Convout shape is',Convout.shape)
        Convout=Convout.reshape(shape,-1)
        Convout = torch.flatten(Convout,1)
        Convout = threshold(Convout)
        predictions=self.dense(Convout)

        return predictions



class MTLLSTMModel(nn.Module):
    """Multi task model"""

    def __init__(self, cfg, lstmmodel_cfg):
        super(MTLLSTMModel,self).__init__()
        self.lstm = nn.LSTM(lstmmodel_cfg["input_size"], lstmmodel_cfg["hidden_size"],batch_first=True)
        self.drop = nn.Dropout(p=cfg["dropout_rate"])
        self.dense1 = nn.Linear(lstmmodel_cfg["hidden_size"],lstmmodel_cfg["out_size"])
        self.dense2 = nn.Linear(lstmmodel_cfg["hidden_size"], lstmmodel_cfg["out_size"])
    def forward(self, inputs,aux):
        inputs_new = inputs
        x, _ = self.lstm(inputs_new.float())
        x = self.drop(x)
        # we only predict the last step
        y1 = self.dense1(x[:,-1,:])
        y2 = self.dense2(x[:,-1,:])
        return y1,y2
class GradModel(nn.Module):
    """Gradnorm task model"""

    def __init__(self, cfg, lstmmodel_cfg):
        super(GradModel,self).__init__()
        self.Weightloss1 = torch.nn.Parameter(torch.tensor(1.0).float())
        self.Weightloss2 = torch.nn.Parameter(torch.tensor(1.0).float())
        self.lstm = nn.LSTM(lstmmodel_cfg["input_size"], lstmmodel_cfg["hidden_size"],batch_first=True)
        self.drop = nn.Dropout(p=cfg["dropout_rate"])
        self.densesoft = nn.Linear(lstmmodel_cfg["hidden_size"], lstmmodel_cfg["hidden_size"])
        self.dense1 = nn.Linear(lstmmodel_cfg["hidden_size"],lstmmodel_cfg["out_size"])
        self.dense2 = nn.Linear(lstmmodel_cfg["hidden_size"], lstmmodel_cfg["out_size"])
    def forward(self, inputs,aux):
        inputs_new = inputs
        x, _ = self.lstm(inputs_new.float())
        x = self.drop(x)
        # we only predict the last step
        y = self.densesoft(x)
        y1 = self.dense1(y[:, -1, :])
        y2 = self.dense2(y[:, -1, :])
        return y1,y2
    def get_shared_layer(self):
        return self.densesoft