import torch.nn
import torch

class NaNMSELoss():
    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg["modelname"]

    def fit(self, y_pred,y_true,lossmse):
        mask = y_true == y_true
        y_true = y_true[mask]
        y_pred = torch.squeeze(y_pred[mask])
        loss = torch.sqrt(lossmse(y_true, y_pred))
        return loss

class NaNKDEMSELoss():
    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg["modelname"]

    def fit(self, y_pred,y_true,weight):
        mask = y_true == y_true
        y_true = y_true[mask]
        y_pred = torch.squeeze(y_pred[mask])
        weight = weight[mask]
        #print(weight)
        loss = torch.sqrt(torch.mean(weight*((y_true-y_pred)**2)))#wait for modify
        return loss

class GradMSELoss():
    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg["modelname"]

    def fit(self, y_pred, y_true, lossmse):
        device = torch.device('cuda')
        mask = y_true == y_true
        y_true = y_true[mask]
        y_pred = torch.squeeze(y_pred[mask])
        loss = torch.sqrt(lossmse(y_true, y_pred))
        return loss