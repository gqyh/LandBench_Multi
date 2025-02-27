import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import unbiased_rmse, _rmse, _bias,  r2_score,GetNSE,GetKGE
from config import get_args


def lon_transform(x):
    x_new = np.zeros(x.shape)
    x_new[:, :, :int(x.shape[2] / 2)] = x[:, :, int(x.shape[2] / 2):]
    x_new[:, :, int(x.shape[2] / 2):] = x[:, :, :int(x.shape[2] / 2)]
    return x_new

def postprocess(cfg):
    PATH = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/'
    file_name_mask = 'Mask with {sr} spatial resolution.npy'.format(sr=cfg['spatial_resolution'])
    mask = np.load(PATH+file_name_mask)
# ------------------------------------------------------------------------------------------------------------------------------
    if cfg['modelname'] in ['LSTM']:
        out_path_lstm = cfg['inputs_path'] + cfg['product'] + '/' + str(cfg['spatial_resolution']) + '/' + cfg['workname'] + '/' + cfg['modelname'] + '/focast_time ' + str(cfg['forcast_time']) + '/'
        y_pred_lstm = np.load(out_path_lstm + '_predictions.npy')
        y_test_lstm = np.load(out_path_lstm + 'observations.npy')
        if cfg['task'] in ['multi']:
            y_pred_lstm_2 = np.load(out_path_lstm + '_predictions_2.npy')
            y_test_lstm_2 = np.load(out_path_lstm + 'observations_2.npy')

        print(y_pred_lstm.shape, y_test_lstm.shape)
        # get shape
        nt, nlat, nlon = y_test_lstm.shape
        # cal perf
        r2_lstm = np.full((nlat, nlon), np.nan)
        urmse_lstm = np.full((nlat, nlon), np.nan)
        r_lstm = np.full((nlat, nlon), np.nan)
        r_yy = np.full((nlat, nlon), np.nan)
        rmse_lstm = np.full((nlat, nlon), np.nan)
        bias_lstm = np.full((nlat, nlon), np.nan)
        for i in range(nlat):
            for j in range(nlon):
                if not (np.isnan(y_test_lstm[:, i, j]).any()):
                    urmse_lstm[i, j] = unbiased_rmse(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    r2_lstm[i, j] = r2_score(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    r_lstm[i, j] = np.corrcoef(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])[0, 1]
                    r_yy[i,j]= np.corrcoef(y_test_lstm[:, i, j], y_test_lstm_2[:, i, j])[0, 1]
                    rmse_lstm[i, j] = _rmse(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    bias_lstm[i, j] = _bias(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
        np.save(out_path_lstm + 'r2_' + 'LSTM' + '.npy', r2_lstm)
        np.save(out_path_lstm + 'r_' + 'LSTM' + '.npy', r_lstm)
        np.save(out_path_lstm + 'rmse_' + cfg['modelname'] + '.npy', rmse_lstm)
        np.save(out_path_lstm + 'bias_' + cfg['modelname'] + '.npy', bias_lstm)
        np.save(out_path_lstm + 'urmse_' + 'LSTM' + '.npy', urmse_lstm)
        print(np.nanmedian(r_yy[mask==1]))
        if cfg['task'] in ['multi']:
            nt_2, nlat_2, nlon_2 = y_test_lstm_2.shape
            r2_lstm_2 = np.full(( nlat_2, nlon_2), np.nan)
            urmse_lstm_2 = np.full(( nlat_2, nlon_2), np.nan)
            r_lstm_2 = np.full(( nlat_2, nlon_2), np.nan)
            rmse_lstm_2 = np.full(( nlat_2, nlon_2), np.nan)
            bias_lstm_2 = np.full(( nlat_2, nlon_2), np.nan)
            for i in range(nlat_2):
                for j in range(nlon_2):
                    if not (np.isnan(y_test_lstm_2[:, i, j]).any()):
                        urmse_lstm_2[i, j] = unbiased_rmse(y_test_lstm_2[:, i, j], y_pred_lstm_2[:, i, j])
                        r2_lstm_2[i, j] = r2_score(y_test_lstm_2[:, i, j], y_pred_lstm_2[:, i, j])
                        r_lstm_2[i, j] = np.corrcoef(y_test_lstm_2[:, i, j], y_pred_lstm_2[:, i, j])[0,1]
                        rmse_lstm_2[i, j] = _rmse(y_test_lstm_2[:, i, j], y_pred_lstm_2[:, i, j])
                        bias_lstm_2[i, j] = _bias(y_test_lstm_2[:, i, j], y_pred_lstm_2[:, i, j])
            np.save(out_path_lstm + 'r2_2'+'LSTM'+'.npy', r2_lstm_2)
            np.save(out_path_lstm + 'r_2'+'LSTM'+'.npy', r_lstm_2)
            np.save(out_path_lstm + 'rmse_2'+cfg['modelname']+'.npy', rmse_lstm_2)
            np.save(out_path_lstm + 'bias_2'+cfg['modelname']+'.npy', bias_lstm_2)
            np.save(out_path_lstm + 'urmse_2'+'LSTM'+'.npy', urmse_lstm_2)
        print('postprocess ove, please go on')


# ------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    cfg = get_args()
    postprocess(cfg)

    







               


