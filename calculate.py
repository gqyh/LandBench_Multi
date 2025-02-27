import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
import os

import numpy as np
from config import get_args
# ---------------------------------# ---------------------------------

def lon_transform(x):
  x_new = np.zeros(x.shape)
  x_new[:,:,:int(x.shape[2]/2)] = x[:,:,int(x.shape[2]/2):] 
  x_new[:,:,int(x.shape[2]/2):] = x[:,:,:int(x.shape[2]/2)] 
  return x_new

def two_dim_lon_transform(x):
  x_new = np.zeros(x.shape)
  x_new[:,:int(x.shape[1]/2)] = x[:,int(x.shape[1]/2):] 
  x_new[:,int(x.shape[1]/2):] = x[:,:int(x.shape[1]/2)] 
  return x_new

# configures
cfg = get_args()
PATH = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/'
file_name_mask = 'Mask with {sr} spatial resolution.npy'.format(sr=cfg['spatial_resolution'])
mask = np.load(PATH+file_name_mask)
mask = two_dim_lon_transform(mask)

for i in ['LSTM','BILSTM','GRU','KDE_LSTM','KDE_BILSTM','KDE_GRU']:
    out_path = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' + cfg['workname'] + '/' + str(i) +'/focast_time '+ str(cfg['forcast_time']) +'/'
    y_pred = np.load(out_path+'_predictions.npy')
    y_pred = lon_transform(y_pred)
    
    name_pred = cfg['modelname']
    y_test = np.load(out_path+'observations.npy')
    y_test = lon_transform(y_test)
    mask[-int(mask.shape[0]/5.4):,:]=0
    min_map = np.min(y_test,axis=0)
    max_map = np.max(y_test,axis=0)
    mask[min_map==max_map] = 0
    r2_  = np.load(out_path+'r2_'+str(i) +'.npy')
    r2_ = two_dim_lon_transform(r2_)
    r_  = np.load(out_path+'r_'+str(i) +'.npy')
    r_ = two_dim_lon_transform(r_)
    urmse_  = np.load(out_path+'urmse_'+str(i) +'.npy')
    urmse_ = two_dim_lon_transform(urmse_)
    rmse_  = np.load(out_path+'rmse_'+str(i) +'.npy')
    rmse_ = two_dim_lon_transform(rmse_)
    bias_  = np.load(out_path+'bias_'+str(i) +'.npy')
    bias_ = two_dim_lon_transform(bias_)
    KGE_  = np.load(out_path+'KGE_'+str(i) +'.npy')
    KGE_ = two_dim_lon_transform(KGE_)
    NSE_  = np.load(out_path+'NSE_'+str(i) +'.npy')
    NSE_ = two_dim_lon_transform(NSE_)

    PATH = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/'
    
    lat_file_name = 'lat_{s}.npy'.format(s=cfg['spatial_resolution'])
    lon_file_name = 'lon_{s}.npy'.format(s=cfg['spatial_resolution'])
    
    # gernate lon and lat
    lat_ = np.load(PATH+lat_file_name)
    lon_ = np.load(PATH+lon_file_name)
    lon_ = np.linspace(-180,179,int(y_pred.shape[2]))

    sites_lon_index=[100,100,100,100,100]
    sites_lat_index=[55,50,45,43,42]

    mask_data = r2_[mask==1]
    total_data = mask_data.shape[0]
    sea_nannum = np.sum(mask==0)
    r_nannum = np.isnan(r_).sum()
    print('the r NAN numble of',str(i),'model is :',r_nannum-sea_nannum)
    print('the average r2 of',str(i),'model is :',np.nanmedian(r2_[mask==1]))
    print('the average ubrmse of',str(i),'model is :',np.nanmedian(urmse_[mask==1]))
    print('the average r of',str(i),'model is :',np.nanmedian(r_[mask==1]))
    print('the average rmse of',str(i),'model is :',np.nanmedian(rmse_[mask==1]))
    print('the average bias of',str(i),'model is :',np.nanmedian(bias_[mask==1]))
    print('the average KGE of',str(i),'model is :',np.nanmedian(KGE_[mask==1]))
    print('the average NSE of',str(i),'model is :',np.nanmedian(NSE_[mask==1]))

# ---------------------------------


