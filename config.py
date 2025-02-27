import argparse
import pickle
from pathlib import PosixPath, Path
def get_args() -> dict:
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')#'cuda:0'
    parser.add_argument('--seed', type=float, default=9999)#9999
    # path
    parser.add_argument('--inputs_path', type=str, default='/media/a/Data/data/test/')
    parser.add_argument('--nc_data_path', type=str, default='/media/a/Data/')
    parser.add_argument('--product', type=str, default='LandBench')
    parser.add_argument('--workname', type=str, default='LandBench')
    parser.add_argument('--modelname', type=str, default='LSTM')
    parser.add_argument('--label',nargs='+', type=str, default=["volumetric_soil_water_layer_1"])
    parser.add_argument('--label_2', nargs='+', type=str, default=["surface_sensible_heat_flux"])
    parser.add_argument('--stride', type=float, default=20) 
    parser.add_argument('--data_type', type=str, default='float32') 
    # data
    parser.add_argument('--selected_year', nargs='+', type=int, default=[1990,2020])#1979-2020
    parser.add_argument('--forcing_list', nargs='+', type=str, default=["2m_temperature","10m_u_component_of_wind","10m_v_component_of_wind","precipitation","surface_pressure","specific_humidity"])
    parser.add_argument('--land_surface_list', nargs='+', type=str, default=["surface_solar_radiation_downwards_w_m2","surface_thermal_radiation_downwards_w_m2","total_runoff"])
    parser.add_argument('--static_list', nargs='+', type=str, default=["soil_water_capacity"])

    parser.add_argument('--memmap', type=bool, default=True)
    parser.add_argument('--test_year', nargs='+', type=int, default=[2020])
    parser.add_argument('--input_size', type=float, default=10)
    parser.add_argument('--spatial_resolution', type=float, default=2)
    parser.add_argument('--normalize', type=bool, default=True)
    parser.add_argument('--split_ratio', type=float, default=0.8)
    parser.add_argument('--spatial_offset', type=float, default=3) #CNN
    parser.add_argument('--valid_split', type=bool, default=False) 
   
    # model
    parser.add_argument('--normalize_type', type=str, default='region')#global, #region
    parser.add_argument('--forcast_time', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--hidden_size', type=float, default=128)
    parser.add_argument('--batch_size', type=float, default=512)#512
    parser.add_argument('--patience', type=int, default=10) 
    parser.add_argument('--seq_len', type=float, default=365) #365 or 7;   
    parser.add_argument('--epochs', type=float, default=500)#500
    parser.add_argument('--niter', type=float, default=300) #200
    parser.add_argument('--num_repeat', type=float, default=1)#default :1
    parser.add_argument('--dropout_rate', type=float, default=0.15)
    parser.add_argument('--input_size_cnn', type=float, default=64) #CNN (seq_len)*(num of forcing_list+num of land_surface_list)+1
    parser.add_argument('--kernel_size', type=float, default=3) #CNN
    parser.add_argument('--stride_cnn', type=float, default=2) #CNN
    parser.add_argument('--task', type=str, default='multi')  # multi or single
    parser.add_argument('--n_tasks', type=int, default=2)  # number of tasks
    cfg = vars(parser.parse_args())

    return cfg
