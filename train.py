import time
import numpy as np
import torch
import torch.nn
import os
from tqdm import trange
from data_gen import load_test_data_for_rnn,load_train_data_for_rnn,load_test_data_for_cnn, load_train_data_for_cnn,erath_data_transform,sea_mask_rnn,sea_mask_cnn,load_train_data_for_kde,sea_mask_rnn_MTL,load_train_data_for_rnn_MTL
from loss import  NaNMSELoss,NaNKDEMSELoss
from model import LSTMModel,CNN,ConvLSTMModel,BILSTMModel,GRUModel,GradModel
from utils import calculate_kde_weight_renorm
import matplotlib.pyplot as plt
import random
def Gradtrain(x,
          y,
          y_2,
          static,
          mask,
          scaler_x,
          scaler_y,
          scaler_y_2,
          cfg,
          num_repeat,
          PATH,
          out_path,
          device,
          num_task=None,
          valid_split=True):
    out_path_lstm = cfg['inputs_path'] + cfg['product'] + '/' + str(cfg['spatial_resolution']) + '/' + cfg['workname'] + '/' + cfg['modelname'] + '/focast_time ' + str(cfg['forcast_time']) + '/'
    file_name_mask = 'Mask with {sr} spatial resolution.npy'.format(sr=cfg['spatial_resolution'])
    mask = np.load(PATH + file_name_mask)
    y_test_lstm = np.load(out_path_lstm + 'obser.npy')
    if cfg['task'] in ['multi']:
        y_test_lstm_2 = np.load(out_path_lstm + 'obser2.npy')
    out_path_lstm = cfg['inputs_path'] + cfg['product'] + '/' + str(cfg['spatial_resolution']) + '/' + cfg['workname'] + '/' + cfg['modelname'] + '/focast_time ' + str(cfg['forcast_time']) + '/'
    y_test_lstm=np.squeeze(y_test_lstm)
    y_test_lstm_2=np.squeeze(y_test_lstm_2)
    nt, nlat, nlon = y_test_lstm.shape
    r_yy = np.full((nlat, nlon), np.nan)
    for i in range(nlat):
        for j in range(nlon):
            if not (np.isnan(y_test_lstm[:, i, j]).any()):
                r_yy[i, j] = np.corrcoef(y_test_lstm[:, i, j], y_test_lstm_2[:, i, j])[0, 1]
    alpha=1/np.abs(np.nanmedian(r_yy[mask == 1]))
    patience = cfg['patience']
    wait = 0
    best = 9999
    coef = 0
    count = 1
    task_losses_1 = []
    task_losses_2 = []
    loss_ratios = []
    weights_1 = []
    weights_2 = []
    seed = cfg['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    n_tasks = cfg['n_tasks']
    valid_split = cfg['valid_split']
    print('the device is {d}'.format(d=device))
    print('y type is {d_p}'.format(d_p=y.dtype))
    print('static type is {d_p}'.format(d_p=static.dtype))
    if cfg['modelname'] in ['CNN', 'ConvLSTM']:
        #	Splice x according to the sphere shape
        lat_index, lon_index = erath_data_transform(cfg, x)
        print(
            '\033[1;31m%s\033[0m' % "Applied Model is {m_n}, we need to transform the data according to the sphere shape".format(
                m_n=cfg['modelname']))
    if valid_split:
        nt, nf, nlat, nlon = x.shape  # x shape :nt,nf,nlat,nlon
        # Partition validation set and training set
        N = int(nt * cfg['split_ratio'])
        x_valid, y_valid,y_valid_2,static_valid = x[N:],y[N:],y_2[N:],static
        x, y ,y_2= x[:N], y[:N] ,y_2[:N]

    #	filter Antatctica
    print('x_train shape is', x.shape)
    print('y_train shape is', y.shape)
    print('static_train shape is', static.shape)
    print('mask shape is', mask.shape)

    # mask see regions
    # Determine the land boundary
    if cfg['modelname'] in ['LSTM']:
        if valid_split:
           x_valid, y_valid, y_valid_2,static_valid = sea_mask_rnn_MTL(cfg, x_valid, y_valid, y_valid_2, static_valid, mask)
        x, y,y_2,static = sea_mask_rnn_MTL(cfg, x, y,y_2, static, mask)
    elif cfg['modelname'] in ['CNN', 'ConvLSTM']:
        x, y, static, mask_index = sea_mask_cnn(cfg, x, y, static, mask)

    # train and validate
    # NOTE: We preprare two callbacks for training:
    #       early stopping and save best model.
    for num_ in range(cfg['num_repeat']):
        # prepare models
        # Selection model
        if cfg['modelname'] in ['LSTM']:
            lstmmodel_cfg = {}
            lstmmodel_cfg['input_size'] = cfg["input_size"]
            lstmmodel_cfg['hidden_size'] = cfg["hidden_size"] * 1
            lstmmodel_cfg['out_size'] = 1
            model = GradModel(cfg, lstmmodel_cfg).to(device)
        elif cfg['modelname'] in ['CNN']:
            model = CNN(cfg).to(device)
        elif cfg['modelname'] in ['ConvLSTM']:
            model = ConvLSTMModel(cfg).to(device)
        lossmse = torch.nn.MSELoss()
        Weightloss1 = model.Weightloss1
        Weightloss2 = model.Weightloss2
        Weightloss1.requires_grad = True
        Weightloss2.requires_grad = True
        learning_rate=cfg['learning_rate']
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
        with trange(1, cfg['epochs'] + 1) as pbar:
            for epoch in pbar:
                pbar.set_description(
                    cfg['modelname'] + ' ' + str(num_repeat))
                t_begin = time.time()
                # train
                MSELoss = 0
                for iter in range(0, cfg["niter"]):
                    # ------------------------------------------------------------------------------------------------------------------------------
                    #  train way for LSTM model
                    if cfg["modelname"] in \
                            ['LSTM']:
                        x_batch, y_batch,y_batch_2, aux_batch, _, _ = \
                            load_train_data_for_rnn_MTL(cfg, x, y,y_2,static, scaler_y)
                        x_batch = torch.from_numpy(x_batch).to(device)
                        aux_batch = torch.from_numpy(aux_batch).to(device)
                        y_batch = torch.from_numpy(y_batch).to(device)
                        y_batch_2 = torch.from_numpy(y_batch_2).to(device)
                        aux_batch = aux_batch.unsqueeze(1)
                        aux_batch = aux_batch.repeat(1, x_batch.shape[1], 1)
                        x_batch = torch.cat([x_batch, aux_batch], 2)
                        pred,pred_2,= model(x_batch, aux_batch)
                        pred = torch.squeeze(pred, 1)
                        pred_2 = torch.squeeze(pred_2, 1)
                    # ------------------------------------------------------------------------------------------------------------------------------
                    loss_1 = NaNMSELoss.fit(cfg, pred.float(), y_batch.float(), lossmse)
                    loss_2 = NaNMSELoss.fit(cfg, pred_2.float(), y_batch_2.float(), lossmse)
                    weighted_task_loss_1 = torch.mul(Weightloss1,loss_1)
                    weighted_task_loss_2 = torch.mul(Weightloss2,loss_2)
                    if iter==0:
                        if torch.cuda.is_available():
                            initial_task_loss_1=loss_1.data.cpu()
                            initial_task_loss_2=loss_2.data.cpu()
                            print('cpu')
                        else:
                            initial_task_loss_1=loss_1.data
                            initial_task_loss_2=loss_2.data
                        initial_task_loss_1=initial_task_loss_1.numpy()
                        initial_task_loss_2=initial_task_loss_2.numpy()
                        initial_task_loss = torch.log(torch.tensor(n_tasks))
                        initial_task_loss = initial_task_loss.numpy()
                    loss = weighted_task_loss_1+weighted_task_loss_2
                    optim.zero_grad()
                    loss.backward(retain_graph=True)
                    Weightloss1.grad.data = Weightloss1.grad.data * 0.0
                    Weightloss2.grad.data = Weightloss2.grad.data * 0.0
                    W = model.get_shared_layer()
                    norms = []
                    gygw_1 = torch.autograd.grad(loss_1, W.parameters(), retain_graph=True)
                    gygw_2 = torch.autograd.grad(loss_2, W.parameters(), retain_graph=True)
                    # compute the norm
                    norm_1 = torch.norm(torch.mul(Weightloss1, gygw_1[0]))
                    norm_2 = torch.norm(torch.mul(Weightloss2, gygw_2[0]))
                    norms.append(norm_1)
                    norms.append(norm_2)
                    norms = torch.stack(norms)
                    loss_ratio_1 = loss_1.data.cpu().numpy()
                    loss_ratio_2 = loss_2.data.cpu().numpy()
                    loss_ratio_1 = loss_ratio_1/initial_task_loss
                    loss_ratio_2 = loss_ratio_2/initial_task_loss
                    inverse_train_rate_1 = loss_ratio_1 / np.mean(loss_ratio_1)
                    inverse_train_rate_2 = loss_ratio_2 / np.mean(loss_ratio_2)
                    mean_norm = np.mean(norms.data.cpu().numpy())
                    constant_term_1 = torch.tensor(mean_norm * (inverse_train_rate_1 ** alpha), requires_grad=False)
                    constant_term_2 = torch.tensor(mean_norm * (inverse_train_rate_2 ** alpha), requires_grad=False)
                    constant_term_1 = constant_term_1.cuda()
                    constant_term_2 = constant_term_2.cuda()
                    grad_norm_loss_1 = torch.sum(torch.abs(norm_1 - constant_term_1))
                    grad_norm_loss_2 = torch.sum(torch.abs(norm_2 - constant_term_2))
                    Weightloss1.grad = torch.autograd.grad(grad_norm_loss_1, Weightloss1,retain_graph=True)[0]
                    Weightloss2.grad = torch.autograd.grad(grad_norm_loss_2, Weightloss2,retain_graph=True)[0]
                    optim.step()
                    MSELoss += loss.item()

                    normalize_coeff = n_tasks / torch.add(Weightloss1, Weightloss2)
                    Weightloss1.data = Weightloss1.data * normalize_coeff
                    Weightloss2.data = Weightloss2.data * normalize_coeff
                # ------------------------------------------------------------------------------------------------------------------------------
                t_end = time.time()
                loss_str = "Epoch {} Train MSE Loss {:.3f} time {:.2f}".format(epoch, MSELoss / cfg["niter"],
                                                               t_end - t_begin)
                print(loss_str)
                print(Weightloss1,Weightloss2,loss_1.data,loss_2.data,initial_task_loss)
                task_losses_1.append(loss_1.data.cpu().numpy())
                task_losses_2.append(loss_2.data.cpu().numpy())
                loss_ratios.append(np.sum(task_losses_2[-1]/ task_losses_1[-1]))
                weights_1.append(Weightloss1.data.cpu().numpy())
                weights_2.append(Weightloss2.data.cpu().numpy())
                if valid_split:
                    del x_batch, y_batch,y_batch_2, aux_batch
                    MSE_valid_loss = 0
                    if epoch % 20 == 0:
                        wait += 1
                        t_begin = time.time()
                        if cfg["modelname"] in ['LSTM']:
                            gt_list = [i for i in range(0, x_valid.shape[0] - cfg['seq_len'], cfg["stride"])]
                            n = (x_valid.shape[0] - cfg["seq_len"]) // cfg["stride"]
                            for i in range(0, n):
                                # mask
                                x_valid_batch, y_valid_batch,y_valid_batch_2,aux_valid_batch, _, _ = \
                                    load_test_data_for_rnn_MTL(cfg, x_valid, y_valid,y_valid_2,static_valid, scaler_y, cfg["stride"],i, n)
                                x_valid_batch = torch.Tensor(x_valid_batch).to(device)
                                y_valid_batch = torch.Tensor(y_valid_batch).to(device)
                                y_valid_batch_2 = torch.Tensor(y_valid_batch_2).to(device)
                                aux_valid_batch = torch.Tensor(aux_valid_batch).to(device)
                                aux_valid_batch = aux_valid_batch.unsqueeze(1)
                                aux_valid_batch = aux_valid_batch.repeat(1, x_valid_batch.shape[1], 1)
                                x_valid_batch = torch.cat([x_valid_batch, aux_valid_batch], 2)
                                with torch.no_grad():
                                    pred_valid,pred_valid_2= model(x_valid_batch, aux_valid_batch)
                                mse_valid_loss_1 = NaNMSELoss.fit(cfg, pred_valid.squeeze(1), y_valid_batch, lossmse)
                                mse_valid_loss_2 = NaNMSELoss.fit(cfg, pred_valid_2.squeeze(1), y_valid_batch_2, lossmse)
                                weighted_mse_valid_loss_1 = torch.mul(Weightloss1, mse_valid_loss_1)
                                weighted_mse_valid_loss_2 = torch.mul(Weightloss2, mse_valid_loss_2)
                                mse_valid_loss=weighted_mse_valid_loss_1+weighted_mse_valid_loss_2
                                MSE_valid_loss += mse_valid_loss.item()


                        t_end = time.time()
                        mse_valid_loss = MSE_valid_loss / (len(gt_list))
                        # get loss log
                        loss_str = '\033[1;31m%s\033[0m' % \
                                   "Epoch {} Val MSE Loss {:.3f}  time {:.2f}".format(epoch, mse_valid_loss,
                                                                                      t_end - t_begin)
                        print(loss_str)
                        val_save_acc = mse_valid_loss

                        # save best model by val loss
                        # NOTE: save best MSE results get `single_task` better than `multi_tasks`
                        #       save best NSE results get `multi_tasks` better than `single_task`
                        if val_save_acc < best:
                            # if MSE_valid_loss < best:
                            torch.save(model, out_path + cfg['modelname'] + '_para.pkl')
                            wait = 0  # release wait
                            best = val_save_acc  # MSE_valid_loss
                            print('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model')
                else:
                    # save best model by train loss
                    if MSELoss < best:
                        best = MSELoss
                        wait = 0
                        torch.save(model, out_path + cfg['modelname'] + '_para.pkl')
                        print('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model')
                        # early stopping
                        if wait >= patience:
                            return
            plt.rc('text')
            plt.rc('font', family='serif')
            fig = plt.figure()
            ax1 = fig.add_subplot(2, 3, 1)
            ax1.set_title(r'Loss_1')
            ax2 = fig.add_subplot(2, 3, 3)
            ax2.set_title(r'Loss_2')
            ax3 = fig.add_subplot(2, 3, 4)
            ax3.set_title(r'Weight of loss_1')
            ax4 = fig.add_subplot(2, 3, 6)
            ax4.set_title(r'Weight of loss_2')
            ax1.plot(task_losses_1)
            ax2.plot(task_losses_2)
            ax3.plot(weights_1)
            ax4.plot(weights_2)
            plt.show()
            return
