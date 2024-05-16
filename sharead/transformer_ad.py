import logging

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
#from torch_geometric.data import Data, Batch, DataListLoader
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange,tqdm
import IPython
import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler
import torch.cuda
from numba import jit
from torch.autograd import Function
from torch.optim.lr_scheduler import StepLR
from numba import cuda
import math
import robust_loss_pytorch
from .soft_dtw_cuda import SoftDTW
from torch.utils.tensorboard import SummaryWriter

from .algorithm_utils import Algorithm, PyTorchUtils
import gc
import os
from sklearn.metrics import precision_recall_fscore_support
import pickle as pk
import optuna

class Transformer_AD(Algorithm, PyTorchUtils):
    def __init__(self, name: str='Relation_AD', num_epochs: int=20, batch_size: int=64, lr: float=1e-4,
                 hidden_dim: int=8, sequence_length: int=100, use_sfa: bool=False, 
                 seed: int=0, gpu: int = None, no_longterm: bool = False, no_featerm: bool = False, noisy_rate: float=0, details=True):
        Algorithm.__init__(self, __name__, name, seed, details=details)
        PyTorchUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        
        self.use_sfa = use_sfa
        self.no_longterm = no_longterm
        self.no_featerm = no_featerm
        self.noisy_rate = noisy_rate

        self.ed = None
        self.align_part = None
        
        #self.sscaler = StandardScaler()
    
    def self_supervised_target_generate(self, batch):
        batch_with_anomaly = self.to_var(batch) # 如果不放心就deepcopy
        size = batch_with_anomaly.shape # [batch_size, longterm_size, fea_dim, seq_len]
        anomaly_target = self.to_var(torch.zeros_like(batch_with_anomaly))
        proportion = np.random.rand() * np.random.rand()
        node_choices = np.random.choice(size[2], max(int(size[2] * proportion), 1))
        
        start = np.random.randint(self.sequence_length)
        for node in node_choices:
            anomaly_target[:, 0, node, start:] = 1

            situation = np.random.rand()
            if situation < 0.33:
                add_plat = min(0.5, abs(np.random.randn()))
                if np.random.rand() <= 0.5:
                    batch_with_anomaly[:, 0, node, start:] += add_plat
                else:
                    batch_with_anomaly[:, 0, node, start:] -= add_plat
            elif situation < 0.66:
                add_plat = min(1, abs(np.random.randn()))
                if np.random.rand() <= 0.5:
                    batch_with_anomaly[:, 0, node, start:] += torch.rand(size[0], self.sequence_length-start, device=self.device) * add_plat
                else:
                    batch_with_anomaly[:, 0, node, start:] -= torch.rand(size[0], self.sequence_length-start, device=self.device) * add_plat
            else:
                add_plat = min(1, abs(np.random.randn()))
                if np.random.rand() <= 0.5:
                    batch_with_anomaly[:, 0, node, start:] += torch.rand(size[0], self.sequence_length-start, device=self.device).sort()[0] * add_plat
                else:
                    batch_with_anomaly[:, 0, node, start:] -= torch.rand(size[0], self.sequence_length-start, device=self.device).sort()[0] * add_plat
        return batch_with_anomaly, anomaly_target
        
    def organize_data(self, X_list):
        '''
        input:
        X_list: 由(X, period)的元组组成的列表,训练集,包括多条多维时间序列数据（每条的维度可不一致）
            X: 一个多维时间序列，存成pd.DataFrame，行是每个不同特征的值，列是不同时间戳的值，索引是时间戳
            period：一个参考周期包含的点数
        '''
        dataslices_list = []
        for X, period in X_list:
            yester_1 = X.shift(period)
            yester_2 = X.shift(period*2)
            #yester_7 = example_all.shift(1440*7)
            #yester_14 = example_all.shift(1440*14)
            dataslices = []
            for i in range(period*2, len(X) - self.sequence_length +1): 
                dataslices.append(np.stack((X[i:i + self.sequence_length].swapaxes(0,1).values,
                                            yester_1[i:i + self.sequence_length].swapaxes(0,1).values,
                                            yester_2[i:i + self.sequence_length].swapaxes(0,1).values),
                                            #yester_7[i:i + seq_length].swapaxes(0,1).values.flatten(),
                                            #yester_14[i:i + seq_length].swapaxes(0,1).values.flatten()),
                                            axis=0))
            
            dataslices_list.append(dataslices)
        return dataslices_list
    
    def align_ssft_baseline_slavae(self, dataset: Dataset, feedback_labels: dict, num_epochs: int=1, weight_decay: float=1e-4):
        #self.align_part = AlignModule_2(metric_fea_size=metric_fea_size, entity_num=entity_num, seed=self.seed, gpu=self.gpu)

        self.ed.train()
        optimizer = torch.optim.Adam(self.ed.parameters(), lr=self.lr, weight_decay=weight_decay)
        lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

        
        loss_logs = []

        dataset.feedback(feedback_labels)
        feedback_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in trange(num_epochs):
            for (i_feed, batch) in enumerate(tqdm(feedback_loader)):
                data, pre, label, entity_id = batch
                data, pre, label, entity_id = self.to_var(data), self.to_var(pre), self.to_var(label), self.to_var(entity_id)

                x_reconstruct, x_origin, z = self.ed(data, with_z=True)

                feedback_loss = self.ed.align_mse_loss(x_reconstruct, x_origin, label)

                self.ed.zero_grad()
                feedback_loss.backward()
                #nn.utils.clip_grad_norm_(self.align_part.parameters(), 10)
                optimizer.step()

                loss_logs.append(feedback_loss.data.item())

                if i_feed % 20 == 0:
                    IPython.display.clear_output()
                    plt.plot(np.array(loss_logs))
                    plt.show()

            lr_scheduler.step()
            gc.collect()

    

    def align_sft_baseline(self, dataset: Dataset, feedback_labels: dict, hid_dim: int=512, num_epochs: int=1, use_bce: bool=True):
        self.align_part = AlignModule_SFT_baseline(input_dim=self.sequence_length, hid_dim=hid_dim, seed=self.seed, gpu=self.gpu)

        self.to_device(self.align_part)
        optimizer = torch.optim.Adam(self.align_part.parameters(), lr=self.lr, weight_decay=1e-4)
        lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

        self.ed.eval()
        self.align_part.train()
        loss_logs = []

        dataset.feedback(feedback_labels)
        feedback_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in trange(num_epochs):
            for (i_feed, batch) in enumerate(tqdm(feedback_loader)):
                data, pre, label, entity_id = batch
                data, pre, label, entity_id = self.to_var(data), self.to_var(pre), self.to_var(label), self.to_var(entity_id)


                x_reconstruct, x_origin, z = self.ed(data, with_z=True)
                output = (x_reconstruct - x_origin)**2

                #print(output.shape, z.shape, label.shape)
                output = output.transpose(1,2)
                final_score = self.align_part(output)

                #print(final_score.device, label.device)
                #print(final_score.shape, label.shape)
                if use_bce:
                    feedback_loss = F.binary_cross_entropy_with_logits(final_score, label)
                else:
                    feedback_loss = self.align_part.loss_func_f1(final_score, label)#+F.binary_cross_entropy_with_logits(final_score, label)

                self.align_part.zero_grad()
                feedback_loss.backward()
                #nn.utils.clip_grad_norm_(self.align_part.parameters(), 10)
                optimizer.step()

                loss_logs.append(feedback_loss.data.item())

                if i_feed % 20 == 0:
                    IPython.display.clear_output()
                    plt.plot(np.array(loss_logs))
                    plt.show()

            lr_scheduler.step()
            gc.collect()


    def align_sft_baseline_predict(self, dataslices: Dataset):
        self.ed.eval()
        self.align_part.eval()

        data_loader = DataLoader(dataset=dataslices, batch_size=1024, shuffle = False)
        outputs = []
        
        for (i, batch) in enumerate(tqdm(data_loader)):
            x, entity_id = batch
            x, entity_id = self.to_var(x), self.to_var(entity_id)

            x_reconstruct, x_origin, z = self.ed(self.to_var(x), with_z=True)
            output = (x_reconstruct - x_origin)**2

            output = output.transpose(1,2)

            final_score = self.align_part(output)

            outputs.append(final_score.data.cpu().numpy())


        outputs = np.concatenate(outputs)
        gc.collect()
        
        return outputs


    def align(self, dataset: Dataset, feedback_labels: dict, feedback_amount: float=0.3, seed: int=0, ):
        condition_embedding = []
        threshold_embedding = []
        mac_ids = dataset.mac_ids
        for i in range(len(mac_ids)):
            dataset.test(mac_ids[i])

            label = feedback_labels[mac_ids[i]]
            

            pred = self.predict(dataset)

            label3 = label[:int(feedback_amount*len(label))]

            if len(np.unique(label3))==1:
                params = {'context_len': 11, 'threshold_percentile': 98}
            else:
                def objective(trial):
                    context_len = trial.suggest_int('context_len', 6, 100)
                    use_pred = np.mean(pred, axis = 2)[1:,-1]-np.max(np.mean(pred, axis = 2)[1:,-context_len:-1], axis= 1)
                    
                    pred3 = use_pred[:int(feedback_amount*len(use_pred))]
                    use_threshold_percentile = trial.suggest_float('threshold_percentile', 85, 99)
                    use_threshold = np.percentile(pred3, use_threshold_percentile)

                    adjust_pred = adjust_predicts(pred3, label3, use_threshold, is_filter=False)
                    f1 = calc_point2point(adjust_pred, label3)[0]
                    return f1
                
                sampler = optuna.samplers.TPESampler(seed=seed)
                study = optuna.create_study(sampler=sampler, direction='maximize')
                study.optimize(objective, n_trials=50)
                params = study.best_params

            pred = np.mean(pred, axis = 2)[1:,-1]-np.max(np.mean(pred, axis = 2)[1:,-params['context_len']:-1], axis= 1)
            
            pred3 = pred[:int(feedback_amount*len(pred))]
            align_threshold = np.percentile(pred3, params['threshold_percentile'])

            condition_embedding.append(params['context_len'])
            threshold_embedding.append(align_threshold)
        
        # or we can load these into nn.Embedding using nn.Embedding.from_pretrained
        self.condition_embedding = np.array(condition_embedding)
        self.threshold_embedding = np.array(threshold_embedding)



    
    def fit(self, data_list: Dataset, valid_data_list: Dataset=None, loss_func: str='huber', log_step:int=2, patience: int=3, is_ipython: bool=True, test_epochs: bool=False):
        '''
        input:
        data_list: 由(dataslices, edge_index)的元组组成的列表,训练集,包括多条多维时间序列数据（每条的维度可不一致）
            dataslices: 一个多维时间序列切片的列表。其中，每一项dataslice存成np.array, 张量[longterm_dim, fea_dim, seq_len]
        valid_data_list:与data_list格式一致
        '''
        if test_epochs:
            patience = self.num_epochs + 1
        if valid_data_list is None:
            indices = np.random.permutation(len(data_list))
            split_point = int(0.25 * len(data_list))
            train_loader = DataLoader(dataset=data_list, batch_size=self.batch_size, drop_last=True,
                                      sampler=SubsetRandomSampler(indices[:-split_point]))
            valid_loader = DataLoader(dataset=data_list, batch_size=1024, drop_last=True,
                                               sampler=SubsetRandomSampler(indices[-split_point:]))
        else:
            train_loader = DataLoader(dataset=data_list, batch_size=self.batch_size, drop_last=False, shuffle=True)
            valid_loader = DataLoader(dataset=valid_data_list, batch_size=1024, drop_last=False, shuffle=False)
        
        if self.ed is None:
            self.ed = RelationNet(input_dim=self.sequence_length, hidden_dim=self.hidden_dim, seed=self.seed, gpu=self.gpu, use_sfa=self.use_sfa, no_longterm=self.no_longterm, no_featerm=self.no_featerm, sfa_dim=data_list.sfa_dim, noisy_rate=self.noisy_rate)
            print(self.ed.parameters())
        
        #self.lstmed.init_weights()
        self.to_device(self.ed)
        optimizer = torch.optim.Adam(self.ed.parameters(), lr=self.lr, weight_decay=1e-4)
        lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
        iters_per_epoch = len(train_loader)
        counter = 0
        best_val_loss = np.Inf

        for epoch in trange(self.num_epochs):
            logging.debug(f'Epoch {epoch+1}/{self.num_epochs}.')
            self.ed.train()
            for (i, batch) in enumerate(tqdm(train_loader)):
                # x with shape[batch_size, longterm_size, fea_dim, seq_len], sfa with shape [batch_size, fea_dim, sfa_len]
                x, sfa = batch
                
                if self.use_sfa:
                    x_reconstruct, x_origin = self.ed(self.to_var(x), self.to_var(sfa))
                else:
                    x_reconstruct, x_origin = self.ed(self.to_var(x))
                
                if loss_func == 'rank':
                    x_with_anomaly, anomaly_target = self.self_supervised_target_generate(x)
                    anomaly_x_reconstruct = self.ed(x_with_anomaly)
                    total_loss = self.ed.loss_func_rank(anomaly_x_reconstruct, x_reconstruct)
                elif loss_func == 'mse':
                    total_loss = self.ed.loss_func_mse(x_reconstruct, x_origin)
                elif loss_func == 'mae':
                    total_loss = self.ed.loss_func_mae(x_reconstruct, x_origin)
                elif loss_func == 'dtw':
                    total_loss = self.ed.loss_func_sdtw(x_reconstruct, x_origin)
                elif loss_func == 'huber':
                    total_loss = self.ed.loss_func_huber(x_reconstruct, x_origin)
                elif loss_func == 'robust':
                    total_loss = self.ed.loss_func_robust(x_reconstruct, x_origin)
                    
                        
                loss = {}
                loss['total_loss'] = total_loss.data.item()

                self.ed.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.ed.parameters(), 10)
                optimizer.step()
                if (i+1) % log_step == 0:
                    if is_ipython:
                        IPython.display.clear_output()
                    else:
                        plt.figure()
                    log = "Epoch [{}/{}], Iter [{}/{}]".format(
                        epoch+1, self.num_epochs, i+1, iters_per_epoch)

                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)
                
                    plt_ctr = 1
                    if not hasattr(self,"loss_logs"):
                        self.loss_logs = {}
                        for loss_key in loss:
                            self.loss_logs[loss_key] = [loss[loss_key]]
                            plt.subplot(2,2,plt_ctr)
                            plt.plot(np.array(self.loss_logs[loss_key]), label=loss_key)
                            plt.legend()
                            plt_ctr += 1
                    else:
                        for loss_key in loss:
                            self.loss_logs[loss_key].append(loss[loss_key])
                            plt.subplot(2,2,plt_ctr)
                            plt.plot(np.array(self.loss_logs[loss_key]), label=loss_key)
                            plt.legend()
                            plt_ctr += 1
                        if 'valid_loss' in self.loss_logs:
                            plt.subplot(2,2,plt_ctr)
                            plt.plot(np.array(self.loss_logs['valid_loss']), label='valid_loss')
                            plt.legend()
                            print("valid_loss:", self.loss_logs['valid_loss'])
                    if is_ipython:
                        plt.show()
                    else:
                        if not os.path.exists('plots'):
                            os.makedirs('plots/')
                        plt.savefig('plots/loss.pdf')
            
            lr_scheduler.step()
            if self.noisy_rate != 0:
                self.noisy_rate += (1-self.noisy_rate)*0.5
            self.ed.eval()
            valid_losses = []
            for (i, batch) in enumerate(tqdm(valid_loader)):
                x, sfa = batch
                if self.use_sfa:
                    x_reconstruct, x_origin = self.ed(self.to_var(x), self.to_var(sfa))
                else:
                    x_reconstruct, x_origin = self.ed(self.to_var(x))
                
                if loss_func == 'rank':
                    x_with_anomaly, anomaly_target = self.self_supervised_target_generate(x)
                    anomaly_x_reconstruct = self.ed(x_with_anomaly)
                    total_loss = self.ed.loss_func_rank(anomaly_x_reconstruct, x_reconstruct)
                elif loss_func == 'mse':
                    total_loss = self.ed.loss_func_mse(x_reconstruct, x_origin)
                elif loss_func == 'mae':
                    total_loss = self.ed.loss_func_mae(x_reconstruct, x_origin)
                elif loss_func == 'dtw':
                    total_loss = self.ed.loss_func_sdtw(x_reconstruct, x_origin)
                elif loss_func == 'huber':
                    total_loss = self.ed.loss_func_huber(x_reconstruct, x_origin)
                elif loss_func == 'robust':
                    total_loss = self.ed.loss_func_robust(x_reconstruct, x_origin)
                
                valid_losses.append(total_loss.item())
            valid_loss = np.average(valid_losses)
            if 'valid_loss' in self.loss_logs:
                self.loss_logs['valid_loss'].append(valid_loss)
            else:
                self.loss_logs['valid_loss'] = [valid_loss]
            with SummaryWriter() as w:
                w.add_scalar(self.name+'/Loss/Valid', valid_loss, epoch)
            gc.collect()

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                if not os.path.exists('check_points'):
                    os.makedirs('check_points/')
                torch.save(self.ed.state_dict(), 'check_points/'+self.name+'_'+str(self.gpu)+'_'+'checkpoint.pt')
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print ("early stoppong")
                    self.ed.load_state_dict(torch.load('check_points/'+self.name+'_'+str(self.gpu)+'_'+'checkpoint.pt'))
                    break
        if not test_epochs:
            self.ed.load_state_dict(torch.load('check_points/'+self.name+'_'+str(self.gpu)+'_'+'checkpoint.pt'))
    
    def load(self, sfa_dim, file: str=None):
        if self.ed is None:
            self.ed = RelationNet(input_dim=self.sequence_length, hidden_dim=self.hidden_dim, seed=self.seed, gpu=self.gpu, use_sfa=self.use_sfa, no_longterm=self.no_longterm, no_featerm=self.no_featerm, sfa_dim=sfa_dim, noisy_rate=self.noisy_rate)
            self.to_device(self.ed)
        if file is None:
            self.ed.load_state_dict(torch.load('check_points/'+self.name+'_'+str(self.gpu)+'_'+'checkpoint.pt'))
        else:
            self.ed.load_state_dict(torch.load(file))
    
    def predict(self, dataslices: Dataset, way: str='mse'):
        '''
        return output with size[dataset_len, seq_len, fea_len]
        '''
        data_loader = DataLoader(dataset=dataslices, batch_size=1024, shuffle = False)
        
        self.ed.eval()

        scores_sum = []
        scores_max = []
        outputs = []
        
        if way == 'mse':
            for (i, batch) in enumerate(tqdm(data_loader)):
                x, sfa = batch

                if self.use_sfa:
                    x_reconstruct, x_origin = self.ed(self.to_var(x), self.to_var(sfa))
                    output = (x_reconstruct - x_origin)**2
                else:
                    x_reconstruct, x_origin = self.ed(self.to_var(x))
                    output = (x_reconstruct - x_origin)**2

                outputs.append(output.data.cpu().numpy())
        else:
            for (i, batch) in enumerate(tqdm(data_loader)):
                x, sfa = batch

                if self.use_sfa:
                    x_reconstruct, x_origin = self.ed(self.to_var(x), self.to_var(sfa))
                    output = torch.abs(x_reconstruct - x_origin)
                else:
                    x_reconstruct, x_origin = self.ed(self.to_var(x))
                    output = torch.abs(x_reconstruct - x_origin)

                outputs.append(output.data.cpu().numpy())

        outputs = np.concatenate(outputs)
        gc.collect()
        
        return outputs
    
    def predict_raw(self, dataslices: Dataset, way: str='mse'):
        '''
        return output with size[dataset_len, seq_len, fea_len]
        '''
        data_loader = DataLoader(dataset=dataslices, batch_size=1024, shuffle = False)
        
        self.ed.eval()

        scores_sum = []
        scores_max = []
        outputs = []
        
        for (i, batch) in enumerate(tqdm(data_loader)):
            x, sfa = batch
            x = x[:,0].transpose(1,2).contiguous()

            outputs.append(x.data.cpu().numpy())

        outputs = np.concatenate(outputs)
        gc.collect()
        
        return outputs
    
    def reconstruct(self, dataslices: Dataset):
        '''
        return output with size[dataset_len, seq_len, fea_len]
        '''
        data_loader = DataLoader(dataset=dataslices, batch_size=1024, shuffle = False)
        
        self.ed.eval()

        scores_sum = []
        scores_max = []
        outputs = []
        
        for (i, batch) in enumerate(tqdm(data_loader)):
            x, sfa = batch
            
            if self.use_sfa:
                x_reconstruct, x_origin = self.ed(self.to_var(x), self.to_var(sfa))
                output = x_reconstruct
            else:
                x_reconstruct, x_origin = self.ed(self.to_var(x))
                output = x_reconstruct
            
            outputs.append(output.data.cpu().numpy())

        outputs = np.concatenate(outputs)
        return outputs

def up2(x):
    return 2**(int(np.ceil(np.log2(x))))


class AlignModule_SFT_baseline(nn.Module, PyTorchUtils):
    def __init__(self, input_dim, hid_dim=512, nhead=4, dropout: float=0.2, seed: int=0, gpu: int=None):
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)

        submodules = [nn.Linear(input_dim, hid_dim)]
        submodules.append(TransformerEncoderLayer(d_model=hid_dim, dim_feedforward=hid_dim, nhead=nhead, dropout=dropout))
        submodules.append(TransformerEncoderLayer(d_model=hid_dim, dim_feedforward=hid_dim, nhead=nhead, dropout=dropout))
        submodules.append(nn.Linear(hid_dim, 1))

        self.attention = nn.Sequential(*submodules)


    def forward(self, x):
        # x with shape [batch_size, fea_dim, seq_len]
        # z with shape [batch_size, fea_dim, seq_len]

        # become shape [fea_dim, batch_size, input_size]
        x = x[:,:,:].transpose(0,1).contiguous()
        

        #x_z = x#torch.cat((x, z), 2)
        attention_weight = self.attention(x)
        #threshold_weight = self.thresholding(x_z)

        attention_weight = attention_weight.transpose(0,1)#.contiguous()
        #threshold_weight = threshold_weight.transpose(0,1).contiguous()
        
        #x = x.transpose(0,1)#.contiguous()

        final_score = torch.mean(attention_weight, (1,2))
        
        return final_score

    def loss_func_f1(self, final_score, label):
        final_score = F.sigmoid(final_score)
        TP = (label * final_score).sum()
        FP = ((1-label) * final_score).sum()
        FN = (label * (1-final_score)).sum()
        F1 = (2*TP) / (2*TP + FP + FN + 1e-5)
        F1 = torch.where(torch.isnan(F1), torch.full_like(F1, 0), F1)
        return -F1.mean()
        


    
class RelationNet(nn.Module, PyTorchUtils):
    def __init__(self, input_dim: int, hidden_dim: int = 10,
                 head: int = 4, dropout: float=0.2, num_layers: int=2, 
                 bias: bool=True, use_sfa: bool=False, sfa_dim: int=0, no_longterm: bool=False, no_featerm: bool=False, noisy_rate: float=0, seed: int=0, gpu: int=None):
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        self.input_dim = input_dim
        self.use_sfa = use_sfa
        self.noisy_rate = noisy_rate
        
        if not use_sfa:
            sfa_dim = hidden_dim
        
        # encoder
        
        dims = [up2(input_dim//(2**(n_layer+1))) for n_layer in range(num_layers)]
        
        submodules = [CoderBlock(input_dim, up2(input_dim//2), dropout=dropout)]
        submodules.extend([CoderBlock(up2(input_dim//(2**n_layer)), up2(input_dim//(2**(n_layer+1))), dropout=dropout) for n_layer in range(1, num_layers)])
        submodules.append(nn.Linear(dims[-1], sfa_dim))
        self.encoder = nn.Sequential(*submodules)
        
        if not no_featerm:
            fea_encoder_layers = [TransformerEncoderLayer(d_model=input_dim, nhead=head, dim_feedforward=up2(input_dim//2), dropout=dropout)]
            fea_encoder_layers.extend([TransformerEncoderLayer(d_model=up2(input_dim//(2**n_layer)), nhead=head, dim_feedforward=up2(input_dim//(2**(n_layer+1))), dropout=dropout) for n_layer in range(1, num_layers)])
            fea_encoder_layers.append(nn.Linear(dims[-1], sfa_dim))
            self.fea_transformer = nn.Sequential(*fea_encoder_layers)
        
        if not no_longterm:
            longterm_encoder_layers = [TransformerEncoderLayer(d_model=input_dim, nhead=head, dim_feedforward=up2(input_dim//2), dropout=dropout)]
            longterm_encoder_layers.extend([TransformerEncoderLayer(d_model=up2(input_dim//(2**n_layer)), nhead=head, dim_feedforward=up2(input_dim//(2**(n_layer+1))), dropout=dropout) for n_layer in range(1, num_layers)])
            longterm_encoder_layers.append(nn.Linear(dims[-1], sfa_dim))
            self.longterm_transformer = nn.Sequential(*longterm_encoder_layers)
        
        # conditioned dropout
        if use_sfa:
            self.sfa_droput = ConditionedDropout()
        
        # decoder
        if no_longterm and no_featerm:
            submodules_for_decoders = [CoderBlock(sfa_dim, sfa_dim * int(dims[-1]//hidden_dim), dropout=dropout)]
            for d in range(1, len(dims)):
                submodules_for_decoders.append(CoderBlock(sfa_dim * int(dims[-d]//hidden_dim), sfa_dim * int(dims[-d-1]//hidden_dim), dropout=dropout))
            self.decoders = nn.ModuleList(submodules_for_decoders)
            self.final_decoder = nn.Linear(sfa_dim * int(dims[0]//hidden_dim), input_dim)
        elif no_longterm or no_featerm:
            submodules_for_decoders = [CoderBlock(sfa_dim*2, sfa_dim * int(dims[-1]//hidden_dim), dropout=dropout)]
            for d in range(1, len(dims)):
                submodules_for_decoders.append(CoderBlock(sfa_dim * int(dims[-d]//hidden_dim), sfa_dim * int(dims[-d-1]//hidden_dim), dropout=dropout))
            self.decoders = nn.ModuleList(submodules_for_decoders)
            self.final_decoder = nn.Linear(sfa_dim * int(dims[0]//hidden_dim), input_dim)
        else:
            submodules_for_decoders = [CoderBlock(sfa_dim*3, sfa_dim * int(dims[-1]//hidden_dim), dropout=dropout)]
            for d in range(1, len(dims)):
                submodules_for_decoders.append(CoderBlock(sfa_dim * int(dims[-d]//hidden_dim), sfa_dim * int(dims[-d-1]//hidden_dim), dropout=dropout))
            self.decoders = nn.ModuleList(submodules_for_decoders)
            self.final_decoder = nn.Linear(sfa_dim * int(dims[0]//hidden_dim), input_dim)
            
        
        self.sdtw = SoftDTW(use_cuda=True, gamma=0.1)
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.huber = nn.SmoothL1Loss()
        self.robustloss = robust_loss_pytorch.adaptive.AdaptiveLossFunction(num_dims=input_dim, float_dtype=np.float32, device=self.device)
        
        self.no_featerm = no_featerm
        self.no_longterm = no_longterm
    
    
    def noisy_target_generate(self, batch):
        batch_with_anomaly = self.to_var(batch).clone() # 深复制
        size = batch_with_anomaly.shape # [batch_size, longterm_size, fea_dim, seq_len]
        #anomaly_target = self.to_var(torch.zeros_like(batch_with_anomaly))
        proportion = np.random.rand()# * np.random.rand()
        num_noise_node = max(int(size[2] * proportion), 1)
        node_choices = np.random.choice(size[2], num_noise_node)
        node_choices = torch.from_numpy(node_choices).to(self.device)
        
        start = np.random.randint(self.input_dim)
        
        situation = np.random.rand()
        
        if situation < 0.33:
            add_plat = np.random.randn()
            noise = torch.ones(size[0], num_noise_node, self.input_dim-start, device=self.device) * add_plat
            batch_with_anomaly[:,0,:,start:].index_add_(1, node_choices, noise)
        elif situation < 0.66:
            add_plat = np.random.randn()
            noise = torch.rand(size[0], num_noise_node, self.input_dim-start, device=self.device) * add_plat
            batch_with_anomaly[:,0,:,start:].index_add_(1, node_choices, noise)
        else:
            add_plat = np.random.randn()
            noise = torch.rand(size[0], num_noise_node, self.input_dim-start, device=self.device).sort()[0] * add_plat
            batch_with_anomaly[:,0,:,start:].index_add_(1, node_choices, noise)
            
        return batch_with_anomaly#, anomaly_target
    
    def forward(self, x, sfa=None, with_z=False):
        '''
        x: [batch_size, longterm_size, fea_dim, seq_len]
        sfa: [batch_size, fea_dim, sfa_dim]
        '''
        # x_origin with shape [batch_size, seq_len, fea_dim]
        x_origin = x[:,0].transpose(1,2).contiguous()
        
        if self.training and np.random.rand() < self.noisy_rate:
            x = self.noisy_target_generate(x)
        
        #x_yester1 = x[:,1].contiguous()
        #x_yester2 = x[:,2].contiguous()
        
        #encoder
        # z_origin with shape [batch_size, fea_dim, hid_siz]
        z_origin = self.encoder(x[:,0].contiguous())
        if self.use_sfa:
            z_origin = self.sfa_droput(z_origin, sfa)
        
        z = z_origin
            
        if not self.no_featerm:
            # x_fea with shape [fea_dim, batch_size, seq_len]
            x_fea = x[:,0].transpose(0,1).contiguous()
            # z_fea with shape [batch_size, fea_dim, hid_siz]
            z_fea = self.fea_transformer(x_fea).transpose(0,1)
            
            if self.use_sfa:
                z_fea = self.sfa_droput(z_fea, sfa)
            
            z = torch.cat((z, z_fea), 2)
            
        if not self.no_longterm:
            size = x.shape
            # x_longterm with shape [longterm_size, batch_size*fea_dim, seq_len]
            x_longterm = x.transpose(0,1).contiguous().view(size[1], size[0]*size[2], size[3])
            # z_longterm with shape [batch_size, fea_dim, hid_siz]
            z_longterm = self.longterm_transformer(x_longterm)[0].view(size[0], size[2], -1)
            
            if self.use_sfa:
                z_longterm = self.sfa_droput(z_longterm, sfa)
            
            z = torch.cat((z, z_longterm), 2)
        
        # decoder
        if self.use_sfa:
            x_reconstruct = z
            for d in self.decoders:
                x_reconstruct = d(x_reconstruct)
                x_reconstruct = self.sfa_droput(x_reconstruct, sfa)
            x_reconstruct = self.final_decoder(x_reconstruct)
        else:
            x_reconstruct = z
            for d in self.decoders:
                x_reconstruct = d(x_reconstruct)
            x_reconstruct = self.final_decoder(x_reconstruct)
        
        #return torch.min(torch.mean((x_yester1 + x_reconstruct- x_origin)**2, dim=2), torch.mean((x_yester2 + x_reconstruct- x_origin)**2, dim=2))
        
        #x_pred1 = (x_yester1 + x_reconstruct).transpose(1,2).contiguous()
        #x_pred2 = (x_yester2 + x_reconstruct).transpose(1,2).contiguous()
        #output = torch.min(self.sdtw(x_pred1, x_origin), self.sdtw(x_pred2, x_origin))
        #output = self.sdtw(x_reconstruct.transpose(1,2).contiguous(), x_origin)
        #output = (x_reconstruct.transpose(1,2).contiguous() - x_origin)**2
        #print('origin', x_origin)
        #print('reconstruct', x_reconstruct)
        #print('output', output)
        x_reconstruct = x_reconstruct.transpose(1,2).contiguous()

        if with_z:
            return x_reconstruct, x_origin, z
        else:
            return x_reconstruct, x_origin
        
    
    def loss_func_rank(self, anomaly_pred, origin_pred):
        ot = 1 - anomaly_pred.view(-1,1).repeat((1,origin_pred.size(0))) + origin_pred.view(1,-1).repeat((anomaly_pred.size(0),1))
        loss = torch.clamp(ot,0.0).sum() # 将小于0的元素变为0。
        return loss
    
    def loss_func_mse(self, x_reconstruct, x_origin):
        loss = self.mse(x_reconstruct, x_origin)
        #print('loss', loss)
        return loss
    
    def align_mse_loss(self, x_reconstruct, x_origin, label):
        loss = ((x_reconstruct - x_origin)**2).mean(1).mean(1)
        #print(2*label-1)
        loss = loss * (1-2*label)
        loss = loss.mean()
        return loss
    
    def loss_func_mae(self, x_reconstruct, x_origin):
        loss = self.mae(x_reconstruct, x_origin)
        #print('loss', loss)
        return loss
    
    def loss_func_huber(self, x_reconstruct, x_origin):
        loss = self.huber(x_reconstruct, x_origin)
        #print('loss', loss)
        return loss
    
    def loss_func_sdtw(self, x_reconstruct, x_origin):
        loss = torch.mean(self.sdtw(x_reconstruct, x_origin))
        #print('loss', loss)
        return loss
    
    def loss_func_robust(self, x_reconstruct, x_origin):
        size = x_origin.shape
        diff = (x_reconstruct - x_origin).transpose(1,2).contiguous().view(size[0]*size[2], size[1])
        #print(diff.shape)
        loss = torch.mean(self.robustloss.lossfun(diff))
        #print('loss', loss)
        return loss
    
    
class CoderBlock(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super(CoderBlock, self).__init__()
        self.self_no_attn = nn.Linear(d_model, d_model)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_feedforward)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()
        
    def forward(self, src):
        src2 = self.self_no_attn(src)
        #print(src2.shape)
        src = src + self.dropout1(src2)
        src = self.linear1(self.norm1(src))
        #print(src.shape)
        src2 = self.linear2(self.dropout(self.activation(src)))
        #print(src2.shape)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

#a little modification
class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        
        # here is the modification
        self.linear2 = nn.Linear(dim_feedforward, dim_feedforward)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()


    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.linear1(self.norm1(src))
        src2 = self.linear2(self.dropout(self.activation(src)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    
class ConditionedDropout(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, sfa):
        #print(x.shape)
        #print(sfa.shape)
        scale_shape = [1 for i in range(x.dim())]
        scale_shape[-1] = x.shape[-1] // sfa.shape[-1]
        #print(scale_shape)
        sfa = sfa.repeat(scale_shape)
        x = x.masked_fill(sfa == 0, 0)
        return x


def shift(arr, num, fill_value=np.nan):
    arr = np.roll(arr, num)
    if num < 0:
        arr[num:] = fill_value
    elif num > 0:
        arr[:num] = fill_value
    return arr
def filter_pred(values_pred, scale=3):
    predicted_anomalies_ = np.argwhere(values_pred == 1).ravel()
    predicted_anomalies_shift_forward = shift(predicted_anomalies_, 1, fill_value=predicted_anomalies_[0])
    predicted_anomalies_shift_backward = shift(predicted_anomalies_, -1, fill_value=predicted_anomalies_[-1])
    predicted_anomalies_start = np.argwhere(
        (predicted_anomalies_shift_forward - predicted_anomalies_) != -1
    ).ravel()
    predicted_anomalies_finish = np.argwhere(
        (predicted_anomalies_ - predicted_anomalies_shift_backward) != -1
    ).ravel()
    predicted_anomalies = np.hstack(
        [
            predicted_anomalies_[predicted_anomalies_start].reshape(-1, 1),
            predicted_anomalies_[predicted_anomalies_finish].reshape(-1, 1),
        ]
    )
    for a_range in predicted_anomalies:
        if a_range[1]-a_range[0]<=scale-1:
            values_pred[a_range[0]:a_range[1]+1] = 0
    return values_pred

from prts import ts_precision, ts_recall, ts_fscore
def bf_search(label, score, verbose=True, is_filter=False):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).
    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    start = 90
    search_range = [np.percentile(score, q) for q in np.arange(start, 100, 0.1)]
    m = {'f1-score':-1., 'precision':-1., 'recall':-1.}
    m_t = 0.0
    #print(len(score))
    #print(len(search_range))
    for threshold in sorted(search_range)[::-1]:
        real = label
        pred = score > threshold
        #print(np.unique(pred))
        if is_filter:
            pred = filter_pred(pred, scale=1)
        #pred = filter_pred(pred, scale=3)
        if len(np.unique(pred))==1:
            continue
        target = ts_fscore(real, pred, beta=1.0, p_alpha=0.0, r_alpha=0.5, cardinality="reciprocal", p_bias="front", r_bias="front")
        if target > m['f1-score']:
            m_t = threshold
            m['f1-score'] = target
            m['precision'] = ts_precision(real, pred, alpha=0.0, cardinality="reciprocal", bias="front")
            m['recall'] = ts_recall(real, pred, alpha=0.5, cardinality="reciprocal", bias="front")
            if verbose:
                print("cur thr: ", threshold, target, m, m_t)
    print(m, m_t)
    return m, m_t

def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.

    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall

def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False,
                    is_filter=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.

    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):

    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
        if is_filter:
            predict = filter_pred(predict, scale=1)
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict

def bf_search_omni(label, score, verbose=True, is_filter=False):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).
    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    start = 90
    search_range = [np.percentile(score, q) for q in np.arange(start, 100, 0.1)]
    m = {'f1-score':-1., 'precision':-1., 'recall':-1.}
    m_t = 0.0
    #print(len(score))
    #print(len(search_range))
    for threshold in sorted(search_range)[::-1]:
        real = label
        pred = adjust_predicts(score, label, threshold, is_filter=is_filter)
        #print(np.unique(pred))
        if len(np.unique(pred))==1:
            continue
        target = calc_point2point(pred, label)
        if target[0] > m['f1-score']:
            m_t = threshold
            m['f1-score'] = target[0]
            m['precision'] = target[1]
            m['recall'] = target[2]
            if verbose:
                print("cur thr: ", threshold, target, m, m_t)
    print(m, m_t)
    return m, m_t