import pickle
import random
from time import time
from typing import Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import math
from logging import getLogger
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel

class DMFSTN(AbstractTrafficStateModel):
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    GENERIC_BLOCK = 'generic'
    def __init__(self,config, data_feature):
        super().__init__(config, data_feature)

        self._logger = getLogger()
        self.dropout = config.get('dropout', 0.2)
        self.batch_size = config.get('batch_size',64) 
        self.device = config.get('device', torch.device('cuda:0'))
        self.input_dim = config.get('input_dim', 64)
        self.hidden_dim = config.get('hidden_dim', 64)
        self.output_dim = config.get('output_dim', 1)
        self.feature_dim = config.get('feature_dim')
        self.add_time_in_day = config.get('add_time_in_day')
        self.add_day_in_week = config.get('add_day_in_week')
        self.backcast_length = config.get('backcast_length', 12)
        self.forecast_length = config.get('forecast_length', 12)


        self._scaler = self.data_feature.get('scaler')
        self.adj = self.data_feature.get('adj_mx')
        self.num_nodes = self.data_feature.get('num_nodes')
        self.nb_blocks_per_stack = self.data_feature.get('nb_blocks_per_stack', 3)

        if self.add_time_in_day == True:
            self.time_day_embedding = nn.Linear(1,1)
        if self.add_day_in_week == True:
            self.day_week_embedding = nn.Linear(7,1)

        self.fc_ox = nn.Linear(self.feature_dim,64)

        self.stack_types =(DMFSTN.TREND_BLOCK, DMFSTN.SEASONALITY_BLOCK,DMFSTN.GENERIC_BLOCK)
        self.thetas_dim = (3, 16,16)
    
        self.stacks = []
        self.parameters = []

        self.adj = self.calculate_laplacian(self.adj)

        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))
        self.parameters = nn.ParameterList(self.parameters)
        self.to(self.device)
        self._loss = None
        self._opt = None
        self._gen_intermediate_outputs = False
        self._intermediary_outputs = []
        self.seasonal_data = 0.
        self.trend_data = 0.
        self.generic_data = 0.

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        # print(f'| --  Stack {stack_type.title()} (#{stack_id}) (share_weights_in_stack={self.share_weights_in_stack})')
        blocks = []
        for block_id in range(self.nb_blocks_per_stack):
            block_init = DMFSTN.select_block(stack_type)
            if block_id != 0:
                block = blocks[-1]
            else:
                block = block_init(self.thetas_dim[stack_id], self.device,self.batch_size, self.num_nodes, self.feature_dim, self.input_dim, self.hidden_dim, self.backcast_length, self.forecast_length)
                self.parameters.extend(block.parameters())
            # print(f'     | -- {block}')
            blocks.append(block)
        return blocks

    def disable_intermediate_outputs(self):
        self._gen_intermediate_outputs = False

    def enable_intermediate_outputs(self):
        self._gen_intermediate_outputs = True

    def save(self, filename: str):
        torch.save(self, filename)

    @staticmethod
    def load(f, map_location=None, pickle_module=pickle, **pickle_load_args):
        return torch.load(f, map_location, pickle_module, **pickle_load_args)

    @staticmethod
    def select_block(block_type):
        if block_type == DMFSTN.SEASONALITY_BLOCK:
            return SeasonalityBlock
        elif block_type == DMFSTN.TREND_BLOCK:
            return TrendBlock
        else:
            return GenericBlock


    def calculate_laplacian(self, adj):
        adj = torch.tensor(adj).to(self.device)
        adj = adj + torch.eye(adj.size(0)).to(self.device)
        d = adj.sum(1)
        d_inv_sqrt = torch.pow(d, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        normalized_laplacian = adj.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
        return normalized_laplacian


    def forward(self, batch):
        x = batch['X']

        original_x = x[:,:,:,:self.feature_dim]
        trend_data = original_x.detach().unsqueeze(0)
        seasonal_data = original_x.detach().unsqueeze(0)

        x_embedding = self.fc_ox(original_x)    #(64,12,207,3)->(64,12,207,64)
        if self.add_time_in_day == True:
            time_in_day = x[:,:,:,self.feature_dim].unsqueeze(-1)
            x_embedding = x_embedding + self.time_day_embedding(time_in_day)
        if self.add_day_in_week == True:
            day_in_week = x[:,:,:,self.feature_dim+int(self.add_time_in_day):]
            x_embedding = x_embedding + self.day_week_embedding(day_in_week)

        backcast = x_embedding    #(64,12,207,64)
        
        forecast = torch.zeros(size=(self.batch_size, self.forecast_length,self.num_nodes,self.feature_dim)).to(self.device)    #(64,12,207,1)
        
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast,self.adj)
                backcast = backcast - b
                forecast = forecast + f

                stack_type = self.stack_types[stack_id]
                if stack_type == DMFSTN.SEASONALITY_BLOCK:
                    seasonal_data = torch.cat([seasonal_data,f.detach().unsqueeze(0)],dim=0)
                if stack_type == DMFSTN.TREND_BLOCK:
                    trend_data = torch.cat([trend_data,f.detach().unsqueeze(0)],dim=0)

        return backcast, forecast, self.seasonal_data, self.trend_data,self.generic_data
    
    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted)
     
        loss_predict = loss.masked_mae_torch(y_predicted, y_true, 0)

        return loss_predict
    
    def predict(self, batch):
        return self.forward(batch)[1]

class GCN(nn.Module):
    def __init__(self,num_nodes,input_dim=64, hidden_dim=64, output_dim=64,dropout=0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim,input_dim)  
        self.w_k = nn.Parameter(torch.FloatTensor(input_dim, input_dim))
        self.w_q = nn.Parameter(torch.FloatTensor(input_dim, input_dim))
        stdv = 1. / math.sqrt(self.w_k.size(1))
        self.w_k.data.uniform_(-stdv, stdv)
        self.w_q.data.uniform_(-stdv, stdv)

        self.w_z = nn.Linear(self.hidden_dim*2,self.hidden_dim)
        self.w_zi = nn.Linear(self.hidden_dim*2,self.hidden_dim)
        self.w_zf = nn.Linear(self.hidden_dim*2,self.hidden_dim)
        self.w_zo = nn.Linear(self.hidden_dim*2,self.hidden_dim)
        self.w_h = nn.Linear(self.hidden_dim,self.hidden_dim)

        self.gc1 = GraphConvolution(self.hidden_dim,self.hidden_dim)
        self.gc2 = GraphConvolution(self.hidden_dim,self.hidden_dim)

        self.dgc1 = DynamicGraphConvolution(self.hidden_dim,self.hidden_dim)
        self.dgc2 = DynamicGraphConvolution(self.hidden_dim,self.hidden_dim)

        self.ln = nn.LayerNorm([self.num_nodes,hidden_dim])
        self.dropout = nn.Dropout(dropout)
    def forward(self,x, h, c, adj):

    
        x = self.fc1(x) #新状态 x(64,207,1)->(64,207,64)

        state0= self.gc1(x,adj) #state0(64,207,64)
        state0 = self.ln(state0)
        state0 = self.dropout(self.relu(state0))

        state1 = self.gc2(state0,adj)
        state1 = self.ln(state1)
        state1 = self.dropout(self.relu(state1))


        q = torch.einsum("ijk, kl->ijl", [x, self.w_q])
        k = torch.einsum("ijk, kl->ijl", [x, self.w_k])
        attn = torch.einsum("bnd,bdm->bnm", [q, k.permute(0,2,1)])/math.sqrt(self.input_dim)
        attn = attn.softmax(dim=-1)

        dstate0 = self.dgc1(x,attn)
        dstate0 = self.ln(dstate0)
        dstate0 = self.dropout(self.relu(dstate0))

        dstate1 = self.dgc2(dstate0,attn)
        dstate1 = self.ln(dstate1)
        dstate1 = self.dropout(self.relu(dstate1))


        state1 = state1 + x + dstate1 


        x_h = torch.cat([state1, h], dim=2)
        z = F.tanh(self.w_z(x_h))
        zi = F.sigmoid(self.w_zi(x_h))
        zf = F.sigmoid(self.w_zf(x_h))
        zo = F.sigmoid(self.w_zo(x_h))

        ct = zf.mul(c)+zi.mul(z)
        ht = zo.mul(F.tanh(ct))

        return ht,ct

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.einsum("ijk, kl->ijl", [x, self.weight])
        support = support+self.bias
        output = torch.einsum("ij, bjf->bif", [adj, support])
        return output
    
class DynamicGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(DynamicGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.einsum("ijk, kl->ijl", [x, self.weight])
        support = support+self.bias
        output = torch.einsum("bnm, bnd->bnd", [adj, support])
        return output

class Block(nn.Module):

    def __init__(self, thetas_dim,  device,batch_size,num_nodes,feature_dim = 1, input_dim = 64, hidden_dim=64, backcast_length=12, forecast_length=12):
        super(Block, self).__init__()
        
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.feature_dim = feature_dim
        self.input_dim = input_dim
        self.gcn = GCN(num_nodes=num_nodes,input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim,dropout=0.2)

        self.device = device
        self.batch_size = batch_size
        self.num_nodes = num_nodes

        self.fc_ht = nn.Linear(64,64)
        self.fc_ct = nn.Linear(64,64)


        self.theta_b_fc = nn.Linear(64*self.backcast_length, thetas_dim*hidden_dim, bias=False)  #(64*12,4*64)
        self.theta_f_fc = nn.Linear(64*self.forecast_length, thetas_dim*self.feature_dim, bias=False)#(64*12,4)

        self.backcast_linspace = np.arange(0,self.backcast_length)/self.backcast_length
        self.forecast_linspace = np.arange(0,self.forecast_length)/self.forecast_length

        self.fc_out = nn.Linear(hidden_dim*backcast_length,64)
        self.fc_fore = nn.Linear(64,self.forecast_length*self.feature_dim)
        
        
    def forward(self, x, adj):
        x = x.reshape(self.batch_size,self.backcast_length,-1)  #（64，12，207*64）
        x = x.permute(0,2,1) #(64,207*64,12)
        batch_size = x.shape[0]
        num_nodes = int(x.shape[1]/self.input_dim)

        ht = x[:,:,0].reshape(batch_size,num_nodes,self.input_dim)  #,self.input_dim
        ct = x[:,:,0].reshape(batch_size,num_nodes,self.input_dim)

        ht = self.fc_ht(ht) #(64,207,64)
        ct = self.fc_ct(ct) #(64,207,64)
        output = ht
        for i in range(self.backcast_length):
            ht, ct = self.gcn(x[:,:,i].reshape(batch_size,num_nodes,self.input_dim), ht,ct, adj)  #(64,207,64)
            output = torch.cat((output,ht),dim=-1)           #(12, 64,207,64)

        output = output[:,:,64:]    #（64，207，64*12）

        return output

    
    
def seasonality_model(thetas, t, device):
    p = thetas.size()[-1]
    #assert p <= thetas.shape[1], 'thetas_dim is too big.'
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    s1 = torch.tensor(np.array([np.cos(2 * np.pi * i * t) for i in range(p1)])).float()  # H/2-1
    s2 = torch.tensor(np.array([np.sin(2 * np.pi * i * t) for i in range(p2)])).float()
    S = torch.cat([s1, s2])
    y = torch.einsum("bfk, kt->bft", [thetas, S.to(device)])
    return y


class SeasonalityBlock(Block):

    def __init__(self,  thetas_dim, device,batch_size,num_nodes, feature_dim, input_dim, hidden_dim, backcast_length=12, forecast_length=12):

        super(SeasonalityBlock, self).__init__(thetas_dim, device,batch_size,num_nodes, feature_dim, input_dim, hidden_dim, backcast_length, forecast_length)
        
    def forward(self, x,adj):
        x = super(SeasonalityBlock, self).forward(x,adj)   

        backcast = self.theta_b_fc(x).reshape(self.batch_size,-1,self.thetas_dim)   #(64,207,64,8)*(8,12)
        forecast = self.theta_f_fc(x).reshape(self.batch_size,-1,self.thetas_dim)    #(64,207,1,8)

        backcast = seasonality_model(backcast, self.backcast_linspace, self.device)
        forecast = seasonality_model(forecast, self.forecast_linspace, self.device)
        backcast = backcast.reshape(self.batch_size,self.num_nodes,-1,self.backcast_length).permute(0,3,1,2)
        forecast = forecast.reshape(self.batch_size,self.num_nodes,-1,self.forecast_length).permute(0,3,1,2)
        
        return backcast, forecast


class TrendBlock(Block):

    def __init__(self,thetas_dim, device,batch_size,num_nodes, feature_dim,input_dim, hidden_dim, backcast_length=12, forecast_length=12):
        super(TrendBlock, self).__init__(thetas_dim, device, batch_size,num_nodes,feature_dim, input_dim, hidden_dim, backcast_length, forecast_length)

    def forward(self, x, adj):
        x = super(TrendBlock, self).forward(x,adj)  #(64,207,64*12)

        backcast = self.theta_b_fc(x).reshape(self.batch_size,-1,self.thetas_dim)    #(64,207*64,8)*(8,12)
        forecast = self.theta_f_fc(x).reshape(self.batch_size,-1,self.thetas_dim)    #(64,207*1,8)
        b_T = torch.tensor(np.array([self.backcast_linspace ** i for i in range(self.thetas_dim)])).float()
        f_T = torch.tensor(np.array([self.forecast_linspace ** i for i in range(self.thetas_dim)])).float()

        backcast = torch.einsum("bfk, kt->bft", [backcast, b_T.to(self.device)])  #(64,207*64,12)
        forecast = torch.einsum("bfk, kt->bft", [forecast, f_T.to(self.device)])  #(64,207*1,12)

        backcast = backcast.reshape(self.batch_size,self.num_nodes,-1,self.backcast_length).permute(0,3,1,2)    #(64,12,207,64)
        forecast = forecast.reshape(self.batch_size,self.num_nodes,-1,self.forecast_length).permute(0,3,1,2)    #(64,12,207,64)
        
        return backcast, forecast


class GenericBlock(Block):

    def __init__(self,  thetas_dim, device,batch_size,num_nodes,feature_dim, input_dim,hidden_dim, backcast_length, forecast_length):
        super(GenericBlock, self).__init__( thetas_dim, device,batch_size,num_nodes, feature_dim, input_dim, hidden_dim, backcast_length, forecast_length)

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)
        self.relu = nn.ReLU()

    def forward(self, x,adj):

        x = super(GenericBlock, self).forward(x,adj)

        theta_b = self.theta_b_fc(x).reshape(self.batch_size,-1,self.thetas_dim)    #(64,207*64,8)
        theta_f = self.theta_f_fc(x).reshape(self.batch_size,-1,self.thetas_dim)    #(64,207*1,8)

        backcast = self.backcast_fc(theta_b)  
        forecast = self.forecast_fc(theta_f)  
        backcast = backcast.reshape(self.batch_size,self.num_nodes,-1,self.backcast_length).permute(0,3,1,2)
        forecast = forecast.reshape(self.batch_size,self.num_nodes,-1,self.forecast_length).permute(0,3,1,2)

        return backcast, forecast
