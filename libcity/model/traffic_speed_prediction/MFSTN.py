from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
import math
import torch
import torch.nn as nn
import numpy as np
from logging import getLogger
from libcity.model import loss
from torch.nn import functional as F


class GCN(nn.Module):
    def __init__(self,feature_dim, num_nodes, input_dim=64, hidden_dim=64,dropout=0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.num_nodes = num_nodes
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim,input_dim)   #((self.feature_dim,input_dim))
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
        self.gc3 = GraphConvolution(self.hidden_dim,self.hidden_dim)

        self.dgc1 = DynamicGraphConvolution(self.hidden_dim,self.hidden_dim)
        self.dgc2 = DynamicGraphConvolution(self.hidden_dim,self.hidden_dim)
        self.bn = nn.BatchNorm1d(64)
        self.ln = nn.LayerNorm([self.num_nodes,64])
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


        state1 = x + state1 + dstate1

        
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
        output = torch.einsum("ij, bjf->bif", [adj, support])
        output = output+self.bias
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


class MFSTN(AbstractTrafficStateModel):
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


        if self.add_time_in_day == True:
            self.time_day_embedding = nn.Linear(1,self.feature_dim)
        if self.add_day_in_week == True:
            self.day_week_embedding = nn.Linear(7,self.feature_dim)

        self.gcn = GCN(feature_dim=self.feature_dim,num_nodes=self.num_nodes, input_dim=self.input_dim, hidden_dim=self.hidden_dim, dropout=self.dropout)

        self.fc_ox = nn.Linear(self.feature_dim,64)

        self.fc_ht = nn.Linear(64,64)
        self.fc_ct = nn.Linear(64,64)


        self.fc_out = nn.Linear(64*12,self.forecast_length*self.feature_dim)
        self.adj = self.calculate_laplacian(self.adj)



        
    def forward(self, batch):
        x = batch['X']  # [batch_size, input_window, num_nodes, feature_dim]

        original_x = x[:,:,:,:self.feature_dim]
       

        x_embedding = self.fc_ox(original_x)    #(64,12,207,3)->(64,12,207,64)
        if self.add_time_in_day == True:
            time_in_day = x[:,:,:,self.feature_dim].unsqueeze(-1)
            x_embedding = x_embedding + self.time_day_embedding(time_in_day)
        if self.add_day_in_week == True:
            day_in_week = x[:,:,:,self.feature_dim+int(self.add_time_in_day):]
            x_embedding = x_embedding + self.day_week_embedding(day_in_week)


        x = x_embedding

        x = x.reshape(self.batch_size,self.backcast_length,-1)  #(64,12,207*64)
        x = x.permute(0,2,1)
      
        ht = x[:,:,0].reshape(self.batch_size,self.num_nodes,self.input_dim)  #,self.input_dim
        ct = x[:,:,0].reshape(self.batch_size,self.num_nodes,self.input_dim)

        ht = self.fc_ht(ht) #(64,207,64)
        ct = self.fc_ct(ct) #(64,207,64)
        output = ht
        for i in range(self.backcast_length):
            ht, ct = self.gcn(x[:,:,i].reshape(self.batch_size,self.num_nodes,self.input_dim), ht,ct, self.adj)  #(64,207,64)
            output = torch.cat((output,ht),dim=-1)           #(12, 64,207,64)

        output = output[:,:,64:]
        output = self.fc_out(output)
        output = output.reshape(self.batch_size,self.num_nodes, self.backcast_length,-1)

        output = output.permute(0,2,1,3)

        return output

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted)
        
        loss_predict = loss.masked_mae_torch(y_predicted, y_true, 0)

        return loss_predict
    
    
    
    def predict(self, batch):
        return self.forward(batch)
    
    def calculate_laplacian(self, adj):
        adj = torch.tensor(adj).to(self.device)
        adj = adj + torch.eye(adj.size(0)).to(self.device)
        d = adj.sum(1)
        d_inv_sqrt = torch.pow(d, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        normalized_laplacian = adj.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
        return normalized_laplacian
