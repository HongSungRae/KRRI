# library
import dgl
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import copy
from dgl.dataloading import GraphDataLoader


# local
from dataset import Dataset



# Railway models
class Model(nn.Module):
    def __init__(self,type='straight',aggregate='mean',rnn='transformer',ws=10,gpu=True,hidden_size=20,bidirectional=True,**kwargs):
        super().__init__()
        assert type in ['straight', 'curve']
        assert rnn in ['lstm', 'transformer']
        
        # variables
        self.type = type
        self.damper_dict = {30:0, 40:1, 50:2, 70:3, 100:4}
        self.ws = ws
        self.gpu = gpu
        self.rnn = rnn
        self.bi = 2 if bidirectional else 1

        # model
        self.feature_embedding_wheel = nn.Sequential(nn.Linear(40,20),
                                                     nn.LeakyReLU(),
                                                     nn.Linear(20,15))
        self.feature_embedding_sensor = nn.Sequential(nn.Linear(20,18),
                                                      nn.LeakyReLU(),
                                                      nn.Linear(18,15))
        self.conv1 = dglnn.HeteroGraphConv({'front':dglnn.GraphConv(15,8),
                                            'rear':dglnn.GraphConv(15,8),
                                            'right':dglnn.GraphConv(15,8),
                                            'left':dglnn.GraphConv(15,8),
                                            'connect':dglnn.GraphConv(15,8)},
                                           aggregate=aggregate)
        self.conv2 = dglnn.HeteroGraphConv({'front':dglnn.GraphConv(8,4),
                                            'rear':dglnn.GraphConv(8,4),
                                            'right':dglnn.GraphConv(8,4),
                                            'left':dglnn.GraphConv(8,4),
                                            'connect':dglnn.GraphConv(8,4)},
                                           aggregate=aggregate)
        self.reaky_relu = nn.LeakyReLU()
        self.damper_embedding = nn.Embedding(5,7)
        self.norm_target_embedding = nn.Sequential(nn.Linear(20,10),
                                                   nn.ReLU(),
                                                   nn.Linear(10,5))
        self.lane_embedding = nn.Linear(5,3)
        self.rnn_shape_embedding = nn.Linear(37,28) if type=='curve' else nn.Linear(40,28)
        if rnn == 'transformer':
            self.transformer = Transformer(dim=28,
                                           depth=3,
                                           heads=4,
                                           mlp_dim=14)
            self.linear = nn.Sequential(nn.Linear(ws*28, 128),
                                        nn.ReLU(),
                                        nn.Linear(128,32),
                                        nn.ReLU(),
                                        nn.Linear(32,4))
        elif rnn == 'lstm':
            self.lstm = nn.LSTM(input_size=28,
                                hidden_size=hidden_size,
                                num_layers=4,
                                batch_first=True,
                                bidirectional=bidirectional)
            self.linear = nn.Sequential(nn.Linear(ws*hidden_size*self.bi,int(ws*hidden_size*self.bi/2)),
                                        nn.ReLU(),
                                        nn.Linear(int(ws*hidden_size*self.bi/2),50),
                                        nn.ReLU(),
                                        nn.Linear(50,4))
            

    def forward(self, distance, lane, graph, norm_target, damper):
        '''
        distance : torch.tensor, (bs, ws, 1) shape
        graph : dgl.graph type
                graph contains ['wheel'] feature which has (4*bs,ws,5,8) shape and
                               ['sensor'] feature which has (2*bs,ws,5,4) shape.
        norm_target : torch.tensor, (bs, ws, 5, 4) shape
        damper : a scalar in [30,40,50,70,100]
        '''
        # graph
        graph.nodes['wheel'].data['feature'] = rearrange(graph.nodes['wheel'].data['feature'],
                                                         '(bs_and_wheels) ws dampers f -> (bs_and_wheels) ws (dampers f)') # (4*bs, ws, 40)
        graph.nodes['sensor'].data['feature'] = rearrange(graph.nodes['sensor'].data['feature'],
                                                         '(bs_and_sensors) ws dampers f -> (bs_and_sensors) ws (dampers f)') # (2*bs, ws, 40)
        graph.nodes['wheel'].data['feature'] = self.feature_embedding_wheel(graph.nodes['wheel'].data['feature']) # (4*bs, ws, 15)
        graph.nodes['sensor'].data['feature'] = self.feature_embedding_sensor(graph.nodes['sensor'].data['feature']) # (4*bs, ws, 15)
        h0 = {'wheel':graph.nodes['wheel'].data['feature'],
              'sensor':graph.nodes['sensor'].data['feature']}
        h1 = self.conv1(graph, h0)
        h1['wheel'] = self.reaky_relu(h1['wheel'])
        h1['sensor'] = self.reaky_relu(h1['sensor'])
        h2 = self.conv2(graph, h1)
        wheels = rearrange(h2['wheel'], '(bs wheels) ws d -> bs ws (wheels d)', bs=distance.shape[0]) # (bs, ws, 16)
        sensors = rearrange(h2['sensor'], '(bs sensors) ws d -> bs ws (sensors d)', bs=distance.shape[0]) # (bs, ws, 8)

        # other variables
        damper = self.damper_embedding(torch.LongTensor([self.damper_dict[damper]]).cuda()) # (1, 7)
        damper = torch.squeeze(damper) # (7)
        damper = repeat(damper, 'd -> bs ws d', bs=distance.shape[0], ws=self.ws) # (bs, ws, 7)
        norm_target = rearrange(norm_target, 'bs ws dampers targets -> bs ws (dampers targets)') # (bs, ws, 20)
        norm_target = self.norm_target_embedding(norm_target) # (bs, ws, 5)
        
        # concat all variables
        if self.type == 'straight':
            lane = self.lane_embedding(lane) # (bs, ws, 3)
            x = torch.cat([distance, lane, wheels, sensors, norm_target, damper], dim=-1) # (bs, ws, 1+3+16+8+5+7) = (bs, ws, 40)
        else:
            x = torch.cat([distance, wheels, sensors, norm_target, damper], dim=-1) # (bs, ws, 1+16+8+5+7) = (bs, ws, 37)
        
        # forward to RNN model and inference
        x = self.rnn_shape_embedding(x) # (bs, ws, 28)
        if self.rnn == 'lstm':
            x,_ = self.lstm(x)
            x = rearrange(x, 'bs ws (hs bi) -> bs (ws hs bi)', bi=self.bi) # (bs, ws*hidden_size*bi)
            x = self.linear(x)
        elif self.rnn == 'transformer':
            x = self.transformer(x)
            x = rearrange(x, 'bs ws d -> bs (ws d)') # (bs, 280)
            x = self.linear(x)

        return x # (bs, 4)

    



# Transformers
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x




if __name__ == '__main__':
    # Curve Model Test
    ## data
    dataset = Dataset('curve', 10, 'train')
    dataloader = GraphDataLoader(dataset, batch_size=32, shuffle=True, drop_last=False)
    distance, lane, graph, norm_target, target = next(iter(dataloader))
    distance = distance.cuda()
    lane = lane.cuda()
    graph = graph.to(torch.cuda.current_device())
    norm_target = norm_target.cuda()
    target = target.cuda()

    ## model
    curve_trans = Model(type='curve').cuda()
    curve_lstm = Model(type='curve', rnn='lstm')
    pred_curve_trans = curve_trans(distance, lane, copy.deepcopy(graph), norm_target, 30)
    pred_curve_lstm = curve_trans(distance, lane, copy.deepcopy(graph), norm_target, 40)
    print(f'Curve Transformer output : {pred_curve_trans.shape}')
    print(f'Curve LSTM output : {pred_curve_lstm.shape}')
    del curve_lstm, curve_trans, dataset, dataloader


    # Straight Model Test
    ## data
    dataset = Dataset('straight', 10, 'train')
    dataloader = GraphDataLoader(dataset, batch_size=32, shuffle=True, drop_last=False)
    distance, lane, graph, norm_target, target = next(iter(dataloader))
    distance = distance.cuda()
    lane = lane.cuda()
    graph = graph.to(torch.cuda.current_device())
    norm_target = norm_target.cuda()
    target = target.cuda()

    ## model
    straight_trans = Model(type='straight').cuda()
    straight_lstm = Model(type='straight', rnn='lstm')
    pred_straight_trans = straight_trans(distance, lane, copy.deepcopy(graph), norm_target, 50)
    pred_straight_lstm = straight_trans(distance, lane, copy.deepcopy(graph), norm_target, 100)
    print(f'straight Transformer output : {pred_straight_trans.shape}')
    print(f'Straight LSTM output : {pred_straight_lstm.shape}')