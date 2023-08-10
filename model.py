# library
import dgl
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import copy



# Railway models
class CurveModel(nn.Module):
    def __init__(self,aggregate='sum',rnn='transformer',ws=10,gpu=True,hidden_size=20,bidirectional=True,**kwargs):
        super().__init__()
        assert rnn in ['lstm', 'transformer']
        
        # variables
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
        self.rnn_shape_embedding = nn.Linear(37,28)
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
        distance : torch.tensor, (ws, 1) shape
        graph : list, [graph_1, graph_2, ..., graph_ws]
                each graph contains ['wheel'] feature which has (4,5,8) shape and
                                    ['sensor'] feature which has (2,5,4) shape.
        norm_target : torch.tensor, (ws, 5, 4) shape
        damper : a scalar in [30,40,50,70,100]
        '''
        # graph
        copyed_graph = copy.deepcopy(graph)
        wheels = torch.zeros((self.ws,4,4))
        sensors = torch.zeros((self.ws,2,4))
        for i,g in enumerate(copyed_graph):
            # reshape
            g.nodes['wheel'].data['feature'] = rearrange(g.nodes['wheel'].data['feature'],
                                                         'w d f -> w (d f)')
            g.nodes['sensor'].data['feature'] = rearrange(g.nodes['sensor'].data['feature'],
                                                          'w d f -> w (d f)')
            # forward
            g.nodes['wheel'].data['feature'] = self.feature_embedding_wheel(g.nodes['wheel'].data['feature'])
            g.nodes['sensor'].data['feature'] = self.feature_embedding_sensor(g.nodes['sensor'].data['feature'])
            h0 = {'wheel':g.nodes['wheel'].data['feature'],
                  'sensor':g.nodes['sensor'].data['feature']}
            h1 = self.conv1(g, h0)
            h1['wheel'] = self.reaky_relu(h1['wheel'])
            h1['sensor'] = self.reaky_relu(h1['sensor'])
            h2 = self.conv2(g, h1)

            # concat
            wheels[i] = h2['wheel']
            sensors[i] = h2['sensor']
        wheels = rearrange(wheels, 'ws w d -> ws (w d)') # (ws, 16)
        sensors = rearrange(sensors, 'ws w d -> ws (w d)') # (ws, 8)
        wheels = wheels.cuda() if self.gpu else wheels
        sensors = sensors.cuda() if self.gpu else sensors

        # other variables
        damper = self.damper_embedding(torch.LongTensor([self.damper_dict[damper]]).cuda()) # (1, 7)
        damper = torch.squeeze(damper) # (7)
        damper = repeat(damper, 'd -> ws d', ws=self.ws) # (ws, 7)
        norm_target = rearrange(norm_target, 'ws dampers targets -> ws (dampers targets)') # (ws, 20)
        norm_target = self.norm_target_embedding(norm_target) # (ws, 5)
        
        # concat all variables
        x = torch.cat([distance, wheels, sensors, norm_target, damper], dim=1) # (ws, 1+16+8+5+7) = (ws, 37)
        x = x[None,...] # (1, ws, 37) # make it as 1 batch
        x = self.rnn_shape_embedding(x)

        # forward to RNN model and inference
        if self.rnn == 'lstm':
            x,_ = self.lstm(x)
            x = rearrange(x, 'b ws (hs bi) -> b (ws hs bi)', bi=self.bi) # (1, ws*hidden_size*bi)
            x = self.linear(x)
        elif self.rnn == 'transformer':
            x = self.transformer(x)
            x = rearrange(x, 'b ws d -> b (ws d)') # (1, 280)
            x = self.linear(x)

        return x # (1, 4)
        




class StraightModel(nn.Module):
    def __init__(self,aggregate='sum',rnn='transformer',ws=10,gpu=True,hidden_size=20,bidirectional=True,**kwargs):
        super().__init__()
        assert rnn in ['lstm', 'transformer']
        
        # variables
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
        self.lane_embedding = nn.Linear(5,3)
        self.norm_target_embedding = nn.Sequential(nn.Linear(20,10),
                                                   nn.ReLU(),
                                                   nn.Linear(10,5))
        self.rnn_shape_embedding = nn.Linear(40,28)
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
        distance : torch.tensor, (ws, 1) shape
        lane : torch.tensor, (ws, 5) shape
        graph : list, [graph_1, graph_2, ..., graph_ws]
                each graph contains ['wheel'] feature which has (4,5,8) shape and
                                    ['sensor'] feature which has (2,5,4) shape.
        norm_target : torch.tensor, (ws, 5, 4) shape
        damper : a scalar in [30,40,50,70,100]
        '''
        # graph
        copyed_graph = copy.deepcopy(graph)
        wheels = torch.zeros((self.ws,4,4))
        sensors = torch.zeros((self.ws,2,4))
        for i,g in enumerate(copyed_graph):
            # reshape
            # print(g.nodes['wheel'].data['feature'].shape)
            # print(g.nodes['sensor'].data['feature'].shape)
            g.nodes['wheel'].data['feature'] = rearrange(g.nodes['wheel'].data['feature'],
                                                         'w d f -> w (d f)')
            g.nodes['sensor'].data['feature'] = rearrange(g.nodes['sensor'].data['feature'],
                                                          'w d f -> w (d f)')
            # forward
            g.nodes['wheel'].data['feature'] = self.feature_embedding_wheel(g.nodes['wheel'].data['feature'])
            g.nodes['sensor'].data['feature'] = self.feature_embedding_sensor(g.nodes['sensor'].data['feature'])
            h0 = {'wheel':g.nodes['wheel'].data['feature'],
                  'sensor':g.nodes['sensor'].data['feature']}
            h1 = self.conv1(g, h0)
            h1['wheel'] = self.reaky_relu(h1['wheel'])
            h1['sensor'] = self.reaky_relu(h1['sensor'])
            h2 = self.conv2(g, h1)

            # concat
            wheels[i] = h2['wheel']
            sensors[i] = h2['sensor']
        wheels = rearrange(wheels, 'ws w d -> ws (w d)') # (ws, 16)
        sensors = rearrange(sensors, 'ws w d -> ws (w d)') # (ws, 8)
        wheels = wheels.cuda() if self.gpu else wheels
        sensors = sensors.cuda() if self.gpu else sensors

        # other variables
        damper = self.damper_embedding(torch.LongTensor([self.damper_dict[damper]]).cuda()) # (1, 7)
        damper = torch.squeeze(damper) # (7)
        damper = repeat(damper, 'd -> ws d', ws=self.ws) # (ws, 7)
        lane = self.lane_embedding(lane) # (ws, 3)
        norm_target = rearrange(norm_target, 'ws dampers targets -> ws (dampers targets)') # (ws, 20)
        norm_target = self.norm_target_embedding(norm_target) # (ws, 5)
        
        # concat all variables
        x = torch.cat([distance, lane, wheels, sensors, norm_target, damper], dim=1) # (ws, 1+3+16+8+5+7) = (ws, 40)
        x = x[None,...] # (1, ws, 40) # make it as 1 batch
        x = self.rnn_shape_embedding(x)

        # forward to RNN model and inference
        if self.rnn == 'lstm':
            x,_ = self.lstm(x)
            x = rearrange(x, 'b ws (hs bi) -> b (ws hs bi)', bi=self.bi) # (1, ws*hidden_size*bi)
            x = self.linear(x)
        elif self.rnn == 'transformer':
            x = self.transformer(x)
            x = rearrange(x, 'b ws d -> b (ws d)') # (1, 280)
            x = self.linear(x)

        return x # (1, 4)
    



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
    # dummy data
    ws = 10
    distance = torch.randn((ws,1)).cuda()
    norm_target = torch.randn((ws,5,4)).cuda()
    lane = torch.randn((ws,5)).cuda()
    graph_transformer = []
    graph_lstm = []
    for i in range(ws):
        g = dgl.heterograph({('wheel', 'front', 'wheel'): ([0,3],[3,0]),
                             ('wheel', 'left', 'wheel'): ([0,1],[1,0]),
                             ('wheel', 'right', 'wheel'): ([3,2],[2,3]),
                             ('wheel', 'rear', 'wheel'): ([1,2],[2,1]),
                             ('wheel', 'connect', 'sensor'):([0,1,2,3],[0,0,1,1]),
                             ('sensor', 'connect', 'wheel'):([0,0,1,1],[0,1,2,3])})
        g.nodes['wheel'].data['feature'] = torch.randn((4,5,8))
        g.nodes['sensor'].data['feature'] = torch.randn((2,5,4))
        g = g.to(torch.cuda.current_device())
        graph_transformer.append(g)
        graph_lstm.append(g.clone()) # g의 다른 메모리 주소를 참조

    # Curve model forward
    curvemodel_transformer = CurveModel().cuda()
    curvemodel_lstm = CurveModel(rnn='lstm').cuda()
    pred_y_transformer = curvemodel_transformer(distance, lane, graph_transformer, norm_target, 30)
    pred_y_lstm = curvemodel_lstm(distance, lane, graph_lstm, norm_target, 100)
    print(f'Curve transformer : {pred_y_transformer.shape}')
    print(f'Curve lstm : {pred_y_lstm.shape}')
    del curvemodel_lstm, curvemodel_transformer

    # dummy data
    graph_transformer = []
    graph_lstm = []
    for i in range(ws):
        g = dgl.heterograph({('wheel', 'front', 'wheel'): ([0,3],[3,0]),
                             ('wheel', 'left', 'wheel'): ([0,1],[1,0]),
                             ('wheel', 'right', 'wheel'): ([3,2],[2,3]),
                             ('wheel', 'rear', 'wheel'): ([1,2],[2,1]),
                             ('wheel', 'connect', 'sensor'):([0,1,2,3],[0,0,1,1]),
                             ('sensor', 'connect', 'wheel'):([0,0,1,1],[0,1,2,3])})
        g.nodes['wheel'].data['feature'] = torch.randn((4,5,8))
        g.nodes['sensor'].data['feature'] = torch.randn((2,5,4))
        g = g.to(torch.cuda.current_device())
        graph_transformer.append(g)
        graph_lstm.append(g.clone()) # g의 다른 메모리 주소를 참조
    
    # Straight model forward
    straightmodel_transformer = StraightModel().cuda()
    straightmodel_lstm = StraightModel(rnn='lstm').cuda()
    pred_y_transformer = straightmodel_transformer(distance, lane, graph_transformer, norm_target, 50)
    pred_y_lstm = straightmodel_lstm(distance, lane, graph_lstm, norm_target, 70)
    print(f'Straight transformer : {pred_y_transformer.shape}')
    print(f'Straight lstm : {pred_y_lstm.shape}')
    del straightmodel_lstm, straightmodel_transformer