import torch
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
import numpy as np
import dgl



class Dataset(DGLDataset):
    def __init__(self, type=None, ws=None, split=None):
        super().__init__(name='dataset')
        assert type != None, 'type should be in ["straight", "curve"]'
        assert ws != None, 'Error : window size should NOT be None type!'
        assert split in ['train', 'inference'], 'Dataset split error'

        # init variables
        self.type = type
        self.ws = ws
        self.split = split

        # load data
        self.data = np.load('./data/s_data.npy', allow_pickle=True) if type=='straight' else np.load('./data/c_data.npy', allow_pickle=True)
        if split == 'inference':
            self.data = self.data[10001-ws+1:10001-ws+1+1999]

    def __len__(self):
        if self.split == 'train':
            return 10001-self.ws+1
        else: # 'inference'
            return 1999

    def __getitem__(self, idx):
        distance = torch.zeros((self.ws, 1))
        lane = torch.zeros((self.ws, 5))
        norm_target = torch.zeros((self.ws, 5, 4))
        feature_wheel = torch.zeros((4, self.ws, 5, 8))
        feature_sensor = torch.zeros((2, self.ws , 5, 4))
        graph = dgl.heterograph({('wheel', 'front', 'wheel'): ([0,3],[3,0]),
                                 ('wheel', 'left', 'wheel'): ([0,1],[1,0]),
                                 ('wheel', 'right', 'wheel'): ([3,2],[2,3]),
                                 ('wheel', 'rear', 'wheel'): ([1,2],[2,1]),
                                 ('wheel', 'connect', 'sensor'):([0,1,2,3],[0,0,1,1]),
                                 ('sensor', 'connect', 'wheel'):([0,0,1,1],[0,1,2,3])})
        for i in range(self.ws):
            distance[i] = self.data[idx+i]['distance']
            feature_wheel[:,i,...] = self.data[idx+1]['graph'].nodes['wheel'].data['feature']
            feature_sensor[:,i,...] = self.data[idx+1]['graph'].nodes['sensor'].data['feature']
            lane[i] = self.data[idx+i]['lane'] if self.type=='straight' else torch.zeros(5)
            norm_target[i] = self.data[idx+i]['norm_target']
        graph.nodes['wheel'].data['feature'] = feature_wheel.type(torch.float32)
        graph.nodes['sensor'].data['feature'] = feature_sensor.type(torch.float32)
        distance = distance.type(torch.float32)
        lane = lane.type(torch.float32)
        norm_target = norm_target.type(torch.float32)

        target = self.data[idx+self.ws]['target'].type(torch.float32)
        return distance, lane, graph, norm_target, target



def test_dataset(type, ws, split):
    dataset = Dataset(type, ws, split)
    dataloader = GraphDataLoader(dataset, batch_size=32, shuffle=True, drop_last=False)
    distance, lane, graph, norm_target, target = next(iter(dataloader))
    print(f'======= Test on [type : {type}, ws : {ws}, split : {split}] ======')
    print(f'dataset length when ws is {ws} : {len(dataset)}')
    print(f'distance : {distance.shape}')
    print(f'lane : {lane.shape}')
    print(f'norm_target : {norm_target.shape}')
    print(f'target : {target.shape}')
    print(f'graph/wheel : {graph.nodes["wheel"].data["feature"].shape}')
    print(f'graph/sensor : {graph.nodes["sensor"].data["feature"].shape}')
    print('==================================================================\n')



if __name__ == '__main__':
    test_dataset('straight', 10, 'train')
    test_dataset('straight', 10, 'inference')
    test_dataset('curve', 10, 'train')
    test_dataset('curve', 10, 'inference')
    '''
    ======= Test on [type : straight, ws : 10, split : train] ======
    dataset length when ws is 10 : 9992
    distance : torch.Size([32, 10, 1])
    lane : torch.Size([32, 10, 5])
    norm_target : torch.Size([32, 10, 5, 4])
    target : torch.Size([32, 4])
    graph/wheel : torch.Size([128, 10, 5, 8])
    graph/sensor : torch.Size([64, 10, 5, 4])
    ==================================================================
    '''