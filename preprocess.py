'''
이 코드에서는 dgl.data 또는 torch.utils.data의 DataLoader, Dataset을 사용하지 않습니다.
미리 모든 데이터셋을 프로세싱하여 로컬에 저장합니다.
'''
# library
import pandas as pd
import numpy as np
import torch
import dgl
from tqdm import tqdm


# local
from utils import normalize_KRRI_data



def processing(type=None):
    assert type in ['straight', 'curve'], 'Data type error'
    type = type[0]
    
    # data : [lenth x 39 cols]
    data_30 = normalize_KRRI_data(pd.read_csv(f'./data/data_{type}30.csv'))
    data_40 = normalize_KRRI_data(pd.read_csv(f'./data/data_{type}40.csv'))
    data_50 = normalize_KRRI_data(pd.read_csv(f'./data/data_{type}50.csv'))
    data_70 = normalize_KRRI_data(pd.read_csv(f'./data/data_{type}70.csv'))
    data_100 = normalize_KRRI_data(pd.read_csv(f'./data/data_{type}100.csv'))

    # lane data
    data_lane = pd.read_csv(f'./data/lane_data_{type}.csv')
    data_lane = (data_lane-data_lane.mean())/data_lane.std()

    # process
    data_list = []
    for idx in tqdm(range(len(data_30)),desc=f'{type} 데이터에서 전처리중...'):
        # graph
        '''
        ~~~~~~~~~~~~~
        (n) : wheel
        [n] : sensor
        ~~~~~~~~~~~~~
        ┌(0) ㅡ (3)┐: front
        | |      | | 
        |[0] ㅡ [1]ㅣ
        | |      | |
        └(1) ㅡ (2)┘ : rear
        '''
        graph = dgl.heterograph({('wheel', 'front', 'wheel'): ([0,3],[3,0]),
                                ('wheel', 'left', 'wheel'): ([0,1],[1,0]),
                                ('wheel', 'right', 'wheel'): ([3,2],[2,3]),
                                ('wheel', 'rear', 'wheel'): ([1,2],[2,1]),
                                ('wheel', 'connect', 'sensor'):([0,1,2,3],[0,0,1,1]),
                                ('sensor', 'connect', 'wheel'):([0,0,1,1],[0,1,2,3])})
        
        ## wheel feature
        feature_wheel_0 = torch.cat((torch.tensor(data_30.iloc[idx,[1,2,3,9,15,19,23,25]])[None,...],
                                     torch.tensor(data_40.iloc[idx,[1,2,3,9,15,19,23,25]])[None,...],
                                     torch.tensor(data_50.iloc[idx,[1,2,3,9,15,19,23,25]])[None,...],
                                     torch.tensor(data_70.iloc[idx,[1,2,3,9,15,19,23,25]])[None,...],
                                     torch.tensor(data_100.iloc[idx,[1,2,3,9,15,19,23,25]])[None,...]),
                                    dim=0) # 좌전륜 # (5,8) shape
        feature_wheel_1 = torch.cat((torch.tensor(data_30.iloc[idx,[1,2,5,11,16,20,27,29]])[None,...],
                                     torch.tensor(data_40.iloc[idx,[1,2,5,11,16,20,27,29]])[None,...],
                                     torch.tensor(data_50.iloc[idx,[1,2,5,11,16,20,27,29]])[None,...],
                                     torch.tensor(data_70.iloc[idx,[1,2,5,11,16,20,27,29]])[None,...],
                                     torch.tensor(data_100.iloc[idx,[1,2,5,11,16,20,27,29]])[None,...]),
                                    dim=0) # 좌후륜 # (5,8) shape
        feature_wheel_2 = torch.cat((torch.tensor(data_30.iloc[idx,[1,2,8,14,18,22,28,30]])[None,...],
                                     torch.tensor(data_40.iloc[idx,[1,2,8,14,18,22,28,30]])[None,...],
                                     torch.tensor(data_50.iloc[idx,[1,2,8,14,18,22,28,30]])[None,...],
                                     torch.tensor(data_70.iloc[idx,[1,2,8,14,18,22,28,30]])[None,...],
                                     torch.tensor(data_100.iloc[idx,[1,2,8,14,18,22,28,30]])[None,...]),
                                    dim=0) # 우후륜 # (5,8) shape
        feature_wheel_3 = torch.cat((torch.tensor(data_30.iloc[idx,[1,2,6,12,17,21,24,26]])[None,...],
                                     torch.tensor(data_40.iloc[idx,[1,2,6,12,17,21,24,26]])[None,...],
                                     torch.tensor(data_50.iloc[idx,[1,2,6,12,17,21,24,26]])[None,...],
                                     torch.tensor(data_70.iloc[idx,[1,2,6,12,17,21,24,26]])[None,...],
                                     torch.tensor(data_100.iloc[idx,[1,2,6,12,17,21,24,26]])[None,...]),
                                    dim=0) # 우전륜 # (5,8) shape
        feature_wheel = torch.cat((feature_wheel_0[None,...],
                                   feature_wheel_1[None,...],
                                   feature_wheel_2[None,...],
                                   feature_wheel_3[None,...]),
                                  dim=0).type(torch.float32) # (4,5,8) shape
        
        ## sensor feature
        feature_sensor_0 = torch.cat((torch.tensor(data_30.iloc[idx,[1,2,4,10]])[None,...],
                                      torch.tensor(data_40.iloc[idx,[1,2,4,10]])[None,...],
                                      torch.tensor(data_50.iloc[idx,[1,2,4,10]])[None,...],
                                      torch.tensor(data_70.iloc[idx,[1,2,4,10]])[None,...],
                                      torch.tensor(data_100.iloc[idx,[1,2,4,10]])[None,...]),
                                      dim=0) # 좌sensor # (5,4) shape
        feature_sensor_1 = torch.cat((torch.tensor(data_30.iloc[idx,[1,2,7,13]])[None,...],
                                      torch.tensor(data_40.iloc[idx,[1,2,7,13]])[None,...],
                                      torch.tensor(data_50.iloc[idx,[1,2,7,13]])[None,...],
                                      torch.tensor(data_70.iloc[idx,[1,2,7,13]])[None,...],
                                      torch.tensor(data_100.iloc[idx,[1,2,7,13]])[None,...]),
                                      dim=0) # 우sensor # (5,4) shape
        feature_sensor = torch.cat((feature_sensor_0[None,...],
                                    feature_sensor_1[None,...]),
                                    dim=0).type(torch.float32) # (2,5,4) shape
        ## feature 삽입
        graph.nodes['wheel'].data['feature'] = feature_wheel
        graph.nodes['sensor'].data['feature'] = feature_sensor
        
        # 기타 variables
        ## distance
        distance = data_30.iloc[idx,0] # scalar
        ## norm_target
        norm_target = torch.tensor([data_30.iloc[idx,-8:-4].values,
                                    data_40.iloc[idx,-8:-4].values,
                                    data_50.iloc[idx,-8:-4].values,
                                    data_70.iloc[idx,-8:-4].values,
                                    data_100.iloc[idx,-8:-4].values]).type(torch.float32) # (5, 4) shape
        ## target
        target = torch.tensor([data_30.iloc[idx,-4:].values,
                               data_40.iloc[idx,-4:].values,
                               data_50.iloc[idx,-4:].values,
                               data_70.iloc[idx,-4:].values,
                               data_100.iloc[idx,-4:].values]).type(torch.float32) # (5, 4) shape
        
        ## lane
        lane = torch.tensor(data_lane.iloc[idx,1:].values).type(torch.float32) if type=='s' else [] # 5 length for straight

        # append
        data_list.append({'distance':distance,
                          'graph':graph,
                          'lane':lane,
                          'norm_target':norm_target,
                          'target':target})

    # save
    np.save(f'./data/{type}_data.npy', np.array(data_list))





def get_mean_std(df):
    mean = torch.zeros(4)
    std = torch.zeros(4)
    mean[0] = df['YL_M1_B1_W1'][0:10001].mean()
    mean[1] = df['YR_M1_B1_W1'][0:10001].mean()
    mean[2] = df['YL_M1_B1_W2'][0:10001].mean()
    mean[3] = df['YR_M1_B1_W2'][0:10001].mean()
    std[0] = df['YL_M1_B1_W1'][0:10001].std()
    std[1] = df['YR_M1_B1_W1'][0:10001].std()
    std[2] = df['YL_M1_B1_W2'][0:10001].std()
    std[3] = df['YR_M1_B1_W2'][0:10001].std()
    mean = mean.type(torch.float32)
    std = std.type(torch.float32)
    return mean, std




if __name__ == '__main__':
    processing('curve')
    processing('straight')