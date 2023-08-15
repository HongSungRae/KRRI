# library
from tqdm import tqdm
import torch
import pandas as pd
import os
import warnings
import argparse
import json
import numpy as np
import dgl
import copy

# local
from utils import make_dir, AverageMeter
from model import Model
from preprocess import get_mean_std




# target의 mean과 std
s_30 = pd.read_csv('./data/data_s30.csv')
s_40 = pd.read_csv('./data/data_s40.csv')
s_50 = pd.read_csv('./data/data_s50.csv')
s_70 = pd.read_csv('./data/data_s70.csv')
s_100 = pd.read_csv('./data/data_s100.csv')

c_30 = pd.read_csv('./data/data_c30.csv')
c_40 = pd.read_csv('./data/data_c40.csv')
c_50 = pd.read_csv('./data/data_c50.csv')
c_70 = pd.read_csv('./data/data_c70.csv')
c_100 = pd.read_csv('./data/data_c100.csv')

s_30_mean, s_30_std = get_mean_std(s_30) # 모두 shape [4]
s_40_mean, s_40_std = get_mean_std(s_40)
s_50_mean, s_50_std = get_mean_std(s_50)
s_70_mean, s_70_std = get_mean_std(s_70)
s_100_mean, s_100_std = get_mean_std(s_100)

c_30_mean, c_30_std = get_mean_std(c_30) # 모두 shape [4]
c_40_mean, c_40_std = get_mean_std(c_40)
c_50_mean, c_50_std = get_mean_std(c_50)
c_70_mean, c_70_std = get_mean_std(c_70)
c_100_mean, c_100_std = get_mean_std(c_100)

mean_std_dic = {'straight':{30:[s_30_mean, s_30_std],
                            40:[s_40_mean, s_40_std],
                            50:[s_50_mean, s_50_std],
                            70:[s_70_mean, s_70_std],
                            100:[s_100_mean, s_100_std]},
                'curve':{30:[c_30_mean, c_30_std],
                         40:[c_40_mean, c_40_std],
                         50:[c_50_mean, c_50_std],
                         70:[c_70_mean, c_70_std],
                         100:[c_100_mean, c_100_std]}}


mean_std_30 = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]
mean_std_40 = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]
mean_std_50 = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]
mean_std_70 = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]
mean_std_100 = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]
mean_std = [mean_std_30,
            mean_std_40,
            mean_std_50,
            mean_std_70,
            mean_std_100]


# argparse
def get_parser():
    parser = argparse.ArgumentParser(description='KRRI inference')
    parser.add_argument('--experiment_straight', '--es', default='', type=str,
                        help='Straight model experiment name')
    parser.add_argument('--experiment_curve', '--ec', default='', type=str,
                        help='Curve model experiment name')
    parser.add_argument('--avg_type', default='inference', type=str, choices=['train', 'inference'],
                        help='추론 단계에서 사용할 target의 mean과 std가 어느 데이터에서 계산되는가')
    args = parser.parse_args()
    return args



def main(args):
    assert args.experiment_straight != '', '추론할 Straight model의 실험 이름을 지정하세요.'
    assert args.experiment_curve != '', '추론할 Curve model의 실험 이름을 지정하세요.'


    # 경로에 추론결과가 없다면 일단 추론
    if not os.path.exists(f'./exp/{args.experiment_straight}/answer.csv'):
        with open(f'./exp/{args.experiment_straight}/configuration.json', 'r') as f:
            configuration = json.load(f)
        type = configuration['type']
        model = torch.load(f'./exp/{args.experiment_straight}/model.pth').cuda()
        inference(model, configuration['ws'], args.experiment_straight, type)
    if not os.path.exists(f'./exp/{args.experiment_curve}/answer.csv'):    
        with open(f'./exp/{args.experiment_curve}/configuration.json', 'r') as f:
            configuration = json.load(f)
        type = configuration['type']
        model = torch.load(f'./exp/{args.experiment_curve}/model.pth').cuda()
        inference(model, configuration['ws'], args.experiment_curve, type)


    # 병합
    merge(args.experiment_straight, args.experiment_curve)





def merge(experiment_straight, experiment_curve):
    make_dir('./answer')
    make_dir(f'./answer/{experiment_straight}_{experiment_curve}')
    answer_straight = pd.read_csv(f'./exp/{experiment_straight}/answer.csv')
    answer_curve = pd.read_csv(f'./exp/{experiment_curve}/answer.csv')
    answer = pd.read_csv('./data/answer_sample.csv')
    answer.iloc[0:,1:1+20] = answer_straight.iloc[0:,1:1+20]
    answer.iloc[0:,21:21+20] = answer_curve.iloc[0:,21:21+20]
    answer.to_csv(f'./answer/{experiment_straight}_{experiment_curve}/answer.csv', index=False)






def inference(model, ws, experiment, type):
    # 이미 추론 결과가 있다면
    if os.path.exists(f'./exp/{experiment}/answer.csv'):
        reply = str(input('이미 추론 결과가 존재합니다. 다시 추론할지 선택하세요. (y/n): ')).lower().strip()
        if reply[0] == 'y':
            pass
        elif reply[0] == 'n':
            print('재추론하지 않습니다. 추론 코드를 종료합니다.')
            return
        else:
            warnings.warn('y 또는 n만 입력하세요.')
            inference(model, ws, experiment, type)
            return
    
    # 없다면, init variables
    data = np.load('./data/s_data.npy', allow_pickle=True) if type=='straight' else np.load('./data/c_data.npy', allow_pickle=True)
    answer = pd.read_csv('./data/answer_sample.csv')
    answer_idx = {30:1, 40:5, 50:9, 70:13, 100:17} if type=='straight' else {30:21, 40:25, 50:29, 70:33, 100:37}
    current_position = 0

    # test 해서 저장

    for start in tqdm(range(10000-ws+1,10000-ws+1+1999)):#tqdm(range(10001-ws)):#tqdm(range(10001-ws+1,10001-ws+1+1999)):
        model.eval()
        distance = torch.zeros((1,ws,1))
        lane = torch.zeros((1,ws,5))
        norm_target = torch.zeros((1,ws,5,4))
        
        # graph
        feature_wheel = torch.zeros((4, ws, 5, 8))
        feature_sensor = torch.zeros((2, ws , 5, 4))
        graph = dgl.heterograph({('wheel', 'front', 'wheel'): ([0,3],[3,0]),
                                 ('wheel', 'left', 'wheel'): ([0,1],[1,0]),
                                 ('wheel', 'right', 'wheel'): ([3,2],[2,3]),
                                 ('wheel', 'rear', 'wheel'): ([1,2],[2,1]),
                                 ('wheel', 'connect', 'sensor'):([0,1,2,3],[0,0,1,1]),
                                 ('sensor', 'connect', 'wheel'):([0,0,1,1],[0,1,2,3])})
        for i in range(ws):
            distance[0,i,...] = data[start+i]['distance']
            feature_wheel[:,i,...] = data[start+i]['graph'].nodes['wheel'].data['feature']
            feature_sensor[:,i,...] = data[start+i]['graph'].nodes['sensor'].data['feature']
            lane[0,i,...] = data[start+i]['lane'] if type=='straight' else torch.zeros(5)
            norm_target[0,i,...] = data[start+i]['norm_target']
        graph.nodes['wheel'].data['feature'] = feature_wheel.type(torch.float32)
        graph.nodes['sensor'].data['feature'] = feature_sensor.type(torch.float32)
        
        # to GPU
        graph = graph.to(torch.cuda.current_device())
        distance = distance.cuda()
        lane = lane.cuda()
        norm_target = norm_target.cuda()

        distances = [copy.deepcopy(distance), copy.deepcopy(distance), copy.deepcopy(distance), copy.deepcopy(distance), copy.deepcopy(distance)]
        lanes = [copy.deepcopy(lane), copy.deepcopy(lane), copy.deepcopy(lane), copy.deepcopy(lane), copy.deepcopy(lane)]
        graphs = [copy.deepcopy(graph), copy.deepcopy(graph), copy.deepcopy(graph), copy.deepcopy(graph), copy.deepcopy(graph)]
        norm_targets = [copy.deepcopy(norm_target), copy.deepcopy(norm_target), copy.deepcopy(norm_target), copy.deepcopy(norm_target), copy.deepcopy(norm_target)]
        for j, damper in enumerate([30,40,50,70,100]):
            model.eval()
            y_pred = model(distances[j], lanes[j], graphs[j], norm_targets[j], damper) # torch.tensor of (1, 4) shape
            answer.iloc[current_position,answer_idx[damper]:answer_idx[damper]+4] = y_pred[0,...].detach().cpu().tolist()
            for k in range(4):
                mean_std[j][k].update(y_pred[0,k].item())
                data[start+ws]['norm_target'][j][k] = (y_pred.detach().cpu().squeeze()[k].item()-mean_std[j][k].avg)/(mean_std[j][k].std+1e-3)
            # norm_y_pred = (y_pred.detach().cpu().squeeze()-mean_std_dic[type][damper][0])/mean_std_dic[type][damper][1]
            # if not (start+ws==12000):
                # data[start+ws]['norm_target'][j] = norm_y_pred.type(torch.float32)
        current_position += 1
    answer.to_csv(f'./exp/{experiment}/answer.csv', index=False)





if __name__ == '__main__':
    args = get_parser()
    main(args)