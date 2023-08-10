# library
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import torch.optim as optim
import argparse
import os
import json


# local
from model import CurveModel, StraightModel
from utils import Logger, make_dir, draw_curve, AverageMeter
from preprocess import processing
from metric import WeightedMAPE
from infer import inference

# parser
parser = argparse.ArgumentParser(description='KRRI')

## code configuration
parser.add_argument('--experiment', '--e', default='', type=str,
                    help='Experiment name')
parser.add_argument('--type', type=str, default='', choices=['straight', 'curve'],
                    help='직선모델 또는 곡선모델')

## model configuration
parser.add_argument('--rnn', default='transformer', type=str, choices=['lstm', 'transformer'],
                    help='RNN model')
parser.add_argument('--ws', default=10, type=int,
                    help='window size')
parser.add_argument('--hidden_size', default=20, type=int,
                    help='LSTM hidden_size')
parser.add_argument('--aggregate','--a', default='mean', type=str, choices=['sum', 'mean', 'max', 'min'],
                    help='GNN aggregation method')

## training configuration
parser.add_argument('--epochs','--eps', default=100, type=int,
                    help='Training epochs')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='Learning rate')
parser.add_argument('--optim', default='adam', type=str,
                    help='optimizer', choices=['sgd','adam','adagrad'])
parser.add_argument('--lr_decay', default=1e-3, type=float,
                    help='learning rate decay')
parser.add_argument('--weight_decay', default=1e-5, type=float,
                    help='weight_decay')
args = parser.parse_args()


## For single GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



def main():
    start = time.time()
    
    # assertion
    assert args.experiment != '', '실험 이름을 지정하시오.'
    assert args.type != '', '어떤 유형의 선로를 예측할지 지정하시오. ["straight", "curve"]'

    # 전처리
    if not os.path.exists('./data/s_data.npy'):
        processing('straight')
    if not os.path.exists('./data/c_data.npy'):
        processing('curve')


    # exp result dir
    make_dir('./exp')
    make_dir(f'./exp/{args.experiment}')
    with open(f'./exp/{args.experiment}/configuration.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)


    # logger
    train_logger = Logger(f'./exp/{args.experiment}/train_loss.log')


    # (model, data) call
    if args.type == 'straight':
        model = StraightModel(aggregate=args.aggregate,
                              rnn=args.rnn,
                              ws=args.ws,
                              gpu=True,
                              hidden_size=args.hidden_size,
                              bidirectional=True).cuda()
        data = np.load('./data/s_data.npy', allow_pickle=True)
    elif args.type == 'curve':
        model = CurveModel(aggregate=args.aggregate,
                           rnn=args.rnn,
                           ws=args.ws,
                           gpu=True,
                           hidden_size=args.hidden_size,
                           bidirectional=True).cuda()
        data = np.load('./data/c_data.npy', allow_pickle=True)
    

    # loss function
    criterion = torch.nn.MSELoss()


    # define Optimizer
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    milestones = [int(args.epochs/3),int(args.epochs/2)]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.7)


    # train
    for epoch in tqdm(range(args.epochs), desc='Training...'):
        train(model, data, criterion, optimizer, epoch, train_logger)
        scheduler.step()
    else: # 학습이 모두 끝나면
        torch.save(model, f'./exp/{args.experiment}/model.pth')
        draw_curve(f'./exp/{args.experiment}', train_logger, train_logger)

    # inference
    inference(model, data, args.ws, args.experiment, args.type)

    # finish
    print(f'Process Completed : it took {(time.time()-start)/60:.2f} minutes...')






def train(model, data, criterion, optimizer, epoch, train_logger):
    model.train()
    epoch_loss = AverageMeter()
    for idx in tqdm(range(10001-args.ws-1),desc=f'index at epoch {epoch}'):
        target = data[idx+args.ws]['target']
        target = target.cuda()
        # if idx == 0:
        distance = torch.zeros((args.ws,1))
        graph = []
        lane = torch.zeros((args.ws,5))
        norm_target = torch.zeros((args.ws,5,4))
        for i in range(args.ws):
            distance[i] = data[idx+i]['distance']
            g = data[idx+i]['graph']
            g = g.to(torch.cuda.current_device())
            graph.append(g)
            lane[i] = data[idx+i]['lane'] if args.type=='straight' else torch.zeros(5)
            norm_target[i] = data[idx+i]['norm_target']
        distance = distance.cuda()
        lane = lane.cuda()
        norm_target = norm_target.cuda()
        # else: # 첫 성분 삭제 및 마지막에 새로운 시간 추가
        #     graph.pop(0)
        #     g = data[idx+args.ws-1]['graph']
        #     g = g.to(torch.cuda.current_device())
        #     graph.append(g)
        #     distance = torch.cat([distance[1:], torch.tensor([[data[idx+args.ws-1]['distance']]]).cuda()], dim=0)
        #     lane = torch.cat([lane[1:], data[idx+args.ws-1]['lane'].cuda()[None,...]], dim=0) if args.type=='straight' else lane
        #     norm_target = torch.cat([norm_target[1:], data[idx+args.ws-1]['norm_target'].cuda()[None,...]], dim=0)

        distance = distance.type(torch.float32)  
        total_loss = 0
        for i,damper in enumerate([30,40,50,70,100]):
            y_pred = model(distance, lane, graph, norm_target, damper)
            loss = criterion(y_pred, target[i][None,...])
            total_loss += torch.sum(loss) # https://jjdeeplearning.tistory.com/19
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        epoch_loss.update(total_loss.item())

    train_logger.write([epoch+1, epoch_loss.avg])

    


if __name__ == '__main__':
    main()