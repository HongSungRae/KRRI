# library
import pandas as pd
import os
from matplotlib import pyplot as plt
from collections.abc import Iterable
import numpy as np


def normalize_KRRI_data(df):
    '''
    - df의 0번 열은 timestamp
    - 1번부터 30번 열은 features
    - 31,32,33,34번 열은 target이다.
    - target 열은 추론해야하는 부분이 모두 0이므로 평균 분산 계산에서는 따지지 말아야한다
    '''
    feature = df.iloc[0:,0:31] # 시간도 그냥 표준화함
    feature = (feature-feature.mean())/feature.std()
    norm_target = df.iloc[0:,31:]
    norm_target.rename(columns={'YR_M1_B1_W2':'norm_YR_M1_B1_W2',
                                'YL_M1_B1_W2':'norm_YL_M1_B1_W2',
                                'YR_M1_B1_W1':'norm_YR_M1_B1_W1',
                                'YL_M1_B1_W1':'norm_YL_M1_B1_W1'},
                       inplace=True)
    norm_target.iloc[0:10001,0:] = (norm_target.iloc[0:10001,0:]-norm_target.iloc[0:10001,0:].mean())/norm_target.iloc[0:10001,0:].std()
    target = df.iloc[0:,31:]
    processed_df = pd.concat([feature, norm_target, target], axis=1, ignore_index=False)
    return processed_df



def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)



class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sum_2 = 0  # sum of squares
        self.count = 0
        self.std = 0

    def update(self, val, n=1):
        if val != None:  # update if val is not None
            self.val = val
            self.sum += val * n
            self.sum_2 += val ** 2 * n
            self.count += n
            self.avg = self.sum / self.count
            self.std = np.sqrt(self.sum_2 / self.count - self.avg ** 2)
        else:
            pass




class Logger(object):
    def __init__(self, path, int_form=':03d', float_form=':.4f'):
        self.path = path
        self.int_form = int_form
        self.float_form = float_form
        self.width = 0

    def __len__(self):
        try:
            return len(self.read())
        except:
            return 0

    def write(self, values):
        if not isinstance(values, Iterable):
            values = [values]
        if self.width == 0:
            self.width = len(values)
        assert self.width == len(values), 'Inconsistent number of items.'
        line = ''
        for v in values:
            if isinstance(v, int):
                line += '{{{}}} '.format(self.int_form).format(v)
            elif isinstance(v, float):
                line += '{{{}}} '.format(self.float_form).format(v)
            elif isinstance(v, str):
                line += '{} '.format(v)
            else:
                raise Exception('Not supported type.', v)
        with open(self.path, 'a') as f:
            f.write(line[:-1] + '\n')

    def read(self):
        with open(self.path, 'r') as f:
            log = []
            for line in f:
                values = []
                for v in line.split(' '):
                    try:
                        v = float(v)
                    except:
                        pass
                    values.append(v)
                log.append(values)
        return log



def draw_curve(work_dir, train_logger, test_logger):
        train_logger = train_logger.read()
        test_logger = test_logger.read()
        epoch, train_loss = zip(*train_logger)
        epoch,test_loss = zip(*test_logger)

        plt.plot(epoch, train_loss, color='blue', label="Train Loss")
        plt.plot(epoch, test_loss, color='red', label="Test Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title("Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(work_dir + '/loss_curve.png')
        plt.close()




if __name__ == '__main__':
    df = pd.read_csv('./data/data_c30.csv')
    df = normalize_KRRI_data(df)
    print(df.head(10))