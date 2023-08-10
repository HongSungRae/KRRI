class WeightedMAPE:
    '''
    정답 set 공개 후에 완성 예정
    
    == 계산 방법 ==
    예측해야하는 sequence 1999개에 대해서 첫 시퀀스부터 1.0001에서 1.1999까지 weight가 선형 증가
    MAPE = \sum_{n}{(weigt|y_true - y_pred|)/(|y_true|)*(1/n)*100}
    '''
    def __init__(self):
        pass

    def get_mape(self):
        pass