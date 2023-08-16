import torch



class WeightedMAPE:
    def __init__(self):
        self.n = 0
        self.sum = 0

    def calc_mape(self, y_pred, y_true, position, n=1):
        self.n += n
        weight = 1.0001 + 0.0001*position
        self.sum += torch.sum(weight*torch.abs(y_true-y_pred)/torch.abs(y_true)).item()
        
    def get_mape(self):
        return self.sum * 1/self.n * 100
    


if __name__ == '__main__':
    w_mape = WeightedMAPE()

    y_true = torch.randn(32,4)
    y_pred = torch.randn(32,4)
    w_mape.calc_mape(y_pred, y_true, 399, y_pred.shape[0])

    y_true = torch.randn(32,4)
    y_pred = torch.randn(32,4)
    w_mape.calc_mape(y_pred, y_true, 400, y_pred.shape[0])

    mape = w_mape.get_mape()
    print(mape)