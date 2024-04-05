import math
import torch
import pandas as pd
import torch.nn as nn
import torchmetrics
from evaluation import metrics
from torch.utils.data import Dataset

class Test():
    def __init__(self, dev: torch.device, batch_size: int, testset: Dataset, net: nn.Module):
        super(Test, self).__init__()
        self.dev = dev
        self.batch_size = batch_size
        self.testset = testset
        self.net = net
        self.Dice = torchmetrics.Dice(zero_division = 1.0, threshold = 0.5).to(self.dev)
        self.Jaccard = torchmetrics.JaccardIndex(task='binary', threshold = 0.5).to(self.dev) 
        self.criterion = nn.BCELoss().to(self.dev)
        self.results = {
            'BCE Loss': [], 
            'Dice loss': [], 
            'Jaccard index': [], 
            'Euclidean dist': []
            }
        
    def test(self) -> None:
        test_loader = torch.utils.data.DataLoader(self.testset, self.batch_size, shuffle=False)

        print('test' + '\n')
        self.net.train(False)

        with torch.no_grad():
            for data, target in test_loader:
                if self.dev.type == 'cuda':
                    data = data.to(self.dev)
                    target = target.to(self.dev)

                output = self.net(data)
                loss = self.criterion(output, target)
                
                dice_loss_value = self.Dice(output, target.int())
                jaccard_value = self.Jaccard(output, target.int())

                e = metrics.centers_of_canals(torch.round(output.cpu().data), target.cpu().int())
                self.results['BCE Loss'].append(loss.item())
                self.results['Dice loss'].append(dice_loss_value.item())
                self.results['Jaccard index'] += [jaccard_value.item()] \
                    if math.isnan(jaccard_value.item()) == False else [1]
                self.results['Euclidean dist'] += [e] if e != 'nan' else ['']
                
        
        df = pd.DataFrame(self.results)
        df.to_csv('C:/Users/orsolya.bankovi/Documents/Uni/rootcanal_segmentation/results.csv', index=True)