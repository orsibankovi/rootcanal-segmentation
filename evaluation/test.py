import math
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torchmetrics
import torchvision
from evaluation import metrics
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class Test():
    def __init__(self, dev: torch.device, batch_size: int, testset: Dataset, net: nn.Module, result_path: str):
        super(Test, self).__init__()
        self.dev = dev
        self.batch_size = batch_size
        self.testset = testset
        self.net = net
        self.results_dir = result_path
        self.Dice = torchmetrics.Dice(zero_division = 1.0, threshold = 0.5).to(self.dev)
        self.Jaccard = torchmetrics.JaccardIndex(task='binary', threshold = 0.5).to(self.dev) 
        self.criterion = nn.BCELoss().to(self.dev)
        self.ROC = torchmetrics.ROC(task='binary', thresholds=10).to(self.dev)
        self.results = {
            'BCE Loss': [], 
            'Dice loss': [], 
            'Jaccard index': [], 
            'Euclidean dist': []
            }
        self.results_roots = {
            'BCE Loss': [], 
            'Dice loss': [], 
            'Jaccard index': [], 
            'Euclidean dist': []
            }
        
    def save_images(self, input: torch.Tensor, output_tensor: torch.Tensor, target_tensor: torch.Tensor, count: int) -> None:
        tf_tn_img = metrics.tf_fn_draw(input[:, 2, :, :], output_tensor.squeeze(1), target_tensor.squeeze(1))
        tf_tn_img.save(self.results_dir + '/visualization/' + str(count) + '.png')
        
    def test(self) -> None:
        test_loader = torch.utils.data.DataLoader(self.testset, self.batch_size, shuffle=False)
        tpr, fpr = [], []

        print('test' + '\n')
        self.net.train(False)

        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                data = data.float()
                target = target.float()
                if self.dev.type == 'cuda':
                    data = data.to(self.dev)
                    target = target.to(self.dev)

                output = self.net(data)
                loss = self.criterion(output, target)
                #self.save_images(data.cpu().data, torch.round(output.cpu().data), target.cpu().data.int(), i)
                if np.max(target.cpu().data.int().numpy()) != 0:  #or np.max(torch.round(output.cpu().data).int().numpy()) != 0:
                    fpr_, tpr_, _ = self.ROC(output, target.int())
                    fpr.append(fpr_.cpu().numpy())
                    tpr.append(tpr_.cpu().numpy())
                
                dice_loss_value = self.Dice(output, target.int())
                jaccard_value = self.Jaccard(output, target.int())

                e, outX, outY, targetX, targetY = metrics.centers_of_canals(torch.round(output.cpu().data), target.cpu().int())
                #metrics.draw_center_of_canal(data.cpu().data[:, 2, :, :], outX, outY, targetX, targetY).save(self.results_dir + '/visualization_roots/' + str(i) + '.png')
                
                self.results['BCE Loss'].append(loss.item())
                self.results['Dice loss'].append(dice_loss_value.item())
                self.results['Jaccard index'] += [jaccard_value.item()] \
                    if math.isnan(jaccard_value.item()) == False else [1]
                self.results['Euclidean dist'] += [e] if e != 'nan' else ['']
                
                if e != 'nan':
                    self.results_roots['BCE Loss'].append(loss.item())
                    self.results_roots['Dice loss'].append(dice_loss_value.item())
                    self.results_roots['Jaccard index'] += [jaccard_value.item()] \
                        if math.isnan(jaccard_value.item()) == False else [1]
                    self.results_roots['Euclidean dist'] += [e]
                    
        fpr_avg = np.average(np.asarray(fpr), axis=0)
        tpr_avg = np.average(np.asarray(tpr), axis=0)
                
        plt.figure(7)
        auc=np.trapz(tpr_avg, x=fpr_avg, dx=0.01)
        plt.plot(fpr_avg, tpr_avg, label='AUC='+str(round(auc, 4)))
        plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), label='baseline', linestyle='--')

        plt.title('ROC Curve', fontsize=18)
        plt.ylabel('TPR', fontsize=16)
        plt.xlabel('FPR', fontsize=16)
        plt.legend(fontsize=12, loc='lower right')
        plt.savefig(self.results_dir + '/ROC.png')
                
        print('BCE Loss: ' + str(np.average(self.results['BCE Loss'])))
        print('Dice Loss: ' +   str(np.average(self.results['Dice loss'])))
        print('Jaccard Index: ' +   str(np.average(self.results['Jaccard index'])))  
        
        df = pd.DataFrame(self.results)
        #df.to_csv(self.results_dir +  '/results.csv', index=True)
        
        df_roots = pd.DataFrame(self.results_roots)
        #df_roots.to_csv(self.results_dir + '/results_roots.csv', index=True)