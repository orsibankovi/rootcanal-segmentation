import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchmetrics

def log_print(sub_list: list) -> str:
    return str(round(np.average(sub_list), 4))


class Train():
    def __init__(self, dev: torch.device, n_epoch: int, batch_size: int, lr: float, net: nn.Module):
        super(Train, self).__init__()
        self.dev = dev
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.Dice = torchmetrics.Dice(zero_division=1.0, threshold=0.5).to(self.dev)
        self.Jaccard = torchmetrics.JaccardIndex(task='binary', threshold=0.5).to(self.dev) 
        self.criterion = nn.BCELoss().to(self.dev)
        self.optimizer = optim.Adam(net.parameters(), self.lr)
 
        
    def train(self, trainset:Dataset, validationset: Dataset, net: nn.Module) -> nn.Module:
        interval = 500
        train_loader = torch.utils.data.DataLoader(trainset, self.batch_size, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(validationset, self.batch_size, shuffle=True)
        val_losses, val_dice_losses, val_jaccard_index = [], [], []
        net.train(True)
        best_jaccard = 0.0
        best_net = None

        print('train')
        for epoch in range(1, self.n_epoch + 1):
            train_losses = []
            dice_losses = []
            jaccard_index = []
        
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.float()
                target = target.float()
                if self.dev.type == 'cuda':
                    data = data.to(self.dev)
                    target = target.to(self.dev)
            
                self.optimizer.zero_grad()

                output = net(data)
                loss = self.criterion(output, target)
                
                loss.backward()
                self.optimizer.step()

                dice_loss_value = self.Dice(output, target.int())
                jaccard_value = self.Jaccard(output, target.int())
                train_losses.append(loss.item())
                dice_losses.append(dice_loss_value.item())

                jaccard_index += [1.0] if math.isnan(jaccard_value.item()) else [jaccard_value.item()]

                if batch_idx % interval == 0:
                    if batch_idx != 0:
                        print('Train Epoch: ' + str(epoch)
                            + " batch_idx: " + str(batch_idx)
                            + "\tLoss: " + log_print(train_losses[-interval])
                            + "\tDiceLoss: " + log_print(dice_losses[-interval])
                            + "\tJaccardIndex: " + log_print(jaccard_index[-interval]))

            loss_, dice_, jaccard_ = self.validation(epoch, net, validation_loader)
            val_losses += loss_
            val_dice_losses += dice_
            val_jaccard_index += jaccard_

            if np.average(jaccard_) > best_jaccard:
                best_jaccard = np.average(jaccard_)
                best_net = net
                torch.save(net, 'UNet3D.pth')
                print('Best Epoch: ' + str(epoch) + ' JaccardIndex: ' + str(best_jaccard))
                
        print('Train loss')
        print('DiceLoss: ' + str(np.average(dice_losses)))
        print('JaccardIndex: ' + str(np.average(jaccard_index)))
        
        print('Validation loss')
        print('DiceLoss: ' + str(np.average(val_dice_losses)))
        print('JaccardIndex: ' + str(np.average(val_jaccard_index)))
        return best_net


    def validation(self, epoch: int, net: nn.Module, validation_loader: DataLoader) -> tuple:
        valid_train_losses = []
        valid_dice_losses = []
        valid_jaccard_index = []

        net.train(False)
        print('Validation')
        for data, target in validation_loader:
            data = data.float()
            target = target.float()
            if self.dev.type == 'cuda':
                data = data.to(self.dev)
                target = target.to(self.dev)

            output = net(data)
            loss = self.criterion(output, target)
            dice_loss_value = self.Dice(output, target.int())
            jaccard_value = self.Jaccard(output, target.int())
            valid_train_losses.append(loss.item())
            valid_dice_losses.append(dice_loss_value.item())
            valid_jaccard_index += [1.0] if math.isnan(jaccard_value.item()) else [jaccard_value.item()]

        print('Train Epoch: ' + str(epoch)
            + ' Valid Loss: ' + log_print(valid_train_losses)
            + ' Valid DiceLoss: ' + log_print(valid_dice_losses)
            + ' Valid JaccardIndex: ' + log_print(valid_jaccard_index))

        return valid_train_losses, valid_dice_losses, valid_jaccard_index