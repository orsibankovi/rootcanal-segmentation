import torch
import preprocess.get_dataset as dataset
from models import unet
from train import train
from evaluation import test
from pathlib import Path


if __name__ == '__main__':
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root_dir = 'C:/Users/orsolya.bankovi/Documents/Uni/rootcanal_segmentation'
    dataset_dir = root_dir + '/segmentation_all'
    
    train_mode = True
    
    if train_mode:
        train_set = dataset.GetDataset(dataset_dir + '/train')
        val_set = dataset.GetDataset(dataset_dir + '/validation')
        
        print('Train set size: ', len(train_set))
        print('Validation set size: ', len(val_set))
        
        net = unet.UNet(in_channels=3, out_channels=1, init_features=32).to(dev)

        train_class = train.Train(dev, 20, 8, 0.001, net)
        trained_model = train_class.train(train_set, val_set, net)
    else:
        trained_model = torch.load('./UNet3D.pth')
    
    test_set = dataset.GetDataset(dataset_dir + '/test')
    print('Test set size: ', len(test_set))
    
    results_dir = root_dir + '/results'
    if Path(results_dir).exists() == False:
        Path(results_dir).mkdir()
        Path(results_dir + '/visualization').mkdir()
    
    test_class = test.Test(dev, 1, test_set, trained_model, results_dir)
    test_class.test()
