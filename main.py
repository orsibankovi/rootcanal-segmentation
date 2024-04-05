import torch
import preprocess.get_dataset as dataset
from models import unet
from train import train
from evaluation import test


if __name__ == '__main__':
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = unet.UNet(in_channels=3, out_channels=1, init_features=32).to(dev)
    
    data_set = dataset.GetDataset('C:/Users/orsolya.bankovi/Documents/Uni/rootcanal_segmentation/segmentation')
    trainsize = int(0.8 * len(data_set))
    valsize = len(data_set) - trainsize
    testsize = valsize - int(0.5 * valsize)
    
    train_set, val_set = torch.utils.data.random_split(data_set, [trainsize, valsize])
    test_set, val_set = torch.utils.data.random_split(val_set, [testsize, valsize - testsize])
    train_class = train.Train(dev, 1, 4, 0.001, net)
    trained_model = train_class.train(train_set, val_set, net)
    
    
    test_class = test.Test(dev, 1, test_set, trained_model)
    test_class.test()
    
    torch.save(trained_model, 'C:/Users/orsolya.bankovi/Documents/Uni/rootcanal_segmentation/UNet.pth')
