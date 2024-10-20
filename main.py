import torch
import preprocess.get_dataset as dataset
from models import unet
from train import train
from evaluation import test
from pathlib import Path
from tempfile import TemporaryDirectory


if __name__ == "__main__":
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root_dir = "C:/Users/orsolya.bankovi/Documents/Uni/rootcanal_segmentation"
    dataset_dir = root_dir + "/segmentation_cropped"
    results_dir = root_dir + "/results/new_structure_original"

    if Path(results_dir).exists() == False:
        Path(results_dir).mkdir()

    if Path(results_dir + "/visualization").exists() == False:
        Path(results_dir + "/visualization").mkdir()

    if Path(results_dir + "/visualization_roots").exists() == False:
        Path(results_dir + "/visualization_roots").mkdir()

    train_mode = False

    if train_mode:
        with TemporaryDirectory(dir=results_dir) as tmp_dir:
            train_set = dataset.GetDataset(
                dataset_dir + "/train", tmp_dir=tmp_dir, mode="train"
            )
            val_set = dataset.GetDataset(
                dataset_dir + "/validation", tmp_dir=tmp_dir, mode="validation"
            )

            print("Train set size: ", len(train_set))
            print("Validation set size: ", len(val_set))

            # net = torch.load(results_dir + '/UNet3D.pth')
            net = unet.UNet(1, 1).to(dev)
            net = net.to(dev)

            train_class = train.Train(dev, 20, 6, 1e-3, net)
            trained_model = train_class.train(train_set, val_set, net)
            torch.save(trained_model, results_dir + "/UNet.pth")
    else:
        with TemporaryDirectory(dir=results_dir) as tmp_dir:
            trained_model = torch.load(results_dir + "/UNet.pth", map_location=dev)

            trained_model.eval()

            test_set = dataset.GetDataset(
                dataset_dir + "/test", tmp_dir=tmp_dir, mode="test"
            )
            print("Test set size: ", len(test_set))

            test_class = test.Test(dev, 1, test_set, trained_model, results_dir)
            test_class.test()
