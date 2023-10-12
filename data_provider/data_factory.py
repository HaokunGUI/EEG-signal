from data_provider.data_loader import Dataset_TUSZ
from torch.utils.data import DataLoader
import argparse

data_dict = {
    'TUSZ': Dataset_TUSZ,
}

def data_provider(args: argparse.Namespace, scalar=None):
    if args.split == 'train':
        shuffle = True
        batch_size = args.train_batch_size
    else:
        shuffle = False
        batch_size = args.test_batch_size
        args.data_augment = False

    dataset = data_dict[args.dataset](args, scalar=scalar)
    return dataset, \
        DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
        )
    
