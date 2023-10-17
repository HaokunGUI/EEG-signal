import argparse
from torch.utils.data import Dataset
import warnings
import os
from typing import Any, Tuple
import h5py
import torch
import numpy as np

warnings.filterwarnings('ignore')


class Dataset_TUSZ(Dataset):
    def __init__(self, args:argparse.Namespace, scalar):
        super(Dataset_TUSZ, self).__init__()
        self.task_name = args.task_name
        self.root_path = args.root_path
        self.marker_dir = args.marker_dir
        self.input_len = args.input_len
        self.output_len = args.output_len
        self.split = args.split
        self.data_augment = args.data_augment
        self.use_graph = args.use_graph
        self.args = args
        self.scalar = scalar

        marker_dir = f'file_markers_{self.task_name}'
        self.marker_dir = os.path.join(self.marker_dir, marker_dir)
        if self.task_name == 'ssl':
            file_name = f'{self.split}Set_seq2seq_{self.input_len}s.txt'
            file_path = os.path.join(self.marker_dir, file_name)
            with open(file_path, 'r') as f:
                f_str = f.readlines()
                self.file_tuples = [f_str[i].strip().split(',') for i in range(len(f_str))]
                self.size = len(self.file_tuples)

        elif self.task_name == 'anomaly_detection':
            nosz_file_name = f'{self.split}Set_seq2seq_{self.input_len}s_nosz.txt'
            sz_file_name = f'{self.split}Set_seq2seq_{self.input_len}s_sz.txt'
            with open(os.path.join(self.marker_dir, nosz_file_name), 'r') as f_nosz:
                with open(os.path.join(self.marker_dir, sz_file_name), 'r') as f_sz:
                    f_nosz_str = f_nosz.readlines()
                    f_sz_str = f_sz.readlines()
            if self.split == 'train':
                num_points = int(self.args.scale_ratio * len(f_sz_str))
                np.random.shuffle(f_nosz_str)
                f_nosz_str = f_nosz_str[:num_points]
                np.random.shuffle(f_sz_str)
                f_sz_str = f_sz_str[:num_points]
                f_combine_str = f_nosz_str + f_sz_str
                np.random.shuffle(f_combine_str)
                for i in range(len(f_combine_str)):
                    f_combine_str[i] = f_combine_str[i].strip('\n').split(',')


            file_path = os.path.join(self.marker_dir, file_name)

        elif self.task_name == 'classification':
            pass

        else:
            raise NotImplementedError


    def __getitem__(self, index: int) -> Any:
        if self.task_name == 'ssl':
            file_name_tuple = self.file_tuples[index]

            x, y = self._getIdx2Slice(file_name_tuple)

            if self.data_augment:
                # TODO
                pass
            if self.scalar is not None:
                x = self.scalar.transform(x)
                y = self.scalar.transform(y)
            # convert to Tensor
            x = torch.Tensor(x)
            y = torch.Tensor(y)
        
        elif self.task_name == 'anomaly_detection':
            pass

        elif self.task_name == 'classification':
            pass

        else:
            raise NotImplementedError
        
        return x, y
    

    def __len__(self) -> int: 
        return self.size

    def _getIdx2Slice(self, file_name_tuple: Tuple[str]):
        file_name_i, file_name_o = file_name_tuple

        slice_num_i = int(file_name_i.split('_')[-1].split('.h5')[0])
        slice_num_o = int(file_name_o.split('_')[-1].split('.h5')[0])
        assert slice_num_o == slice_num_i + 1, 'slice_num_o should be equal to slice_num_i + 1'

        file_name_i = file_name_i.split('.edf')[0] + '.h5'
        file_name_o = file_name_o.split('.edf')[0] + '.h5'
        assert file_name_i == file_name_o, 'file_name_i should be equal to file_name_o'

        file_path = os.path.join(self.root_path, file_name_i)
        with h5py.File(file_path, 'r') as f:
            signals = f["resample_signal"][()]
            freq = f["resample_freq"][()]
        input_node_num = int(freq * self.input_len)
        output_node_num = int(freq * self.output_len)

        start_window = input_node_num * slice_num_i
        end_window = start_window + input_node_num + output_node_num
        # (num_channel, input_len)
        input_slice = signals[:, start_window:end_window]
        return input_slice[:, :input_node_num], input_slice[:, input_node_num:]
