import argparse
from torch.utils.data import Dataset
import warnings
import os
from typing import Any, Tuple
import h5py
from scipy.fftpack import fft
from utils.constants import INCLUDED_CHANNELS
import numpy as np
import torch
from utils.graph import get_supports

warnings.filterwarnings('ignore')


class Dataset_TUSZ(Dataset):
    def __init__(self, args:argparse.Namespace, scalar):
        super(Dataset_TUSZ, self).__init__()
        self.task_name = args.task_name
        self.root_path = args.root_path
        self.marker_dir = args.marker_dir
        self.input_len = args.input_len
        self.output_len = args.output_len
        self.use_fft = args.use_fft
        self.split = args.split
        self.time_step_len = args.time_step_len
        self.data_augment = args.data_augment
        self.use_graph = args.use_graph
        self.preproc_dir = args.preproc_dir
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
            pass

        elif self.task_name == 'classification':
            pass

        else:
            raise NotImplementedError


    def __getitem__(self, index: int) -> Any:
        if self.task_name == 'ssl':
            file_name_tuple = self.file_tuples[index]

            if self.preproc_dir is None:
                x, y = self._getIdx2Slice(file_name_tuple)
            else:
                file_name = f'datasets_{self.split}_{index}.h5'
                with h5py.File(os.path.join(self.preproc_dir, file_name), 'r') as hf:
                    x = hf['x'][()]
                    y = hf['y'][()]

            if self.data_augment:
                # TODO
                pass
            if self.scalar is not None:
                x = self.scalar.transform(x)
                y = self.scalar.transform(y)
            # convert to Tensor
            x = torch.Tensor(x)
            y = torch.Tensor(y)

            if self.use_graph:
                adj_mat, supports = get_supports(self.args, x)
            else:
                adj_mat, supports = None, None
        
        elif self.task_name == 'anomaly_detection':
            pass

        elif self.task_name == 'classification':
            pass

        else:
            raise NotImplementedError
        
        return (x, y, adj_mat, supports)
    

    def __len__(self) -> int: 
        return self.size

    def _getIdx2Slice(self, file_name_tuple: Tuple[str]):
        assert self.input_len % self.time_step_len == 0 and self.output_len % self.time_step_len == 0, \
            'input_len and output_len should be divisible by time_step_len'
        
        file_name_i, file_name_o = file_name_tuple

        slice_num_i = int(file_name_i.split('_')[-1].split('.h5')[0])
        slice_num_o = int(file_name_o.split('_')[-1].split('.h5')[0])
        assert slice_num_o == slice_num_i + 1, 'slice_num_o should be equal to slice_num_i + 1'

        file_name_i = file_name_i.split('.edf')[0] + '.h5'
        file_name_o = file_name_o.split('.edf')[0] + '.h5'
        assert file_name_i == file_name_o, 'file_name_i should be equal to file_name_o'

        file_path = os.path.join(self.root_path, file_name_i)
        with h5py.File(file_path, 'r') as f:
            signals = f["resampled_signal"][()]
            freq = f["resample_freq"][()]
        input_node_num = int(freq * self.input_len)
        output_node_num = int(freq * self.output_len)
        time_step_num = int(freq * self.time_step_len)

        start_window = input_node_num * slice_num_i
        end_window = start_window + input_node_num + output_node_num
        # (num_channel, input_len)
        input_slice = signals[:, start_window:end_window]
        # change to (time_step_num, num_channel, time_step_len)
        start_time_step = 0
        time_steps = []
        while start_time_step + time_step_num <= input_node_num + output_node_num:
            slice = input_slice[:, start_time_step:start_time_step+time_step_num]
            if self.use_fft:
                fft_slice = fft(slice, n=time_step_num, axis=-1)
                idx_pos = int(np.floor(time_step_num / 2))
                fourier_signal = fft_slice[:, :idx_pos]
                amp = np.abs(fourier_signal)
                amp[amp == 0.0] = 1e-8  # avoid log of 0
                time_steps.append(np.log(amp))
            start_time_step += time_step_num
        time_steps = np.stack(time_steps, axis=0)
        return time_steps[:(self.input_len // self.time_step_len), :, :], time_steps[(self.input_len // self.time_step_len):, :, :]
    
    
        
        
