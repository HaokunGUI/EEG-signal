from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping
import torch
import torch.nn as nn
from torch import optim
import os
import warnings
import numpy as np
import argparse
from tqdm import tqdm
import json
from utils.utils import *
from utils.graph import *
from utils.loss import *
from utils.tools import *
from utils.visualize import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.constants import *

warnings.filterwarnings('ignore')

class Exp_SSL(Exp_Basic):
    def __init__(self, args:argparse.Namespace):
        super(Exp_SSL, self).__init__(args)
        self.scalar = self._get_scalar()

    def _build_model(self):
        # model init
        model = self.model_dict[self.args.model].Model(self.args, self.device)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids).cuda()
        return model
    
    def _get_scalar(self):
        if self.args.normalize:
            means_dir = os.path.join(
                self.args.marker_dir, f'file_markers_{self.args.task_name}', 
                'means_seq2seq_fft_'+str(self.args.input_len)+'s_single.pkl'
                )
            stds_dir = os.path.join(
                self.args.marker_dir, f'file_markers_{self.args.task_name}',
                'stds_seq2seq_fft_'+str(self.args.input_len)+'s_single.pkl'
                )
            with open(means_dir, 'rb') as f:
                means = pickle.load(f)
            with open(stds_dir, 'rb') as f:
                stds = pickle.load(f)
            scalar = StandardScaler(mean=means, std=stds, device=self.device)
        else:
            scalar = None
        return scalar

    def _get_data(self, flag, scalar):
        self.args.split = flag
        data_set, data_loader = data_provider(self.args, scalar=scalar)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self):
        criterion = loss_fn(self.scalar, self.args.loss_fn, is_tensor=True, mask_val=0.)
        return criterion
    
    def _select_scheduler(self, optimizer):
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.num_epochs)
        return scheduler

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (x, y_true, _, supports) in tqdm(enumerate(vali_loader)):
                x = x.float().to(self.device)
                y_true = y_true.float().to(self.device)
                if self.args.use_graph:
                    for i in range(len(supports)):
                        supports[i] = supports[i].to(self.device)

                y_pred = self.model(x, y_true, supports, None)

                y_pred = y_pred.detach().cpu()
                loss = criterion(y_true.cpu(), y_pred)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.logging.add_scalar('vali/loss', total_loss, self.steps)

        self.model.train()
        return total_loss

    def train(self):
        _, train_loader = self._get_data(flag='train', scalar=self.scalar)
        _, vali_loader = self._get_data(flag='dev', scalar=self.scalar)

        path = os.path.join(self.args.log_dir, self.args.model, self.args.task_name)
        if not os.path.exists(path):
            os.makedirs(path)
        checkpoint_dir = os.path.join(path, 'checkpoint')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        args_file = os.path.join(self.args.log_dir, self.args.model, self.args.task_name, 'args.json')
        with open(args_file, 'w') as f:
            json.dump(vars(self.args), f, indent=4, sort_keys=True)

        if self.args.use_graph and not os.path.exists(os.path.join(self.args.log_dir, self.args.model, self.args.task_name, 'graph')):
            os.makedirs(os.path.join(self.args.log_dir, self.args.model, self.args.task_name, 'graph'))

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = self._select_scheduler(model_optim)

        for epoch in range(self.args.num_epochs):
            start_draw = True
            self.steps = 0
            self.model.train()

            saver = CheckpointSaver(checkpoint_dir,
                                  metric_name=self.args.loss_fn,
                                  maximize_metric=False)

            with torch.enable_grad() and tqdm(train_loader.dataset, desc=f'Epoch: {epoch + 1} / {self.args.num_epochs}') as progress_bar:
                for i, (x, y_true, adj_mat, supports) in enumerate(train_loader):
                    model_optim.zero_grad()

                    x = x.float().to(self.device)
                    batch_size = x.size(0)
                    y_true = y_true.to(self.device)
                    if self.args.use_graph:
                        for i in range(len(supports)):
                            supports[i] = supports[i].to(self.device)

                    if self.args.use_graph and start_draw:
                        if self.args.graph_type == 'distance' and self.args.adj_every > 0:
                            pos_spec = get_spectral_graph_positions(self.args.marker_dir)
                            fig = draw_graph_weighted_edge(adj_mat, NODE_ID_DICT, pos_spec, title=f'distance_epoch{epoch}.png', 
                                                     is_directed=False, plot_colorbar=True, font_size=30,
                                                     save_dir=os.path.join(self.args.log_dir, self.args.model, self.args.task_name, 'graph'))
                            self.args.adj_every = 0
                            self.logging.add_figure('graph/distance', fig, epoch)
                            start_draw = False
                        elif self.args.graph_type == 'correlation' and (epoch + 1) % self.args.adj_every == 0:
                            pos_spec = get_spectral_graph_positions(self.args.marker_dir)
                            fig = draw_graph_weighted_edge(adj_mat, NODE_ID_DICT, pos_spec, title=f'correlation_epoch{epoch}.png', 
                                                     is_directed=self.args.directed, plot_colorbar=True, font_size=30, 
                                                     save_dir=os.path.join(self.args.log_dir, self.args.model, self.args.task_name, 'graph'))
                            self.logging.add_figure(f'graph/correlation_{epoch}', fig, epoch)
                            start_draw = False

                    seq_pred = self.model(x, y_true, supports, self.steps)
                    loss = criterion(y_true, seq_pred).to(self.device)

                    loss_val = loss.item()
                    loss.backward()

                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_norm)
                    self.steps += batch_size
                    model_optim.step()
                    
                    progress_bar.update(batch_size)
                    progress_bar.set_postfix(loss=loss_val, lr=model_optim.param_groups[0]['lr'])

                    self.logging.add_scalar('train/loss', loss_val, self.steps)
                    self.logging.add_scalar('train/lr', model_optim.param_groups[0]['lr'], self.steps)

                    saver.save(epoch, self.model, model_optim, loss_val)

                    if (i+1) % self.args.eval_every == 0:
                        vali_loss = self.vali(vali_loader, criterion)
                        early_stopping(vali_loss, self.model, path)
                        if early_stopping.early_stop:
                            print("Early stopping")
                            break
                scheduler.step()

        return

    def test(self, model_file:str='best.pth.tar'):
        _, test_loader = self._get_data(flag='eval', scalar=self.scalar)
        print('loading model')
        path = os.path.join(self.args.log_dir, self.args.model, self.args.task_name, model_file)
        load_model_checkpoint(path, self.model)

        criterion = self._select_criterion()

        self.model.eval()
        loss = []
        with torch.no_grad():
            for i, (x, y_true, _, supports) in tqdm(enumerate(test_loader)):
                x = x.float().to(self.device)
                y_true = y_true.float().to(self.device)
                if self.args.use_graph:
                    for i in range(len(supports)):
                        supports[i] = supports[i].to(self.device)

                y_pred = self.model(x, y_true, supports, None)

                loss = criterion(y_true, y_pred).to(self.device)
                loss_val = loss.item()
                loss.append(loss_val)

            loss = np.average(loss)
        self.logging.add_scalar('test/loss', loss, self.steps)
        return