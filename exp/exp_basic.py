import os
import torch
from models import TimesNet, DCRNN
from tensorboardX import SummaryWriter


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'DCRNN': DCRNN,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        logging_dir = os.path.join(self.args.log_dir, self.args.model, self.args.task_name, 'log')
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)
        self.logging = SummaryWriter(logging_dir)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self, vali_loader, criterion):
        pass

    def train(self):
        pass

    def test(self, model_file:str='best.pth.tar'):
        pass
