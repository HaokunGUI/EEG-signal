import os
import torch
from models import TimesNet, DCRNN
from tensorboardX import SummaryWriter
from utils.tools import WriterFilter


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'DCRNN': DCRNN,
        }
        self.device = self._acquire_device()
        self.model = self._build_model()
        self.scalar = self._get_scalar()
        self.world_size = int(os.environ["WORLD_SIZE"])
        logging_dir = os.path.join(self.args.log_dir, self.args.model, self.args.task_name, 'log')
        os.makedirs(logging_dir, exist_ok=True)
        self.logging = WriterFilter(SummaryWriter(logging_dir))
        self.criterion = self._select_criterion()

    def _select_criterion(self):
        raise NotImplementedError

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices
            device = int(os.environ["LOCAL_RANK"])
        else:
            device = torch.device('cpu')
        return device
    
    def _get_scalar(self):
        raise NotImplementedError

    def _get_data(self):
        pass

    def vali(self, vali_loader, criterion):
        pass

    def train(self):
        pass

    def test(self, model_file:str='best.pth.tar'):
        pass
