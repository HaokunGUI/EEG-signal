import torch
import os
import shutil
import queue
from utils.constants import INCLUDED_CHANNELS
import json

class CheckpointSaver:
    """Class to save and load model checkpoints.
    Save the best checkpoints as measured by a metric value passed into the
    `save` method. Overwrite checkpoints with better checkpoints once
    `max_checkpoints` have been saved.
    Args:
        save_dir (str): Directory to save checkpoints.
        metric_name (str): Name of metric used to determine best model.
        maximize_metric (bool): If true, best checkpoint is that which maximizes
            the metric value passed in via `save`. Otherwise, best checkpoint
            minimizes the metric.
        log (logging.Logger): Optional logger for printing information.
    """

    def __init__(self, save_dir, metric_name, maximize_metric=False, fn=lambda : int(os.environ["LOCAL_RANK"])==0):
        super(CheckpointSaver, self).__init__()

        self.save_dir = save_dir
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.fn = fn

    def is_best(self, metric_val):
        """Check whether `metric_val` is the best seen so far.
        Args:
            metric_val (float): Metric value to compare to prior checkpoints.
        """
        if metric_val is None:
            # No metric reported
            return False

        if self.best_val is None:
            # No checkpoint saved yet
            return True

        return ((self.maximize_metric and self.best_val <= metric_val)
                or (not self.maximize_metric and self.best_val >= metric_val))

    def _print(self, message):
        """Print a message if logging is enabled."""
        pass

    def save(self, epoch, model, optimizer, metric_val, param_dict=None):
        """Save model parameters to disk.
        Args:
            epoch (int): Current epoch.
            model (torch.nn.DataParallel): Model to save.
            optimizer: optimizer
            metric_val (float): Determines whether checkpoint is best so far.
        """
        if not self.fn():
            return
        ckpt_dict = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }

        checkpoint_path = os.path.join(self.save_dir, 'last.pth.tar')
        torch.save(ckpt_dict, checkpoint_path)

        best_path = ''
        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, 'best.pth.tar')
            shutil.copy(checkpoint_path, best_path)
            if param_dict is not None:
                with open(os.path.join(self.save_dir, 'params.json'), 'w') as f:
                    json.dump(param_dict, f, indent=4, sort_keys=True)



def load_model_checkpoint(checkpoint_file, model, optimizer=None, map_location=None):
    if map_location is not None:
        loc = f'cuda:{map_location}'
        checkpoint = torch.load(checkpoint_file, map_location=loc)
    else:
        checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state'], strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        return model, optimizer
    return model

def last_relevant_pytorch(output, lengths, batch_first=True):
    # masks of the true seq lengths
    masks = (lengths - 1).view(-1, 1).expand(len(lengths), output.size(2)).cuda()
    time_dimension = 1 if batch_first else 0
    masks = masks.unsqueeze(time_dimension)
    last_output = output.gather(time_dimension, masks).squeeze(time_dimension).cuda()

    return last_output

def get_swap_pairs(channels):
    """
    Swap select adjacenet channels
    Args:
        channels: list of channel names
    Returns:
        list of tuples, each a pair of channel indices being swapped
    """
    swap_pairs = []
    if ("EEG FP1" in channels) and ("EEG FP2" in channels):
        swap_pairs.append([channels.index("EEG FP1"), channels.index("EEG FP2")])
    if ("EEG Fp1" in channels) and ("EEG Fp2" in channels):
        swap_pairs.append([channels.index("EEG Fp1"), channels.index("EEG Fp2")])
    if ("EEG F3" in channels) and ("EEG F4" in channels):
        swap_pairs.append([channels.index("EEG F3"), channels.index("EEG F4")])
    if ("EEG F7" in channels) and ("EEG F8" in channels):
        swap_pairs.append([channels.index("EEG F7"), channels.index("EEG F8")])
    if ("EEG C3" in channels) and ("EEG C4" in channels):
        swap_pairs.append([channels.index("EEG C3"), channels.index("EEG C4")])
    if ("EEG T3" in channels) and ("EEG T4" in channels):
        swap_pairs.append([channels.index("EEG T3"), channels.index("EEG T4")])
    if ("EEG T5" in channels) and ("EEG T6" in channels):
        swap_pairs.append([channels.index("EEG T5"), channels.index("EEG T6")])
    if ("EEG O1" in channels) and ("EEG O2" in channels):
        swap_pairs.append([channels.index("EEG O1"), channels.index("EEG O2")])

    return swap_pairs

def getOriginalData(x:torch.Tensor, isAug:torch.Tensor):
    x_new = x.clone().cuda()
    change_channels = torch.Tensor(get_swap_pairs(INCLUDED_CHANNELS)).int()
    channel_1 = change_channels[:, 0]
    channel_2 = change_channels[:, 1]
    x_new[:, channel_1, :] = x[:, channel_2, :]
    x_new[:, channel_2, :] = x[:, channel_1, :]
    isAug = isAug.clone().reshape(-1, 1, 1).cuda()
    return x_new * isAug + x * (1 - isAug)

def random_masking(xb:torch.Tensor, mask_ratio):
    # x:[batch_size, nvar, patch_num, embedding_dim]
    bs, nvars, L, D= xb.shape
    x = xb.clone()
    
    len_keep = int(L * (1 - mask_ratio))
        
    noise = torch.rand(bs, nvars, L, device=xb.device)  # noise in [0, 1], bs x nvars x L
        
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=2)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=2)                                     # ids_restore: [bs x nvars x L]

    # keep the first subset
    ids_keep = ids_shuffle[:, :, :len_keep]                                              # ids_keep: [bs x nvars x len_keep]         
    x_kept = torch.gather(x, dim=2, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))     # x_kept: [bs x nvars x len_keep x patch_len]
   
    # removed x
    x_removed = torch.zeros(bs, nvars, L-len_keep, D, device=xb.device)                 # x_removed: [bs x nvars x (L-len_keep) x patch_len]
    x_ = torch.cat([x_kept, x_removed], dim=2)                                          # x_: [bs x nvars x L x patch_len]

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=2, index=ids_restore.unsqueeze(-1).repeat(1,1,1,D)) # x_masked: [bs x nvars x num_patch x patch_len]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, nvars, L], device=x.device)                                  # mask: [bs x nvars x num_patch]
    mask[:, :, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=2, index=ids_restore)                                  # [bs x nvars x num_patch]
    return x_masked, x_kept, mask, ids_restore