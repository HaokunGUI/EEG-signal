import argparse
import torch
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from exp.exp_ssl import Exp_SSL
import torch.multiprocessing
from utils.tools import ddp_setup, ddp_cleanup
import os

def main(args: argparse.Namespace):
    if args.use_gpu:
        ddp_setup()
        rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0

    if args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    elif args.task_name == 'ssl':
        Exp = Exp_SSL
    else:
        raise ValueError('task name must be in [anomaly_detection, classification, ssl]')

    exp = Exp(args)
    if rank == 0:
        print('>>>>>>> training : <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train()
    if rank == 0:
        print('>>>>>>> testing : <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.test()
    torch.cuda.empty_cache()
    
    if args.use_gpu:
        ddp_cleanup()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='ssl',
                        help='task name, options:[classification, anomaly_detection, ssl]')
    parser.add_argument('--model', type=str, required=True, default='DCRNN',
                        help='model name, options: [DCRNN, TimesNet]')
    parser.add_argument('--log_dir', type=str, default='/home/guihaokun/Time-Series-Pretrain/logging', help='log dir')

    # data loader
    parser.add_argument('--dataset', type=str, default='TUSZ', help='dataset type, options:[TUSZ]')
    parser.add_argument('--root_path', type=str, default='/data/guihaokun/resample/tuh_eeg_serizure/', help='root path of the data file')
    parser.add_argument('--marker_dir', type=str, default='/home/guihaokun/Time-Series-Pretrain/data', help='marker dir')
    parser.add_argument('--data_augment', action='store_true', help='use data augment or not', default=False)
    parser.add_argument('--normalize', action='store_true', help='normalize data or not', default=False)
    parser.add_argument('--train_batch_size', type=int, default=256, help='batch size of train input data')
    parser.add_argument('--test_batch_size', type=int, default=256, help='batch size of test input data')
    parser.add_argument('--num_workers', type=int, default=16, help='data loader num workers')
    parser.add_argument('--freq', type=int, default=250, help='sample frequency')

    # ssl task
    parser.add_argument('--input_len', type=int, default=60, help='input sequence length')
    parser.add_argument('--output_len', type=int, default=12, help='prediction sequence length')
    parser.add_argument('--time_step_len', type=int, default=1, help='time step length')
    parser.add_argument('--use_fft', action='store_true', help='use fft or not', default=False)
    parser.add_argument('--loss_fn', type=str, default='mse', help='loss function, options:[mse, mae]')

    # graph setting
    parser.add_argument('--graph_type', type=str, default='correlation', help='graph type, option:[distance, correlation]')
    parser.add_argument('--top_k', type=int, default=3, help='top k')
    parser.add_argument('--directed', action='store_true', help='directed graph or not', default=False)
    parser.add_argument('--filter_type', type=str, default='dual_random_walk', help='filter type')

    # model define
    parser.add_argument('--num_nodes',type=int, default=19, help='Number of nodes in graph.')
    parser.add_argument('--num_rnn_layers', type=int, default=2, help='Number of RNN layers in encoder and/or decoder.')
    parser.add_argument('--rnn_units', type=int, default=64, help='Number of hidden units in DCRNN.')
    parser.add_argument('--dcgru_activation', type=str, choices=('relu', 'tanh'), default='tanh', help='Nonlinear activation used in DCGRU cells.')
    parser.add_argument('--input_dim', type=int, default=None, help='Input seq feature dim.')
    parser.add_argument('--output_dim', type=int, default=None, help='Output seq feature dim.')
    parser.add_argument('--max_diffusion_step', type=int, default=2, help='Maximum diffusion step.')
    parser.add_argument('--cl_decay_steps', type=int, default=3000, help='Scheduled sampling decay steps.')
    parser.add_argument('--use_curriculum_learning', default=False, action='store_true', help='Whether to use curriculum training for seq-seq model.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability.')

    # optimization
    parser.add_argument('--num_epochs', type=int, default=40, help='train epochs')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay')
    parser.add_argument('--max_norm', type=float, default=4.0, help='max norm of grad')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # log setting
    parser.add_argument('--eval_every', type=int, default=5, help='evaluate every X epochs')
    parser.add_argument('--adj_every', type=int, default=10, help='display adj matrix every X epochs')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    if args.graph_type in ['distance', 'correlation']:
        args.use_graph = True
    else:
        args.use_graph = False

    if args.use_fft:
        args.input_dim = args.freq // 2
        args.output_dim = args.freq // 2
    else:
        args.input_dim = args.freq
        args.output_dim = args.freq
    
    main(args)