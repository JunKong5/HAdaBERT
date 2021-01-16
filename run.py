import cfgs.config as config
from common.trainer_decoder import DTrainer
from common.adaptive_AGMbert_encoder import adaSTrainer
import argparse, yaml
import random
from easydict import EasyDict as edict
from common.utils import load_pretrained_segment_adaembeddings


def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='HAdaBERT Args')

    parser.add_argument('--run', dest='run_mode',
                        choices=['train', 'val', 'test',''],
                        help='{train, val, test}',
                        type=str, required=True)

    parser.add_argument('--model', dest='model',
                        choices=['e1',''],
                        help='{e1, ...}',
                        default='', type=str)

    parser.add_argument('--mode', dest='mode',
                        choices=['gencoder', 'save_embedding', 'adabert_encoder'],
                        help='{adabert_encoder, ...}',
                        default='encoder', type=str)

    parser.add_argument('--gencoder', dest='gencoder',
                        choices=['e1', 'e2',],
                        help='{train, val, test}',
                        default='e1', type=str)

    parser.add_argument('--dataset', dest='dataset',
                        choices=['yelp_13', 'IMDB',"aapd",'reuters'],
                        help='{IMDB yelp_13}',
                        default='imdb', type=str)

    parser.add_argument('--gpu', dest='gpu',
                        help="gpu select, eg.'0, 1, 2'",
                        type=str,
                        default="0,1")

    parser.add_argument('--seed', dest='seed',
                        help='fix random seed',
                        type=int,
                        default=random.randint(0, 99999999))

    parser.add_argument('--version', dest='version',
                        help='version control',
                        type=str,
                        default="default")


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    __C = config.__C

    args = parse_args()

    args_dict = edict({**vars(args)})
    config.add_edit(args_dict, __C)
    config.proc(__C)


    print('Hyper Parameters:')
    config.config_print(__C)


    if __C.mode == 'adabert_encoder':
        execution = adaSTrainer(__C)
        execution.run(__C.run_mode)
    if __C.mode == 'gencoder':
        execution = DTrainer(__C)
        execution.run(__C.run_mode)
    if __C.mode == 'save_embedding':
        load_pretrained_segment_adaembeddings(__C, load_pretrained=False)



