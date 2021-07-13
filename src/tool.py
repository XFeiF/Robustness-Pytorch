import argparse
import logging
import os
import sys
from os.path import join, exists
from datetime import datetime
import time
import functools
from src import config


def gen_parser():
    parser = argparse.ArgumentParser(prog=config.prog_name, description=config.prog_description)
    parser.add_argument('cmd', choices=config.cmd_list, help="what to do")
    parser.add_argument('--cuda', type=str, default='0', help='which gpu')
    parser.add_argument('--n_dl', type=int, default=16, help='num_worker for dataloader')
    parser.add_argument('--dsid', type=str, default='skin4', help='which data set')
    parser.add_argument('--mid', type=str, default='res50', help='which net')

    parser.add_argument('--acid', type=str, default='g', help='which action')
    parser.add_argument('--epoch', type=int, default=90, help='epoch of train')
    parser.add_argument('--pretrain', action='store_true', default=False, help='pre train')
    parser.add_argument('--batch_train', type=int, default=32, help='train & test batch size')
    parser.add_argument('--no_eval', action='store_true', default=False, help='no need to eval')

    parser.add_argument('--kk', type=str, default='3,0,0.2,1,10', help='k1,k2,...')

    parser.add_argument('--midtf', type=str, default='imagenet100_res18_base2', help='which net')
    parser.add_argument('--atkid', type=str, default='CW', help='which attacker')
    parser.add_argument('--attack_on_train', action='store_true', default=False, help='attack with train set')
    parser.add_argument('--batch_attack', type=int, default=128, help='attack batch size')
    parser.add_argument('--nsize', type=float, default=0.1, help='noise size')
    parser.add_argument('--reattack', action='store_true', default=False, help='restart the attack')

    parser.add_argument('--testidtf', type=str, default='cifar10_res18_base_eval_FGSM_epsilon:0.01', help='test dataset dir')
    parser.add_argument('--logidtf', type=str, default='', help='idef in log dir')
    return parser.parse_args()


def gen_t_name(base_dir, ext):
    if not exists(base_dir):
        os.makedirs(base_dir)
    while True:
        dt = datetime.now()
        temp = join(base_dir, dt.strftime('%Y%m%d_%H%M%S_%f') + ext)
        if exists(temp):
            time.sleep(0.000001)
        else:
            break
    return temp


def gen_logger(fpath, log_name=None):
    if log_name is None:
        log_name = config.log_name
    formatter = logging.Formatter(
        '%(asctime)s@%(name)s %(levelname)s # %(message)s')
    file_handler = logging.FileHandler(fpath)
    file_handler.formatter = formatter
    console_handler = logging.StreamHandler()
    console_handler.formatter = formatter
    logger = logging.getLogger(log_name)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)
    return logger


def log_decorator(call_fn):
    @functools.wraps(call_fn)
    def log(self, *args, move=False):
        if self.current < self.total:
            sys.stdout.write(' ' * (self.width + 26) + '\r')
            sys.stdout.flush()
        call_fn(self, *args)
        if move:
            self._current += 1
            temp = datetime.now()
            delta = temp - self.last_time
            self.last_time = temp
            temp = temp + delta * (self.total - self.current)
            self.ok_time = str(temp).split('.')[0]
        if self.current < self.total:
            progress = int(self.width * self.current / self.total)
            temp = '{:2}%][{}]\r'.format(int(100 * self.current / self.total), self.ok_time)
            sys.stdout.write('[' + '=' * progress + '>' + '-' * (self.width - progress - 1) + temp)
            sys.stdout.flush()
    return log


class ProgressBarLog:
    def __init__(self, total=50, width=76, current=0, logger=None):
        self.width = width - 26
        self.total = total
        self._current = current
        if logger is None:
            if config.logidtf:
                self.logger = gen_logger(gen_t_name(join(config.log_dir, '{}-{}'.format(config.cmd, config.logidtf)),
                                                    '.log'))
            else:
                self.logger = gen_logger(gen_t_name(join(config.log_dir, config.cmd), '.log'))
        else:
            self.logger = logger
        self.last_time = datetime.now()
        self.ok_time = None

    @property
    def current(self):
        return self._current

    @current.setter
    def current(self, value):
        if not isinstance(value, int):
            raise ValueError
        if value < 0 or value >= self.total:
            raise ValueError
        self._current = value

    @log_decorator
    def debug(self, msg):
        self.logger.debug(msg)

    @log_decorator
    def info(self, msg):
        self.logger.info(msg)

    @log_decorator
    def warning(self, msg):
        self.logger.warning(msg)

    @log_decorator
    def error(self, msg):
        self.logger.error(msg)

    @log_decorator
    def exception(self, msg):
        self.logger.exception(msg)

    @log_decorator
    def print(self, *args):
        print(*args)

    @log_decorator
    def refresh(self):
        pass


pblog = None


def get_pblog(*args, **kwargs):
    global pblog
    if pblog is None:
        pblog = ProgressBarLog(*args, **kwargs)
    return pblog


'''
def save_noise_pic(ds):
    ds = dataset.get_dataloader(ds)
    pcf = config.get_pcf()
    args = config.get_args()
    save_dir = os.path.join(pcf.noise_pic_dir, args.dataset)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    for ii in range(ds.num_classes):
        os.makedirs(os.path.join(save_dir, ds.train.dataset.classes[ii]))
    saveid = 0
    topil = transforms.ToPILImage()
    for x, y in ds.train:
        noise_x = torch.empty_like(x)
        for ii, (jj, kk) in enumerate(zip(ds.mean, ds.std)):
            noise_x[:, ii, :, :].uniform_(-args.atth / (100 * kk), kk * args.atth / (100 * kk))
            noise_x[:, ii, :, :] = np.clip(x[:, ii, :, :] + noise_x[:, ii, :, :], -jj / kk, (1 - jj) / kk)
        for ii, (jj, kk) in enumerate(zip(ds.mean, ds.std)):
            x[:, ii, :, :] *= kk
            x[:, ii, :, :] += jj
            noise_x[:, ii, :, :] *= kk
            noise_x[:, ii, :, :] += jj
        for ii in range(y.size(0)):
            path = os.path.join(save_dir, ds.train.dataset.classes[y[ii].item()], '{}o.bmp'.format(saveid))
            topil(x[ii]).save(path)
            path = os.path.join(save_dir, ds.train.dataset.classes[y[ii].item()], '{}n.bmp'.format(saveid))
            topil(noise_x[ii]).save(path)
            saveid += 1
'''

if __name__ == '__main__':
    pass
