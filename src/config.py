from os.path import join, realpath, dirname
from src import tool
# argparse config
prog_name = 'Robust'
prog_description = 'Robust. '
cmd_list = ['temp', 'train', 'attack', 'test', 'v', 'v2']
# logging config
log_name = 'robust'
globals().update(vars(tool.gen_parser()))


def init_path_config(mainfile):
    gv = globals()
    gv['prj_dir'] = prj_dir = dirname(realpath(mainfile))
    gv['data_dir'] = data_dir = join(prj_dir, 'data')
    gv['log_dir'] = join(data_dir, 'log')
    gv['loss_dir'] = join(data_dir, 'loss')
    gv['model_dir'] = join(data_dir, 'model')
    gv['best_model_dir'] = join(data_dir, 'best_model')
    gv['tbx_dir'] = join(data_dir, 'tbx')
    gv['v_dir'] = join(data_dir, 'v')
    gv['mnist_dir'] = join(data_dir, 'mnist')
    gv['cifar10_dir'] = join(data_dir, 'cifar10')
    gv['voc10_dir'] = join(data_dir, 'voc10')
    # gv['miniimg_dir'] = join(data_dir, 'miniimg')
    gv['skin7_dir'] = join(data_dir, 'skin7')
    gv['skin4_dir'] = join(data_dir, 'skin4')
    gv['xray3_dir'] = join(data_dir, 'xray3')
    gv['imagenet100_dir'] = join(data_dir, 'imagenet100')
    gv['adversarial_dir'] = join(data_dir, 'adversarial')
    gv['noise_pic_dir'] = join(data_dir, 'noise_pic')
