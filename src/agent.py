import os
from os.path import join, exists
import shutil
import numpy as np
from PIL import Image
import torch
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from src import tool
from src import dataset as DataSet
from src import net as Model
from src import action as Action
from src import attack as Attack


class Trainer:
    def __init__(self, args):
        os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda']
        self.model_dir = args['model_dir']
        self.best_model_dir = args['best_model_dir']
        self.epoch = args['epoch']
        self.no_eval = args['no_eval']
        self.dl = DataSet.get_dataloader(args)
        args['mean'] = self.dl.mean
        args['std'] = self.dl.std
        args['num_classes'] = self.dl.num_classes
        self.ac = Action.get_action(args)
        self.m = Model.get_net(args)
        self.midtf = '{}_{}_{}{}'.format(args['dsid'], args['mid'], args['acid'], self.ac.strkk)
        self.mpkl = self.midtf + '.model'
        if args['pretrain']:
            p = join(self.best_model_dir, 'Best_'+self.mpkl)
            self.m.load_state_dict(torch.load(p, map_location='cpu'))
        self.m.cuda()
        if torch.cuda.device_count() > 1:
            self.m = torch.nn.DataParallel(self.m)
            self.ism = True
        else:
            self.ism = False
        self.eval_best = 0
        self.eval_best_epoch = 0
        self.pblog = tool.get_pblog()
        self.pblog.total = self.epoch
        self.tblog = SummaryWriter(join(args['tbx_dir'], self.midtf))

    def __del__(self):
        if hasattr(self, 'tblog'):
            self.tblog.close()

    def train(self):
        self.pblog.info(self.midtf)
        for epoch in range(self.epoch):
            temp = self.ac.change_opt(epoch, self.m)
            if temp is not None:
                optimizer = temp
            self.m.train()
            loss_l = []
            loss_n = []
            for t_x, t_y in self.dl.train:
                t_x = t_x.cuda(non_blocking=True)
                t_y = t_y.cuda(non_blocking=True)
                for ii in range(3):
                    t_x[:, ii, :, :].sub_(self.dl.mean[ii]).div_(self.dl.std[ii])
                loss = self.ac.cal_loss(t_x, t_y, self.m)
                optimizer.zero_grad()
                loss[0].backward()
                optimizer.step()
                loss_l.append([ii.item() for ii in loss])
                loss_n.append(t_y.size(0))
            loss_l = np.array(loss_l).T
            loss_n = np.array(loss_n)
            loss = (loss_l*loss_n).sum(axis=1)/loss_n.sum()
            msg = 'Epoch: {:>3}'.format(epoch)
            temp = dict()
            for n, s in zip(loss, self.ac.loss_legend):
                msg += s.format(n)
                temp[s.split(':')[0][2:]] = n
            self.tblog.add_scalars('loss', temp, epoch)
            self.pblog.info(msg, move=True)
            if not self.no_eval:
                self.eval(epoch)
                # with torch.no_grad():
                #     self.eval(epoch)
        if not exists(self.model_dir):
            os.makedirs(self.model_dir)
        path = os.path.join(self.model_dir, self.mpkl)
        if self.ism:
            torch.save(self.m.module.state_dict(), path)
        else:
            torch.save(self.m.state_dict(), path)
        self.pblog.debug('training completed, save model')
        temp = 'Result, Best: {:.2f}%, Epoch: {}'.format(self.eval_best, self.eval_best_epoch)
        self.tblog.add_text('best', temp, self.epoch)
        self.pblog.info(temp)

    def eval(self, epoch):
        self.m.eval()
        ll = len(self.ac.eval_legend)
        if self.ac.eval_on_train:
            c_right = np.zeros(ll, np.float32)
            c_sum = np.zeros(ll, np.float32)
            for x, y in self.dl.train:
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
                for ii in range(3):
                    x[:, ii, :, :].sub_(self.dl.mean[ii]).div_(self.dl.std[ii])
                a, b = self.ac.cal_eval(x, y, self.m)
                c_right += a
                c_sum += b
            msg = 'train->   '
            tbd = dict()
            for n, s in zip(c_right / c_sum, self.ac.eval_legend):
                msg += s.format(n)
                tbd[s.split(':')[0][2:]] = n
            self.tblog.add_scalars('eval/train', tbd, epoch)
            self.pblog.info(msg)
        c_right = np.zeros(ll, np.float32)
        c_sum = np.zeros(ll, np.float32)
        for x, y in self.dl.eval:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            for ii in range(3):
                x[:, ii, :, :].sub_(self.dl.mean[ii]).div_(self.dl.std[ii])
            a, b = self.ac.cal_eval(x, y, self.m)
            c_right += a
            c_sum += b
        msg = 'eval->    '
        c_res = c_right/c_sum
        tbd = dict()
        for n, s in zip(c_res, self.ac.eval_legend):
            msg += s.format(n)
            tbd[s.split(':')[0][2:]] = n
        self.tblog.add_scalars('eval/eval', tbd, epoch)
        self.pblog.info(msg)
        if c_res[0] > self.eval_best and epoch > 30:
            self.eval_best_epoch = epoch
            self.eval_best = c_res[0]
            if not exists(self.best_model_dir):
                os.makedirs(self.best_model_dir)
            path = os.path.join(self.best_model_dir, 'Best_'+self.mpkl)
            if self.ism:
                torch.save(self.m.module.state_dict(), path)
            else:
                torch.save(self.m.state_dict(), path)
            self.pblog.debug('update the best model')


'''
class Adversary:
    def __init__(self, args):
        self.cuda = args['cuda']
        self.dl = DataSet.get_dataloader(args)
        args['num_classes'] = self.dl.num_classes
        self.m = Model.get_net(args)
        p = join(args['best_model_dir'], 'Best_{}.model'.format(args['midtf']))
        self.m.load_state_dict(torch.load(p, map_location='cpu'))
        self.m.eval()
        for p in self.m.parameters():
            p.detach_()
        args['mean'] = self.dl.mean
        args['std'] = self.dl.std
        self.atk = Attack.get_attack(args)
        if args['attack_on_train']:
            temp = 'train'
        else:
            temp = 'eval'
        self.atkidtf = '{}_{}_{}{}'.format(args['midtf'], temp, args['atkid'], self.atk.strkk)
        self.atk_save_dir = join(args['adversarial_dir'], self.atkidtf)
        if exists(self.atk_save_dir):
            if args['reattack']:
                shutil.rmtree(self.atk_save_dir)
            else:
                raise IOError
        os.makedirs(self.atk_save_dir)
        for ii in self.dl.ds_attack.classes:
            os.mkdir(join(self.atk_save_dir, ii))
        self.pblog = tool.get_pblog()
        self.pblog.total = int(len(self.dl.ds_attack)/len(self.cuda.split(','))/self.dl.batch_attack)

    def _mpwk_batch(self, cuda, dl, q):
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda
        self.m.cuda()
        for x, y, fids in dl:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            total = y.size(0)
            for ii in range(3):
                x[:, ii, :, :].sub_(self.dl.mean[ii]).div_(self.dl.std[ii])
            yhat = self.m(x).argmax(1)
            index = (yhat == y).nonzero().squeeze()
            right = index.numel()
            if not right:
                q.put((0, 0, total))
                continue
            x = x.index_select(0, index)
            y = y.index_select(0, index)
            if right > 1:
                fids = [fids[ii] for ii in index]
            else:
                fids = [fids[index]]
            x = self.atk(self.m, x, y)
            x_nml = x.clone().detach()
            for ii in range(3):
                x_nml[:, ii, :, :].sub_(self.dl.mean[ii]*255).div_(self.dl.std[ii]*255)
            yhat = self.m(x_nml).argmax(1)
            index = (yhat != y).nonzero().squeeze()
            adv = index.numel()
            if adv > 1:
                for ii in index:
                    p = join(self.atk_save_dir, self.dl.ds_attack.classes[y[ii]], 'adv_{}.bmp'.format(fids[ii]))
                    Image.fromarray(x[ii].to(device='cpu', dtype=torch.uint8).numpy().transpose((1, 2, 0)),
                                    mode='RGB').save(p)
            elif adv > 0:
                p = join(self.atk_save_dir, self.dl.ds_attack.classes[y[index]], 'adv_{}.bmp'.format(fids[index]))
                Image.fromarray(x[index].to(device='cpu', dtype=torch.uint8).numpy().transpose((1, 2, 0)),
                                mode='RGB').save(p)
            else:
                pass
            q.put((adv, right, total))
        q.put(None)

    def attack(self):
        self.pblog.info(self.atkidtf)
        q = mp.Queue()
        lcuda = self.cuda.split(',')
        dls = self.dl.split_attack(len(lcuda))
        mps = [mp.Process(target=self._mpwk_batch, args=(ii, dl, q)) for ii, dl in zip(lcuda, dls)]
        for pp in mps:
            pp.start()
        lcuda = len(lcuda)
        ii = 2
        n_done = 2
        adv = right = total = 0
        while True:
            msg = q.get()
            if msg is None:
                if n_done > lcuda:
                    break
                else:
                    n_done += 1
                    continue
            adv += msg[0]
            right += msg[1]
            total += msg[2]
            msg = 'original: {}/{}={:.2f}%, adversarial: {}/{}={:.2f}%'.format(right, total, 100*right/total, adv,
                                                                               right, 100*adv/right)
            if ii > lcuda:
                ii = 2
                self.pblog.info(msg, move=True)
            else:
                ii += 1
                self.pblog.info(msg)
        for pp in mps:
            pp.join()
        msg = 'Result, original: {}/{}={:.2f}%, adversarial: {}/{}={:.2f}%'.format(right, total, 100*right/total, adv,
                                                                                   right, 100*adv/right)
        self.pblog.info(msg)
'''


class Adversary:
    def __init__(self, args):
        self.cuda = args['cuda']
        self.dl = DataSet.get_dataloader(args)
        args['num_classes'] = self.dl.num_classes
        self.m = Model.get_net(args)
        p = join(args['best_model_dir'], 'Best_{}.model'.format(args['midtf']))
        self.m.load_state_dict(torch.load(p, map_location='cpu'))
        self.m.eval()
        for p in self.m.parameters():
            p.detach_()
        args['mean'] = self.dl.mean
        args['std'] = self.dl.std
        self.atk = Attack.get_attack(args)
        if args['attack_on_train']:
            temp = 'train'
        else:
            temp = 'eval'
        self.atkidtf = '{}_{}_{}{}'.format(args['midtf'], temp, args['atkid'], self.atk.strkk)
        self.atk_save_dir = join(args['adversarial_dir'], self.atkidtf)
        if exists(self.atk_save_dir):
            if args['reattack']:
                shutil.rmtree(self.atk_save_dir)
            else:
                raise IOError
        os.makedirs(self.atk_save_dir)
        for ii in self.dl.ds_attack.classes:
            os.mkdir(join(self.atk_save_dir, ii))
        self.pblog = tool.get_pblog()
        self.pblog.total = int(len(self.dl.ds_attack)/len(self.cuda.split(','))/self.dl.batch_attack)

    def _mpwk_batch(self, cuda, dl, q):
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda
        self.m.cuda()
        for x, y, fids in dl:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            for ii in range(3):
                x[:, ii, :, :].sub_(self.dl.mean[ii]).div_(self.dl.std[ii])
            x = self.atk(self.m, x, y)
            for img, tag, name in zip(x, y, fids):
                p = join(self.atk_save_dir, self.dl.ds_attack.classes[tag], 'a_{}.bmp'.format(name))
                Image.fromarray(img.to(device='cpu', dtype=torch.uint8).numpy().transpose((1, 2, 0)),
                                mode='RGB').save(p)
            q.put(x.size(0))
        q.put(None)

    def attack(self):
        self.pblog.info(self.atkidtf)
        q = mp.Queue()
        lcuda = self.cuda.split(',')
        dls = self.dl.split_attack(len(lcuda))
        mps = [mp.Process(target=self._mpwk_batch, args=(ii, dl, q)) for ii, dl in zip(lcuda, dls)]
        for pp in mps:
            pp.start()
        lcuda = len(lcuda)
        ii = 2
        n_done = 2
        while True:
            msg = q.get()
            if msg is None:
                if n_done > lcuda:
                    break
                else:
                    n_done += 1
                    continue
            if ii > lcuda:
                ii = 2
                self.pblog.refresh(move=True)
            else:
                ii += 1
        for pp in mps:
            pp.join()


class Tester:
    def __init__(self, args):
        os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda']
        self.midtf = args['midtf']
        self.testidtf = args['testidtf']
        args['test_path'] = join(args['adversarial_dir'], self.testidtf)
        self.dl = DataSet.get_dataloader(args)
        args['num_classes'] = self.dl.num_classes
        self.m = Model.get_net(args)
        p = join(args['best_model_dir'], 'Best_{}.model'.format(self.midtf))
        self.m.load_state_dict(torch.load(p, map_location='cpu'))
        self.m.eval()
        for p in self.m.parameters():
            p.detach_()
        self.m.cuda()
        self.pblog = tool.get_pblog()
        self.pblog.total = len(self.dl.test)

    def test(self):
        self.pblog.info('{}&{}'.format(self.midtf, self.testidtf))
        total = right = 0
        for x, y in self.dl.test:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            for ii in range(3):
                x[:, ii, :, :].sub_(self.dl.mean[ii]).div_(self.dl.std[ii])
            yhat = self.m(x).argmax(1)
            right += (yhat == y).sum().item()
            total += y.size(0)
            self.pblog.info('defense: {}/{}={:.2f}%'.format(right, total, 100 * right / total), move=True)
        self.pblog.info('Result, defense: {}/{}={:.2f}%'.format(right, total, 100 * right / total))


class Vfeat:
    def __init__(self, args):
        os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda']
        self.v_dir = args['v_dir']
        self.midtf = args['midtf']
        self.dl = DataSet.get_dataloader(args)
        args['num_classes'] = self.dl.num_classes
        self.m = Model.get_net(args)
        p = join(args['best_model_dir'], 'Best_{}.model'.format(args['midtf']))
        self.m.load_state_dict(torch.load(p, map_location='cpu'))
        self.m.eval()
        for p in self.m.parameters():
            p.detach_()
        self.m.cuda()

    def gen_v(self):
        import matplotlib as mpl
        mpl.use('Agg')
        from matplotlib import pyplot as plt
        num_classes = self.dl.num_classes
        num_sample = 50
        classes, imgs = self.dl.get_v(num_sample)
        imgs = imgs.cuda()
        for ii in range(3):
            imgs[:, ii, :, :].sub_(self.dl.mean[ii]).div_(self.dl.std[ii])
        logit, feats = self.m(imgs, True)
        feats = feats.cpu().numpy()
        w = self.m.fc.weight.cpu().numpy()
        logit = logit.cpu().numpy()
        plt.figure(figsize=(50, 20))
        kk = 8
        for ii in range(num_classes):
            for jj in range(5):
                plt.subplot(num_classes, kk, ii * kk + jj + 1)
                plt.plot(feats[ii * num_sample + jj, :])
                plt.title('{}_{}'.format(classes[ii], jj))
            plt.subplot(num_classes, kk, ii * kk + 6)
            plt.plot(feats[ii * num_sample:(ii + 1) * num_sample, :].mean(0))
            plt.title('{}_mean50'.format(classes[ii]))

            plt.subplot(num_classes, kk, ii * kk + 7)
            plt.plot(w[ii, :])
            plt.subplot(num_classes, kk, ii * kk + 8)
            plt.plot(logit[ii*num_sample, :])
        if not exists(self.v_dir):
            os.makedirs(self.v_dir)
        plt.savefig(join(self.v_dir, '{}.jpg'.format(self.midtf)))

    def gen_v2(self):
        import matplotlib as mpl
        mpl.use('Agg')
        from matplotlib import pyplot as plt
        num_classes = self.dl.num_classes
        num_sample = 50
        classes, imgs = self.dl.get_v(num_sample)
        imgs = imgs.cuda()
        for ii in range(3):
            imgs[:, ii, :, :].sub_(self.dl.mean[ii]).div_(self.dl.std[ii])
        logit, _, _, f3, _, f5 = self.m(imgs, True)

        feats = f5.cpu().numpy()
        f3 = f3.max(2)[0].max(2)[0].cpu().numpy()
        w = self.m.fc.weight.cpu().numpy()
        logit = logit.cpu().numpy()
        plt.figure(figsize=(50, 40))
        kk = 8+6
        for ii in range(num_classes):
            for jj in range(5):
                plt.subplot(num_classes, kk, ii * kk + jj + 1)
                plt.plot(feats[ii * num_sample + jj, :])
                plt.title('{}_{}'.format(classes[ii], jj))

                plt.subplot(num_classes, kk, ii * kk + jj + 1 + 8)
                plt.plot(f3[ii * num_sample + jj, :])
                plt.title('{}_{}_f3'.format(classes[ii], jj))

            plt.subplot(num_classes, kk, ii * kk + 6)
            plt.plot(feats[ii * num_sample:(ii + 1) * num_sample, :].mean(0))
            plt.title('{}_mean50'.format(classes[ii]))

            plt.subplot(num_classes, kk, ii * kk + 6 + 8)
            plt.plot(f3[ii * num_sample:(ii + 1) * num_sample, :].mean(0))
            plt.title('{}_mean50_f3'.format(classes[ii]))

            plt.subplot(num_classes, kk, ii * kk + 7)
            plt.plot(w[ii, :])
            plt.subplot(num_classes, kk, ii * kk + 8)
            plt.plot(logit[ii*num_sample, :])
        if not exists(self.v_dir):
            os.makedirs(self.v_dir)
        plt.savefig(join(self.v_dir, 'f53f_{}.jpg'.format(self.midtf)))



'''
def gen_v(self):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pyplot as plt
    num_classes = self.dl.num_classes
    num_sample = 50
    classes, imgs = self.dl.get_v(num_sample)
    imgs = imgs.cuda()
    for ii in range(3):
        imgs[:, ii, :, :].sub_(self.dl.mean[ii]).div_(self.dl.std[ii])
    _, feats = self.m(imgs, True)
    feats = feats.cpu().numpy()
    plt.figure(figsize=(50, 20))
    for ii in range(num_classes):
        for jj in range(5):
            plt.subplot(num_classes, 6, ii * 6 + jj + 1)
            plt.plot(feats[ii * num_sample + jj, :])
            plt.title('{}_{}'.format(classes[ii], jj))
        plt.subplot(num_classes, 6, ii * 6 + 6)
        plt.plot(feats[ii * num_sample:(ii + 1) * num_sample, :].mean(0))
        plt.title('{}_mean50'.format(classes[ii]))
    plt.savefig('{}.jpg'.format(self.save_path))
'''


if __name__ == '__main__':
    pass
