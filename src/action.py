import numpy as np
import torch
import torch.nn.functional as F


class BAc:  # base action
    loss_legend = ['| loss: {:0<10.8f}']
    eval_on_train = True
    eval_legend = ['| acc: {:0<5.2f}%']
    strkk = ''

    @staticmethod
    def cal_loss(x, y, net):
        yhat = net(x)
        loss = F.cross_entropy(yhat, y)
        return loss,

    @staticmethod
    def cal_eval(x, y, net):
        count_right = np.empty(1, np.float32)
        count_sum = np.empty(1, np.float32)
        yhat = net(x).argmax(1)
        count_right[0] = (yhat == y).sum().item()
        count_sum[0] = y.size(0)
        return 100*count_right, count_sum

    @staticmethod
    def change_opt(epoch, net):
        if epoch == 0:
            return torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        elif epoch == 30:
            return torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        else:
            return None


class TAc(BAc):  # train action
    loss_legend = ['| loss: {:0<10.8f}']
    eval_on_train = False
    eval_legend = ['| acc c: {:0<5.2f}%', '| acc n: {:0<5.2f}%']

    def __init__(self, args):
        self.strkk = '_rtp' + args['kk']
        self.rtp = args['kk'].split(',')
        self.nrate = int(self.rtp[0])
        if self.rtp[1] == '0':    # uniform
            self.ntype = self._unf           # noise function
            self.nlevel = float(self.rtp[2]) # noise level
        elif self.rtp[1] == '1':  # normal
            self.ntype = self._nm            # noise function
            self.nstd = float(self.rtp[2])   # noise level -> nstd (in normal)
        else:
            raise ValueError('no noise type: ' + self.rtp[1])
        self.mean = args['mean']
        self.std = args['std']

    def cal_loss(self, x, y, net):
        n_x = self._gen_nx(x, self.nrate)
        yhat = net(torch.cat([x, n_x]))
        return F.cross_entropy(yhat, torch.cat([y]*(self.nrate+1))),

    def cal_eval(self, x, y, net):
        count_right = np.empty(2, np.float32)
        count_sum = np.empty(2, np.float32)
        n_x = self._gen_nx(x, 1)
        yhat1, yhat2 = net(torch.cat([x, n_x])).argmax(1).chunk(2)
        count_right[0] = (yhat1 == y).sum().item()
        count_right[1] = (yhat2 == y).sum().item()
        count_sum[0] = count_sum[1] = y.size(0)
        return 100 * count_right, count_sum

    def _gen_nx(self, x, nrate): # generate noise x
        temp = list(x.shape)
        batch = temp[0]
        temp[0] *= nrate
        n_x = torch.empty(temp, dtype=x.dtype, device=x.device)
        self.ntype(n_x) # add noise
        for ii, (jj, kk) in enumerate(zip(self.mean, self.std)):
            for temp in range(nrate):
                n_x[temp * batch:(temp + 1) * batch, ii, :, :].add_(x[:, ii, :, :]).clamp_(-jj / kk, (1 - jj) / kk)
        return n_x

    def _unf(self, n_x):
        for ii, jj in enumerate(self.std):
            n_x[:, ii, :, :].uniform_(-self.nlevel / jj, self.nlevel / jj)

    def _nm(self, n_x):
        for ii, jj in enumerate(self.std):
            n_x[:, ii, :, :].normal_(0, self.nstd)
            n_x[:, ii, :, :].div_(jj)


class NAc(TAc):
    loss_legend = ['| loss: {:0<10.8f}', '| ce: {:0<10.8f}', '| k1: {:0<10.8f}']
    eval_legend = ['| acc: {:0<5.2f}%', '| acc n: {:0<5.2f}%', '| L1: {:0<6.3f}%']

    def __init__(self, args):
        super().__init__(args)
        self.strkk = '_rtpkf' + args['kk']
        self.k = float(self.rtp[3])  # noise weight
        self.f = int(self.rtp[4])    # feature (which block)

    def cal_loss(self, x, y, net):
        n_x = self._gen_nx(x, self.nrate)
        yhat, feat = net(torch.cat([x, n_x]), True)
        loss0 = F.cross_entropy(yhat, torch.cat([y]*(self.nrate+1)))
        if self.f < 6:
            feat = feat[self.f]
            batch = x.size(0)
            loss1 = F.mse_loss(feat[batch:], torch.cat([feat[:batch].detach()]*self.nrate))
        else:
            raise ValueError('no such f')
        loss1 = self.k * loss1
        return loss0 + loss1, loss0, loss1

    def cal_eval(self, x, y, net):
        count_right = np.empty(3, np.float32)
        count_sum = np.empty(3, np.float32)
        n_x = self._gen_nx(x, 1)
        yhat, feat = net(torch.cat([x, n_x]), True)
        yhat1, yhat2 = yhat.argmax(1).chunk(2)
        count_right[0] = (yhat1 == y).sum().item()
        count_right[1] = (yhat2 == y).sum().item()
        count_sum[0] = count_sum[1] = y.size(0)
        if self.f < 6:
            c1, c2 = feat[self.f].chunk(2)
            count_sum[2] = c1.sum().item()
            count_right[2] = count_sum[2] - F.l1_loss(c2, c1, reduction='sum').item()
        else:
            raise ValueError('no such f')
        return 100 * count_right, count_sum


class AAc(TAc):
    loss_legend = ['| loss: {:0<10.8f}', '| ce: {:0<10.8f}', '| k1: {:0<10.8f}']
    eval_legend = ['| acc: {:0<5.2f}%', '| acc n: {:0<5.2f}%', '| L1: {:0<6.3f}%']

    def __init__(self, args):
        super().__init__(args)
        self.strkk = '_rtpk' + args['kk']
        self.k = float(self.rtp[3])

    def cal_loss(self, x, y, net):
        n_x = self._gen_nx(x, self.nrate)
        yhat, rimg = net(torch.cat([x, n_x]), True)
        loss0 = F.cross_entropy(yhat, torch.cat([y] * (self.nrate + 1)))
        xx = x.clone().detach()
        for ii in range(3):
            xx[:, ii, :, :].mul_(2*self.std[ii]).add_(2*self.mean[ii]-1)
        loss1 = self.k * F.mse_loss(rimg[x.size(0):], torch.cat([xx]*self.nrate))
        return loss0 + loss1, loss0, loss1

    def cal_eval(self, x, y, net):
        count_right = np.empty(3, np.float32)
        count_sum = np.empty(3, np.float32)
        n_x = self._gen_nx(x, 1)
        yhat, rimg = net(torch.cat([x, n_x]), True)
        yhat1, yhat2 = yhat.argmax(1).chunk(2)
        count_right[0] = (yhat1 == y).sum().item()
        count_right[1] = (yhat2 == y).sum().item()
        count_sum[0] = count_sum[1] = y.size(0)
        rimg = rimg[x.size(0):]
        count_sum[2] = rimg.abs().sum().item()
        count_right[2] = count_sum[2] - F.l1_loss(rimg, x, reduction='sum').item()
        return 100 * count_right, count_sum


class MAc(TAc):
    loss_legend = ['| loss: {:0<10.8f}', '| ce: {:0<10.8f}', '| k1: {:0<10.8f}', '| k2: {:0<10.8f}']
    eval_legend = ['| acc: {:0<5.2f}%']

    def __init__(self, args):
        super().__init__(args)
        self.strkk = '_rtpkk' + args['kk']
        self.k1 = float(self.rtp[3])
        self.k2 = float(self.rtp[4])

    def cal_loss(self, x, y, net):
        n_x = self._gen_nx(x, self.nrate)
        yhat, feat, rimg = net(torch.cat([x, n_x]), True)
        loss0 = F.cross_entropy(yhat, torch.cat([y]*(self.nrate+1)))
        batch = x.size(0)
        loss1 = self.k1 * F.mse_loss(feat[batch:], torch.cat([feat[:batch].detach()]*self.nrate))
        xx = x.clone().detach()
        for ii in range(3):
            xx[:, ii, :, :].mul_(2 * self.std[ii]).add_(2 * self.mean[ii] - 1)
        loss2 = self.k2 * F.mse_loss(rimg[batch:], torch.cat([xx] * self.nrate))
        return loss0+loss1+loss2, loss0, loss1, loss2

    def cal_eval(self, x, y, net):
        return BAc.cal_eval(x, y, net)


class GAc:
    loss_legend = ['| loss: {:0<10.8f}']
    eval_on_train = False
    eval_legend = ['| acc: {:0<5.2f}%']
    strkk = ''

    def __init__(self, args):
        self.state = 0    # 0:clear 1:adv
        self.lim = 16/255
        self.mean = args['mean']
        self.std = args['std']

    def cal_loss(self, x, y, net):
        if self.state == 1:
            net.eval()
            self._add_n(x, y, net)
            net.train()
        yhat = net(x)
        loss = F.cross_entropy(yhat, y)
        return loss,

    def cal_eval(self, x, y, net):
        count_right = np.empty(1, np.float32)
        count_sum = np.empty(1, np.float32)
        if self.state == 1:
            self._add_n(x, y, net)
        yhat = net(x).argmax(1)
        count_right[0] = (yhat == y).sum().item()
        count_sum[0] = y.size(0)
        return 100 * count_right, count_sum

    def change_opt(self, epoch, net):
        if epoch == 0:
            return torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        elif epoch == 20:
            self.state = 1
            return None
        elif epoch == 30:
            return torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        elif epoch == 60:
            return torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
        else:
            return None

    def _add_n(self, x, y, net):
        tur = torch.zeros_like(x)
        for _ in range(10):
            images = x + tur
            images.requires_grad_()
            predictions = net(images)
            loss = F.cross_entropy(predictions, y, reduction='sum')
            loss.backward()
            grad = images.grad.sign()
            r = np.random.rand() * 0.6 + 0.2
            r *= self.lim
            for ii, kk in enumerate(self.std):
                temp = r / kk
                tur[:, ii, :, :].add_(grad[:, ii, :, :].mul_(temp)).clamp_(-self.lim, self.lim)
        x.add_(tur)
        for ii in range(3):
            x[:, ii, :, :].clamp_(-self.mean[ii] / self.std[ii], (1 - self.mean[ii]) / self.std[ii])


'''
class SparseNAction(BAc):
    loss_legend = ['| loss: {:0<10.8f}', '| ce: {:0<10.8f}', '| L1: {:0<10.8f}']
    eval_on_train = False
    eval_legend = ['| acc: {:0<5.3f}%']

    def __init__(self, args):
        self.k1 = float(args['kk'])
        self.strkk = '_k{}'.format(self.k1)
        self.mean = args['mean']
        self.std = args['std']

    def cal_loss(self, x, y, net):
        yhat, feats = net(x, True)
        loss0 = F.cross_entropy(yhat, y)
        loss1 = self.k1*feats.norm(1, 1).mean()
        return loss0+loss1, loss0, loss1


class SparseAction(BAc):
    loss_legend = ['| loss: {:0<10.8f}', '| ce: {:0<10.8f}', '| L1: {:0<10.8f}', '| L2: {:0<10.8f}']
    eval_on_train = False
    eval_legend = ['| acc: {:0<5.3f}%']

    def __init__(self, args):
        self.k1, self.k2 = args['kk'].split(',')
        self.k1 = float(self.k1)
        self.k2 = float(self.k2)
        self.strkk = '_k{}_p{}'.format(self.k1, self.k2)
        self.mean = args['mean']
        self.std = args['std']

    def cal_loss(self, x, y, net):
        yhat, feats = net(x, True)
        loss0 = F.cross_entropy(yhat, y)
        loss1 = self.k1 * feats.norm(1, 1).mean()
        loss2 = self.k2 * (feats.norm(dim=1)-1).pow(2).mean()
        return loss0 + loss1 + loss2, loss0, loss1, loss2


class SEMDAction(SparseAction):
    def __init__(self, args):
        self.k1, self.k2 = args['kk'].split(',')
        self.k1 = float(self.k1)
        self.k2 = float(self.k2)
        self.strkk = '_ks{}_kd{}'.format(self.k1, self.k2)
        self.mean = args['mean']
        self.std = args['std']
        self.num_classes = args['num_classes']
        self.num_group = 2

    def cal_loss(self, x, y, net):
        yhat, feats = net(x, True)
        loss0 = F.cross_entropy(yhat, y)
        loss1 = self.k1 * feats.norm(1, 1).mean()
        temp = [[] for _ in range(self.num_classes)]
        for ii in range(len(y)):
            temp[y[ii].item()].append(ii)
        res1 = []
        res2 = []
        res3 = []
        for _ in range(self.num_group):
            for ii in range(self.num_classes):
                a, b = np.random.choice(temp[ii], 2, False)
                res1.append(a)
                res2.append(b)
                res3.append(1)
                for jj in range(ii + 1, self.num_classes):
                    a = np.random.choice(temp[ii])
                    b = np.random.choice(temp[jj])
                    res1.append(a)
                    res2.append(b)
                    res3.append(-1)
        f1 = feats.index_select(0, torch.tensor(res1, dtype=torch.int64, device='cuda'))
        f2 = feats.index_select(0, torch.tensor(res2, dtype=torch.int64, device='cuda'))
        loss2 = (f1-f2).abs().sum(1)
        loss2 = loss2 * torch.tensor(res3, dtype=torch.float32, device='cuda')
        loss2 = self.k2 * loss2.mean()
        return loss0 + loss1 + loss2, loss0, loss1, loss2


class MStest(BAc):
    loss_legend = ['| loss: {:0<10.8f}', '| ce: {:0<10.8f}', '| L15: {:0<10.8f}', '| L25: {:0<10.8f}',
                   '| L13: {:0<10.8f}', '| L23: {:0<10.8f}']
    eval_on_train = False

    def __init__(self, args):
        self.ks1, self.kd1, self.ks2, self.kd2 = args['kk'].split(',')
        self.ks1, self.kd1, self.ks2, self.kd2 = float(self.ks1), float(self.kd1), float(self.ks2), float(self.kd2)
        self.strkk = '_ks{}_kd{}_ks{}_kd{}'.format(self.ks1, self.kd1, self.ks2, self.kd2)
        self.num_classes = args['num_classes']
        self.num_group = 2

    def cal_loss(self, x, y, net):
        yhat, _, _, f3, _, f5 = net(x, True)
        loss0 = F.cross_entropy(yhat, y)
        # f5
        loss1 = self.ks1 * f5.norm(1, 1).mean()
        temp = [[] for _ in range(self.num_classes)]
        for ii in range(len(y)):
            temp[y[ii].item()].append(ii)
        res1 = []
        res2 = []
        res3 = []
        for _ in range(self.num_group):
            for ii in range(self.num_classes):
                a, b = np.random.choice(temp[ii], 2, False)
                res1.append(a)
                res2.append(b)
                res3.append(1)
                for jj in range(ii + 1, self.num_classes):
                    a = np.random.choice(temp[ii])
                    b = np.random.choice(temp[jj])
                    res1.append(a)
                    res2.append(b)
                    res3.append(-1)
        sf1 = f5.index_select(0, torch.tensor(res1, dtype=torch.int64, device='cuda'))
        sf2 = f5.index_select(0, torch.tensor(res2, dtype=torch.int64, device='cuda'))
        loss2 = (sf1 - sf2).abs().sum(1)
        loss2 = loss2 * torch.tensor(res3, dtype=torch.float32, device='cuda')
        loss2 = self.kd1 * loss2.mean()
        # f3
        loss3 = self.ks2 * f3.norm(1, 1).mean()
        temp = [[] for _ in range(self.num_classes)]
        for ii in range(len(y)):
            temp[y[ii].item()].append(ii)
        res1 = []
        res2 = []
        res3 = []
        for _ in range(self.num_group):
            for ii in range(self.num_classes):
                a, b = np.random.choice(temp[ii], 2, False)
                res1.append(a)
                res2.append(b)
                res3.append(1)
                for jj in range(ii + 1, self.num_classes):
                    a = np.random.choice(temp[ii])
                    b = np.random.choice(temp[jj])
                    res1.append(a)
                    res2.append(b)
                    res3.append(-1)
        sf1 = f3.max(2)[0].max(2)[0].index_select(0, torch.tensor(res1, dtype=torch.int64, device='cuda'))
        sf2 = f3.max(2)[0].max(2)[0].index_select(0, torch.tensor(res2, dtype=torch.int64, device='cuda'))
        loss4 = (sf1 - sf2).abs().sum(1)
        loss4 = loss4 * torch.tensor(res3, dtype=torch.float32, device='cuda')
        loss4 = self.kd2 * loss4.mean()

        return loss0 + loss1 + loss2 + loss3 + loss4, loss0, loss1, loss2, loss3, loss4


class Base2Action:
    loss_legend = ['| loss: {:0<10.8f}']
    eval_on_train = True
    eval_legend = ['| acc: {:0<5.3f}%']
    strkk = ''

    @staticmethod
    def cal_loss(x, y, net):
        yhat = net(x)
        loss = F.cross_entropy(yhat, y)
        return loss,

    @staticmethod
    def cal_eval(x, y, net):
        count_right = np.empty(1, np.float32)
        count_sum = np.empty(1, np.float32)
        yhat = net(x).argmax(1)
        count_right[0] = (yhat == y).sum().item()
        count_sum[0] = y.size(0)
        return 100*count_right, count_sum

    @staticmethod
    def change_opt(epoch, net):
        if epoch == 0:
            return torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        elif epoch == 30:
            return torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
        else:
            return None
'''


def get_action(args):
    if args['acid'] == 'b': # base
        return BAc()
    elif args['acid'] == 't': # train
        return TAc(args)
    elif args['acid'] == 'n': # noise
        return NAc(args)
    elif args['acid'] == 'a': # ae
        return AAc(args)
    elif args['acid'] == 'm': # mix
        return MAc(args)
    elif args['acid'] == 'g': # gaussion noise
        return GAc(args)
    else:
        raise ValueError('No action: {}'.format(args['acid']))
