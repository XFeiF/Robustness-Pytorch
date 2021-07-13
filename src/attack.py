import torch
import torch.nn.functional as F
'''
input  meat,std,p has detached, cls true
output 0~255
'''


class FGSM:
    def __init__(self, args):
        self.epsilon = float(args['kk'])/255
        self.strkk = '_e' + args['kk']
        self.mean = args['mean']
        self.std = args['std']

    def __call__(self, model, images, target):
        images.requires_grad_()
        predictions = model(images)
        loss = F.cross_entropy(predictions, target, reduction='sum')
        loss.backward()
        grad = images.grad.sign_()
        images.detach_()
        for ii, (jj, kk) in enumerate(zip(self.mean, self.std)):
            images[:, ii, :, :].add_(grad[:, ii, :, :].mul_(self.epsilon / kk)).clamp_(-jj / kk, (1 - jj) / kk)
        for ii in range(3):
            images[:, ii, :, :].mul_(self.std[ii]*255).add_(self.mean[ii]*255)
        return images.round_()


class IFGSM:
    def __init__(self, args):
        a, b = args['kk'].split(',')
        self.epsilon = float(a)/255
        self.it = int(b)
        self.strkk = '_ei' + args['kk']
        self.mean = args['mean']
        self.std = args['std']

    def __call__(self, model, images, target):
        tur = torch.zeros_like(images)
        for _ in range(self.it):
            x = images + tur
            x.requires_grad_()
            predictions = model(x)
            loss = F.cross_entropy(predictions, target, reduction='sum')
            loss.backward()
            grad = x.grad.sign()
            for ii, kk in enumerate(self.std):
                temp = self.epsilon / kk
                tur[:, ii, :, :].add_(grad[:, ii, :, :].mul_(temp)).clamp_(-temp, temp)
        images.add_(tur)
        for ii in range(3):
            images[:, ii, :, :].mul_(self.std[ii]*255).add_(self.mean[ii]*255)
        return images.clamp_(0, 255).round_()


class CW:
    def __init__(self, args):
        self.search, self.iteration, self.k, self.lr, self.initc = args['kk'].split(',')
        self.search = int(self.search)
        self.iteration = int(self.search)
        self.k = float(self.k)
        self.lr = float(self.lr)
        self.initc = float(self.initc)
        self.strkk = '_siklc{}'.format(args['kk'])
        self.mean = args['mean']
        self.std = args['std']

    def __call__(self, model, images, target):
        temp = target.size(0)
        rangey = range(temp)
        c = torch.tensor([self.initc]*temp, dtype=torch.float32, device='cuda')
        lower_bound = torch.zeros(temp, dtype=torch.float32, device='cuda')
        upper_bound = torch.tensor([float('inf')]*temp, dtype=torch.float32, device='cuda')
        m = torch.empty_like(images)
        s = torch.empty_like(images)
        for ii in range(3):
            m[:, ii, :, :] = 1 - 2 * self.mean[ii]
            s[:, ii, :, :] = 2 * self.std[ii]
            images[:, ii, :, :].mul_(self.std[ii]).add_(self.mean[ii])
        images.div_(1-images).log_().div_(2)
        for _ in range(self.search):
            x = images.clone().detach().requires_grad_(True)
            opt = torch.optim.SGD([x], lr=self.lr, momentum=0.9)
            for _ in range(self.iteration):
                temp = (x.tanh()+m)/s
                temp = model(temp)
                logits = temp.clone().detach()
                logits[rangey, target] = float('-inf')
                adv_loss = temp[rangey, target] - temp[rangey, logits.argmax(1)]
                d_index = adv_loss > -self.k
                loss = (x.tanh()-images.tanh()).pow(2).sum()/4 + c[d_index].dot(adv_loss[d_index])
                opt.zero_grad()
                loss.backward()
                opt.step()
            lower_bound[d_index] = c[d_index]
            upper_bound[~d_index] = c[~d_index]
            d_index = torch.isinf(upper_bound)
            c[d_index] *= 10
            c[~d_index] = (lower_bound[~d_index]+upper_bound[~d_index])/2
        return x.detach_().tanh_().add_(1).mul_(255/2).round_()


def get_attack(args):
    if args['atkid'] == 'FGSM':
        return FGSM(args)
    elif args['atkid'] == 'IFGSM':
        return IFGSM(args)
    elif args['atkid'] == 'CW':
        return CW(args)
    else:
        raise ValueError
