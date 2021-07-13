import os
from os.path import join, exists
import shutil
from PIL import Image
import numpy as np
import torch
import torchvision
import torch.utils.data as Data
from torchvision import transforms


class AttackImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, root, tf):
        super().__init__(root, transform=tf)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        _, path = os.path.split(path)
        path, _ = os.path.splitext(path)
        return sample, target, path


class DsList2(Data.Dataset):
    def __init__(self, x, y, transform=None, target_transform=None):
        assert len(x) == len(y)
        self.data = x
        self.label = y
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, item):
        img = self.data[item]
        if self.transform is not None:
            img = self.transform(img)
        target = self.label[item]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        ll = len(self.label)
        assert len(self.data) == ll
        return ll


class BaseLoader:
    def __init__(self, args):
        self.n_dl = args['n_dl']
        if 'batch_train' in args:
            self.batch_train = args['batch_train']
        if 'batch_attack' in args:
            self.batch_attack = args['batch_attack']
        if 'attack_on_train' in args:
            self.attack_on_train = args['attack_on_train']
        if 'test_path' in args:
            self.test_path = args['test_path']
        self._ds_train = None
        self._ds_eval = None
        self._ds_attack = None
        self._ds_test = None
        self._ds_normalize = None
        self.dl_train = None
        self.dl_eval = None
        self.dl_test = None

    @property
    def ds_train(self):
        raise NotImplementedError

    @property
    def ds_eval(self):
        raise NotImplementedError

    @property
    def ds_attack(self):
        raise NotImplementedError

    @property
    def ds_test(self):
        if self._ds_test is None:
            self._ds_test = torchvision.datasets.ImageFolder(root=self.test_path, transform=transforms.ToTensor())
        return self._ds_test

    @property
    def ds_normalize(self):
        raise NotImplementedError

    @property
    def train(self):
        if self.dl_train is None:
            self.dl_train = Data.DataLoader(self.ds_train, batch_size=self.batch_train, shuffle=True,
                                            num_workers=self.n_dl, pin_memory=True)
        return self.dl_train

    @property
    def eval(self):
        if self.dl_eval is None:
            self.dl_eval = Data.DataLoader(self.ds_eval, batch_size=self.batch_train, num_workers=self.n_dl,
                                           pin_memory=True)
        return self.dl_eval

    @property
    def test(self):
        if self.dl_test is None:
            self.dl_test = Data.DataLoader(self.ds_test, batch_size=self.batch_attack, num_workers=self.n_dl,
                                           pin_memory=True)
        return self.dl_test

    def split(self, ds, n):
        temp = int(len(ds) / n)
        temp = [temp] * (n - 1) + [len(ds) - temp * (n - 1)]
        ds_split = Data.random_split(ds, temp)
        return [Data.DataLoader(ii, batch_size=self.batch_attack, num_workers=self.n_dl, pin_memory=True) for ii in
                ds_split]

    def split_attack(self, n=1):
        return self.split(self.ds_attack, n)

    def split_test(self, n=1):
        return self.split(self.ds_test, n)

    def cal_normalize(self, n=1, m=1):
        ds = self.ds_normalize
        if len(ds) % m != 0:
            raise ValueError('{}%{}={}'.format(len(ds), m, len(ds) % m))
        dl = Data.DataLoader(dataset=ds, batch_size=int(len(ds)/m), num_workers=self.n_dl, pin_memory=True)
        mean = np.zeros(3, np.float32)
        std = np.zeros(3, np.float32)
        for _ in range(n):
            for x, y in dl:
                x = x.cuda(non_blocking=True)
                for ii in range(3):
                    mean[ii] += x[:, ii, :, :].mean().item()
                    std[ii] += x[:, ii, :, :].std().item()
        print('n: ', n)
        print('mean: ', mean/n/m)
        print('std: ', std/n/m)

    @staticmethod
    def get_v_base(bdir, tots, n):
        classes = os.listdir(bdir)
        classes.sort()
        imgs = []
        for ii in classes:
            tp = join(bdir, ii)
            for fn in np.random.choice(os.listdir(tp), n, False):
                ppp = join(tp, fn)
                imgs.append(tots(Image.open(ppp)))
        return classes, torch.stack(imgs)


class MNist(BaseLoader):
    mean = [0.1307, 0.1307, 0.1307]
    std = [0.3081, 0.3081, 0.3081]
    num_classes = 10

    def __init__(self, args):
        super().__init__(args)
        self.mnist_dir = args['mnist_dir']

    @property
    def ds_train(self):
        if self._ds_train is None:
            self._ds_train = torchvision.datasets.MNIST(
                root=self.mnist_dir,
                train=True,
                transform=transforms.Compose([transforms.ToTensor(),
                                              lambda x: torch.cat((x, x, x), 0)])
            )
        return self._ds_train

    @property
    def ds_eval(self):
        if self._ds_eval is None:
            self._ds_eval = torchvision.datasets.MNIST(
                root=self.mnist_dir,
                train=False,
                transform=transforms.Compose([transforms.ToTensor(),
                                              lambda x: torch.cat((x, x, x), 0)])
            )
        return self._ds_eval

    @property
    def ds_attack(self):
        if self._ds_attack is None:
            if self.attack_on_train:
                ds = os.path.join(self.mnist_dir, 'train')
            else:
                ds = os.path.join(self.mnist_dir, 'eval')
            c = 0
            if os.path.exists(ds):
                classes = os.listdir(ds)
                for ii in classes:
                    p = os.path.join(ds, ii)
                    c += len(os.listdir(p))
            ods = None
            if self.attack_on_train:
                if c != 60000:
                    ods = torchvision.datasets.MNIST(self.mnist_dir)
            else:
                if c != 10000:
                    ods = torchvision.datasets.MNIST(self.mnist_dir, train=False)
            if ods is not None:
                if os.path.exists(ds):
                    shutil.rmtree(ds)
                classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
                for ii in classes:
                    path = os.path.join(ds, ii)
                    os.makedirs(path)
                cid = 0
                for x, y in ods:
                    x.save(os.path.join(ds, classes[y], str(cid) + '.bmp'))
                    cid += 1
            self._ds_attack = AttackImageFolder(
                ds,
                transforms.Compose([transforms.ToTensor(), lambda x: torch.cat((x, x, x), 0)]))
        return self._ds_attack

    @property
    def ds_normalize(self):
        if self._ds_normalize is None:
            self._ds_normalize = torchvision.datasets.MNIST(
                root=self.mnist_dir,
                transform=transforms.Compose([transforms.ToTensor(),
                                             lambda x: torch.cat((x, x, x), 0)])
            )
        return self._ds_normalize

    def get_v(self, n):
        bdir = join(self.mnist_dir, 'eval')
        tots = transforms.ToTensor()
        return self.get_v_base(bdir, tots, n)


class Cifar10(BaseLoader):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    num_classes = 10

    def __init__(self, args):
        super().__init__(args)
        self.cifar10_dir = args['cifar10_dir']

    @property
    def ds_train(self):
        if self._ds_train is None:
            self._ds_train = torchvision.datasets.CIFAR10(
                root=self.cifar10_dir,
                train=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(32, 4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                ])
            )
        return self._ds_train

    @property
    def ds_eval(self):
        if self._ds_eval is None:
            self._ds_eval = torchvision.datasets.CIFAR10(
                root=self.cifar10_dir,
                train=False,
                transform=transforms.ToTensor()
            )
        return self._ds_eval

    @property
    def ds_attack(self):
        if self._ds_attack is None:
            if self.attack_on_train:
                ds = os.path.join(self.cifar10_dir, 'train')
            else:
                ds = os.path.join(self.cifar10_dir, 'eval')
            c = 0
            if os.path.exists(ds):
                classes = os.listdir(ds)
                for ii in classes:
                    p = os.path.join(ds, ii)
                    c += len(os.listdir(p))
            ods = None
            if self.attack_on_train:
                if c != 50000:
                    ods = torchvision.datasets.CIFAR10(self.cifar10_dir)
            else:
                if c != 10000:
                    ods = torchvision.datasets.CIFAR10(self.cifar10_dir, train=False)
            if ods is not None:
                if os.path.exists(ds):
                    shutil.rmtree(ds)
                classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
                for ii in classes:
                    path = os.path.join(ds, ii)
                    os.makedirs(path)
                cid = 0
                for x, y in ods:
                    x.save(os.path.join(ds, classes[y], str(cid) + '.bmp'))
                    cid += 1
            self._ds_attack = AttackImageFolder(ds, transforms.ToTensor())
        return self._ds_attack

    @property
    def ds_normalize(self):
        if self._ds_normalize is None:
            self._ds_normalize = torchvision.datasets.CIFAR10(self.cifar10_dir, transform=transforms.ToTensor())
        return self._ds_normalize

    def get_v(self, n):
        bdir = join(self.cifar10_dir, 'eval')
        tots = transforms.ToTensor()
        return self.get_v_base(bdir, tots, n)


class Voc10(BaseLoader):
    mean = [0.4594, 0.4343, 0.4039]
    std = [0.2646, 0.2621, 0.2768]
    num_classes = 10

    def __init__(self, args):
        super().__init__(args)
        self.voc10_dir = args['voc10_dir']

    @property
    def ds_train(self):
        if self._ds_train is None:
            tf = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(224), transforms.ToTensor()])
            self._ds_train = torchvision.datasets.ImageFolder(join(self.voc10_dir, 'train'), transform=tf)
        return self._ds_train

    @property
    def ds_eval(self):
        if self._ds_eval is None:
            tf = transforms.Compose([transforms.RandomCrop(224), transforms.ToTensor()])
            self._ds_eval = torchvision.datasets.ImageFolder(join(self.voc10_dir, 'val'), transform=tf)
        return self._ds_eval

    @property
    def ds_attack(self):
        if self._ds_attack is None:
            if self.attack_on_train:
                p = join(self.voc10_dir, 'train')
            else:
                p = join(self.voc10_dir, 'val')
            tf = transforms.Compose([transforms.RandomCrop(224), transforms.ToTensor()])
            self._ds_attack = AttackImageFolder(p, tf)
        return self._ds_attack

    @property
    def ds_normalize(self):
        if self._ds_normalize is None:
            tf = transforms.Compose([transforms.RandomCrop(224), transforms.ToTensor()])
            self._ds_normalize = torchvision.datasets.ImageFolder(join(self.voc10_dir, 'train'), transform=tf)
        return self._ds_normalize

    def cal_normalize(self, n=10):
        ds = self.ds_normalize
        dl = Data.DataLoader(dataset=ds, batch_size=len(ds), num_workers=self.n_dl, pin_memory=True)
        m1 = m2 = m3 = s1 = s2 = s3 = 0
        for _ in range(n):
            for x, y in dl:
                x = x.cuda(non_blocking=True)
                m1 += x[:, 0, :, :].mean().item()
                m2 += x[:, 1, :, :].mean().item()
                m3 += x[:, 2, :, :].mean().item()
                s1 += x[:, 0, :, :].std().item()
                s2 += x[:, 1, :, :].std().item()
                s3 += x[:, 2, :, :].std().item()
        print('n: ', n)
        print('mean: ', m1 / n, m2 / n, m3 / n)
        print('std: ', s1 / n, s2 / n, s3 / n)


class Skin4(BaseLoader):
    mean = [0.5899, 0.4281, 0.4477]
    std = [0.3110, 0.2524, 0.2699]
    num_classes = 4

    def __init__(self, args):
        super().__init__(args)
        self.ds_dir = args['skin4_dir']

    @property
    def ds_train(self):
        if self._ds_train is None:
            tf = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
                transforms.RandomRotation([-180, 180]),
                transforms.RandomAffine([-180, 180], translate=[0.1, 0.1], scale=[0.7, 1.3]),
                transforms.RandomCrop(224),
                transforms.ToTensor()
            ])
            self._ds_train = torchvision.datasets.ImageFolder(join(self.ds_dir, 'train'), transform=tf)
        return self._ds_train

    @property
    def ds_eval(self):
        if self._ds_eval is None:
            tf = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomCrop(224),
                transforms.ToTensor()
            ])
            self._ds_eval = torchvision.datasets.ImageFolder(join(self.ds_dir, 'val'), transform=tf)
        return self._ds_eval

    @property
    def ds_attack(self):
        if self._ds_attack is None:
            if self.attack_on_train:
                p = join(self.ds_dir, 'train')
            else:
                p = join(self.ds_dir, 'val')
            tf = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomCrop(224),
                transforms.ToTensor()
            ])
            self._ds_attack = AttackImageFolder(p, tf)
        return self._ds_attack

    @property
    def ds_normalize(self):
        if self._ds_normalize is None:
            tf = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
                transforms.RandomRotation([-180, 180]),
                transforms.RandomAffine([-180, 180], translate=[0.1, 0.1], scale=[0.7, 1.3]),
                transforms.RandomCrop(224),
                transforms.ToTensor()
            ])
            self._ds_normalize = torchvision.datasets.ImageFolder(join(self.ds_dir, 'train'), transform=tf)
        return self._ds_normalize


class Xray3(BaseLoader):
    mean = [0.4889, 0.4889, 0.4889]
    std = [0.2449, 0.2449, 0.2449]
    num_classes = 3

    def __init__(self, args):
        super().__init__(args)
        self.ds_dir = args['xray3_dir']

    @property
    def ds_train(self):
        if self._ds_train is None:
            self._ds_train = torchvision.datasets.ImageFolder(join(self.ds_dir, 'train'), transforms.ToTensor())
        return self._ds_train

    @property
    def ds_eval(self):
        if self._ds_eval is None:
            self._ds_eval = torchvision.datasets.ImageFolder(join(self.ds_dir, 'eval'), transforms.ToTensor())
        return self._ds_eval

    @property
    def ds_attack(self):
        if self._ds_attack is None:
            if self.attack_on_train:
                p = join(self.ds_dir, 'train')
            else:
                p = join(self.ds_dir, 'eval')
            self._ds_attack = AttackImageFolder(p, transforms.ToTensor())
        return self._ds_attack


class ImageNet100(BaseLoader):
    mean = [0.4788, 0.4592, 0.4154]
    std = [0.2772, 0.2708, 0.2867]
    num_classes = 100

    def __init__(self, args):
        super().__init__(args)
        self.ds_dir = args['imagenet100_dir']

    @property
    def ds_train(self):
        if self._ds_train is None:
            tf = transforms.Compose([transforms.RandomCrop(224),
                                     transforms.RandomHorizontalFlip(), transforms.ToTensor()])
            self._ds_train = torchvision.datasets.ImageFolder(join(self.ds_dir, 'train'), transform=tf)
        return self._ds_train

    @property
    def ds_eval(self):
        if self._ds_eval is None:
            tf = transforms.Compose([transforms.RandomCrop(224), transforms.ToTensor()])
            self._ds_eval = torchvision.datasets.ImageFolder(join(self.ds_dir, 'val'), transform=tf)
        return self._ds_eval

    @property
    def ds_attack(self):
        if self._ds_attack is None:
            if self.attack_on_train:
                p = join(self.ds_dir, 'train')
            else:
                p = join(self.ds_dir, 'val')
            tf = transforms.Compose([transforms.RandomCrop(224), transforms.ToTensor()])
            self._ds_attack = AttackImageFolder(p, tf)
        return self._ds_attack

    @property
    def ds_normalize(self):
        if self._ds_normalize is None:
            tf = transforms.Compose([transforms.RandomCrop(224), transforms.ToTensor()])
            self._ds_normalize = torchvision.datasets.ImageFolder(join(self.ds_dir, 'train'), transform=tf)
        return self._ds_normalize

    def cal_normalize(self, n=10):
        ds = self.ds_normalize
        m = 10
        dl = Data.DataLoader(dataset=ds, batch_size=int(len(ds)/m), num_workers=self.n_dl, pin_memory=True, shuffle=True)
        m1 = m2 = m3 = s1 = s2 = s3 = 0
        for _ in range(n):
            ii = 0
            for x, y in dl:
                x = x.cuda(non_blocking=True)
                m1 += x[:, 0, :, :].mean().item()
                m2 += x[:, 1, :, :].mean().item()
                m3 += x[:, 2, :, :].mean().item()
                s1 += x[:, 0, :, :].std().item()
                s2 += x[:, 1, :, :].std().item()
                s3 += x[:, 2, :, :].std().item()
                ii += 1
            assert ii == 10
        n = n * m
        print('n: ', n)
        print('mean: ', m1 / n, m2 / n, m3 / n)
        print('std: ', s1 / n, s2 / n, s3 / n)


'''
class Skin7(BaseLoader):
    mean = [0.591, 0.431, 0.452]
    std = [0.313, 0.254, 0.271]
    num_classes = 7

    def __init__(self, args):
        super().__init__(args)
        self.skin7_dir = args['skin7_dir']

    @property
    def ds_train(self):
        if self._ds_train is None:
            tf = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
                transforms.RandomRotation([-180, 180]),
                transforms.RandomAffine([-180, 180], translate=[0.1, 0.1], scale=[0.7, 1.3]),
                transforms.RandomCrop(224),
                transforms.ToTensor()
            ])
            self._ds_train = torchvision.datasets.ImageFolder(join(self.skin7_dir, 'train'), tf)
        return self._ds_train

    @property
    def ds_eval(self):
        if self._ds_eval is None:
            tf = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomCrop(224),
                transforms.ToTensor()
            ])
            self._ds_eval = torchvision.datasets.ImageFolder(join(self.skin7_dir, 'eval'), transform=tf)
        return self._ds_eval

    @property
    def ds_attack(self):
        if self._ds_attack is None:
            if self.attack_on_train:
                p = join(self.skin7_dir, 'train')
            else:
                p = join(self.skin7_dir, 'eval')
            tf = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomCrop(224),
                transforms.ToTensor()
            ])
            self._ds_attack = AttackImageFolder(p, transform=tf)
        return self._ds_attack

    @property
    def ds_normalize(self):
        if self._ds_normalize is None:
            tf = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomCrop(224),
                transforms.ToTensor()
            ])
            self._ds_normalize = torchvision.datasets.ImageFolder(join(self.skin7_dir, 'train'), transform=tf)
        return self._ds_normalize

    def cal_normalize(self, n=10):
        ds = self.ds_normalize
        dl = Data.DataLoader(dataset=ds, batch_size=len(ds), num_workers=self.n_dl, pin_memory=True)
        m1 = m2 = m3 = s1 = s2 = s3 = 0
        for _ in range(n):
            for x, y in dl:
                x = x.cuda(non_blocking=True)
                m1 += x[:, 0, :, :].mean().item()
                m2 += x[:, 1, :, :].mean().item()
                m3 += x[:, 2, :, :].mean().item()
                s1 += x[:, 0, :, :].std().item()
                s2 += x[:, 1, :, :].std().item()
                s3 += x[:, 2, :, :].std().item()
        print('n: ', n)
        print('mean: ', m1/n, m2/n, m3/n)
        print('std: ', s1/n, s2/n, s3/n)

    def get_v(self, n):
        bdir = join(self.skin7_dir, 'eval')
        tots = transforms.Compose([transforms.Resize(224), transforms.RandomCrop(224), transforms.ToTensor()])
        return self.get_v_base(bdir, tots, n)

'''


def get_dataloader(args):
    if args['dsid'] == 'mnist':
        return MNist(args)
    elif args['dsid'] == 'cifar10':
        return Cifar10(args)
    elif args['dsid'] == 'voc10':
        return Voc10(args)
    elif args['dsid'] == 'skin4':
        return Skin4(args)
    elif args['dsid'] == 'xray3':
        return Xray3(args)
    elif args['dsid'] == 'imagenet100':
        return ImageNet100(args)
    else:
        raise ValueError('No dataset: {}'.format(args['dlid']))


if __name__ == '__main__':
    pass
