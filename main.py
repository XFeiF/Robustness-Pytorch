from src import config
from src import tool
from src import agent

if __name__ == '__main__':
    config.init_path_config(__file__)
    pblog = tool.get_pblog(total=config.epoch)
    try:
        pblog.debug('###start###')
        if config.cmd == 'temp':
            pass
        elif config.cmd == 'train':                             # train
            args = {'cuda': config.cuda,                        # cuda
                    'n_dl': config.n_dl,                        # dataloader num_workers
                    'dsid': config.dsid,                        # dataset id
                    'mnist_dir': config.mnist_dir,              # mnist dir
                    'cifar10_dir': config.cifar10_dir,          # cifar10 dir
                    'voc10_dir': config.voc10_dir,              # voc dir
                    'skin4_dir': config.skin4_dir,              # skin4 dir
                    'xray3_dir': config.xray3_dir,              # xray3 dir
                    'imagenet100_dir': config.imagenet100_dir,  # imagenet100 dir
                    'mid': config.mid,                          # model id
                    'acid': config.acid,                        # action id
                    'epoch': config.epoch,                      # epoch
                    'pretrain': config.pretrain,                # use pretrain or not
                    'batch_train': config.batch_train,          # train batch size
                    'no_eval': config.no_eval,                  # eval or not
                    'kk': config.kk,                            # hyper-parameters default='3,0,0.2,1,10', 
                                                                # help='noise rate一张原始图像生成几张带噪图像, noise type(0 uniform, 1 normal), noise level, noise weight, ae weight...'
                                                                #  (最后两个weight不一定，取决于具体用法，在只有noise时，k3表示noise weight，k4表示取用第几个block的feature计算)
                    'nsize': config.nsize,                      # noise size
                    'model_dir': config.model_dir,              # folder to save model
                    'best_model_dir': config.best_model_dir,    # folder to save best model
                    'tbx_dir': config.tbx_dir}                  # folder to save results
            agent.Trainer(args).train()
        elif config.cmd == 'attack':                            # attack
            args = {'cuda': config.cuda,
                    'n_dl': config.n_dl,
                    'dsid': config.dsid,
                    'cifar10_dir': config.cifar10_dir,
                    'voc10_dir': config.voc10_dir,
                    'xray3_dir': config.xray3_dir,
                    'skin4_dir': config.skin4_dir,
                    'imagenet100_dir': config.imagenet100_dir,
                    'mid': config.mid,
                    'midtf': config.midtf,
                    'atkid': config.atkid,                      # attack id
                    'attack_on_train': config.attack_on_train,  # attack on train or not
                    'batch_attack': config.batch_attack,
                    'nsize': config.nsize,
                    'reattack': config.reattack,
                    'kk': config.kk,
                    'best_model_dir': config.best_model_dir,
                    'adversarial_dir': config.adversarial_dir
                    }
            agent.Adversary(args).attack()
        elif config.cmd == 'test':                              # test
            args = {'cuda': config.cuda,
                    'n_dl': config.n_dl,
                    'dsid': config.dsid,
                    'cifar10_dir': config.cifar10_dir,
                    'voc10_dir': config.voc10_dir,
                    'xray3_dir': config.xray3_dir,
                    'skin4_dir': config.skin4_dir,
                    'imagenet100_dir': config.imagenet100_dir,
                    'mid': config.mid,
                    'midtf': config.midtf,
                    'testidtf': config.testidtf,
                    'batch_attack': config.batch_attack,
                    'best_model_dir': config.best_model_dir,
                    'adversarial_dir': config.adversarial_dir}
            agent.Tester(args).test()
        elif config.cmd == 'v':                                 # feature map -> vector test
            args = {'cuda': config.cuda,
                    'n_dl': config.n_dl,
                    'dsid': config.dsid,
                    'cifar10_dir': config.cifar10_dir,
                    'skin7_dir': config.skin7_dir,
                    'mid': config.mid,
                    'midtf': config.midtf,
                    'best_model_dir': config.best_model_dir,
                    'v_dir': config.v_dir}
            agent.Vfeat(args).gen_v()
        elif config.cmd == 'v2':
            args = {'cuda': config.cuda,
                    'n_dl': config.n_dl,
                    'dsid': config.dsid,
                    'cifar10_dir': config.cifar10_dir,
                    'skin7_dir': config.skin7_dir,
                    'mid': config.mid,
                    'midtf': config.midtf,
                    'best_model_dir': config.best_model_dir,
                    'v_dir': config.v_dir}
            agent.Vfeat(args).gen_v2()
        else:
            raise ValueError('No cmd: {}'.format(config.cmd))
    except:
        pblog.exception('Exception Logged')
        exit(1)
    else:
        pblog.debug('###ok###')
