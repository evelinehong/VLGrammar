import os 
import time, pickle, argparse, logging
import numpy as np
import torch
from tqdm import tqdm
import data
import random
from utils import Vocabulary, save_checkpoint
from evaluation import AverageMeter, LogCollector, validate_parser 

def train(opt, train_loader, model, epoch, vocab, val_loader):
    # average meters to record the training statistics
    train_logger = LogCollector()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    nbatch = len(train_loader)
    # switch to train mode
    end = time.time()
    model.n_word_img = 0
    model.n_word_txt = 0
    model.n_sent = 0
    model.s_time = end
    model.all_stats_img = [[0., 0., 0.]]
    model.all_stats_txt = [[0., 0., 0.]]
    for train_data in tqdm(train_loader):
        # Always reset to train mode
        model.train()
        # measure data loading time
        data_time.update(time.time() - end)
        # make sure train logger is used
        model.logger = train_logger
        # Update the model
        info = model.forward(*train_data, epoch=epoch)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # Print log info
        if model.niter % opt.log_step == 0:
            logger.info(
                'Epoch: [{0}] {e_log} {info}'
                .format(
                    epoch,  e_log=str(model.logger), info=info
                )
            )
        #break
        # validate at every val_step
        # if model.niter % opt.val_step == 0:
    # validate_parser(opt, val_loader, model, vocab, logger, opt.mode)

if __name__ == '__main__':
    # hyper parameters
    parser = argparse.ArgumentParser()

    # Parser: Generative model parameters
    parser.add_argument('--z_dim', default=64, type=int, help='latent dimension')
    parser.add_argument('--t_states', default=30, type=int, help='number of preterminal states')
    parser.add_argument('--vis_t_states', default=13, type=int)
    parser.add_argument('--nt_states', default=20, type=int, help='number of nonterminal states')
    parser.add_argument('--vis_nt_states', default=10, type=int)
    parser.add_argument('--img_states', default=30, type=int)
    parser.add_argument('--state_dim', default=256, type=int, help='symbol embedding dimension')
    # Parser: Inference network parameters
    parser.add_argument('--h_dim', default=512, type=int, help='hidden dim for variational LSTM')
    parser.add_argument('--w_dim', default=512, type=int, help='embedding dim for variational LSTM')
    parser.add_argument('--gpu', default=1, type=int, help='which gpu to use')

    # 
    parser.add_argument('--seed', default=66, type=int, help='random seed')
    parser.add_argument('--vocab_name', default="partnet.dict.pkl", type=str, help='vocab name')
    parser.add_argument('--prefix', default="", type=str, help='prefix')
    parser.add_argument('--parser_type', default='2nd', type=str, help='model name (1st/2nd)')
    parser.add_argument('--share_w2vec', default=False, type=bool, help='shared embeddings')
    #
    parser.add_argument('--sem_dim', default=128, type=int, help='semantic rep. dim')
    parser.add_argument('--word_dim', default=512, type=int,
                        help='dimensionality of the word embedding')
    parser.add_argument('--lstm_dim', default=512, type=int,
            help='dimensionality of the lstm hidden embedding')

    parser.add_argument('--data_path', default='../partit_data/',
                        help='path to datasets')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='rank loss margin')
    parser.add_argument('--num_epochs', default=100, type=int,
                        help='number of training epochs')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='size of a training mini-batch')
    parser.add_argument('--grad_clip', default=3., type=float,
                        help='gradient clipping threshold')
    parser.add_argument('--lr', default=.001, type=float,
                        help='initial learning rate')
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loader workers')
    #
    parser.add_argument('--log_step', default=500, type=int,
                        help='number of steps to print and record the log')
    parser.add_argument('--val_step', default=float("inf"), type=int,
                        help='number of steps to run validation')
    parser.add_argument('--logger_name', default='./output/chair/',
                        help='path to save the model and log')
    #
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='optimizer, can be Adam, SGD, etc.')
    parser.add_argument('--beta1', default=0.75, type=float, help='beta1 for adam')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2 for adam')
    # 
    parser.add_argument('--vse_mt_alpha', type=float, default=0.01)
    parser.add_argument('--vse_lm_alpha', type=float, default=1.0)
    parser.add_argument('--type', default='chair', type = str)

    opt = parser.parse_args()

    if opt.type == "chair":
        opt.vis_t_states = 13
        opt.data_path += '0.chair'
    if opt.type == "table":
        opt.vis_t_states = 10
        opt.data_path += '1.table'
    if opt.type == "bed":
        opt.vis_t_staates = 8
        opt.data_path += '2.bed'
    if opt.type == "bag":
        opt.vis_t_states = 3
        opt.data_path += '3.bag'
    
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    # setup logger
    if os.path.exists(opt.logger_name):
        print(f'Warning: the folder {opt.logger_name} exists.')
    else:
        print('Creating {}'.format(opt.logger_name))
        os.mkdir(opt.logger_name)
        
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(os.path.join(opt.logger_name, 'train.log'), 'w')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.propagate = False
    logger.info('cuda:{}@{}'.format(opt.gpu, os.uname().nodename))
    logger.info(opt)

    # load predefined vocabulary and pretrained word embeddings if applicable
    vocab = pickle.load(open(os.path.join(opt.data_path, opt.vocab_name), 'rb'))
    opt.vocab_size = len(vocab)

    logger.info("|vocab|={}".format(len(vocab)))

    # construct the model

    from model_joint import VGCPCFGs as Model

    sampler = True
    model = Model(opt, vocab, logger)

    # Load data loaders
    train_loader = data.get_eval_iter(
        opt.data_path, opt.prefix + "train", vocab, opt.batch_size, 
        nworker=opt.workers, shuffle=False, type=opt.type, sampler=sampler
    )
    val_loader = data.get_eval_iter(
        opt.data_path, opt.prefix + "test", vocab, opt.batch_size, 
        nworker=opt.workers, shuffle=False, type=opt.type, sampler=sampler
    )

    data.set_rnd_seed(66)

    best_f1 = -float('inf') 
    # model.txt_parser.load_state_dict(torch.load("../checkpoints/language/best.pth.tar"))
    # model.img_parser.load_state_dict(torch.load('../checkpoints/vision/chair.pth.tar'))
    model.img_parser.clustering.load_state_dict(torch.load('../checkpoints/clustering/model.pth.tar_chair')['model'])
    # print ("Successfully load pretrained model")
    validate_parser(opt, val_loader, model, vocab, logger)
    for epoch in range(opt.num_epochs):
        # train for one epoch
        train(opt, train_loader, model, epoch, vocab, val_loader)
        # evaluate on validation set using VSE metrics
        f1_img, f1_txt = validate_parser(opt, val_loader, model, vocab, logger)
        #break
        # remember best R@ sum and save checkpoint
        f1 = f1_img + f1_txt
        is_best = f1 > best_f1
        best_f1 = max(f1, best_f1)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.get_state_dict(),
            'opt': opt,
            'Eiters': model.niter,
        }, is_best, epoch, prefix=opt.logger_name + '/')
