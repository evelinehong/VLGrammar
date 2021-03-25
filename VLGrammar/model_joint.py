import time
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch_struct import SentCFG

import utils
from module import CompoundCFG, VisCompoundCFG, TextEncoder, ImageEncoder2, Similarity, ContrastiveLoss2

class VGCPCFGs(object):
    NS_IMG_PARSER = 'img_parser'
    NS_TXT_PARSER = 'txt_parser'
    NS_IMG_ENCODER = 'img_enc' 
    NS_TXT_ENCODER = 'txt_enc'
    NS_OPTIMIZER = 'optimizer'

    def __init__(self, opt, vocab, logger):
        self.niter = 0
        self.vocab = vocab
        self.log_step = opt.log_step
        self.grad_clip = opt.grad_clip
        self.vse_lm_alpha = opt.vse_lm_alpha
        self.vse_mt_alpha = opt.vse_mt_alpha
        
        self.img_parser = VisCompoundCFG(
            opt.img_states, opt.vis_nt_states, opt.vis_t_states, 
            h_dim = opt.h_dim,
            w_dim = opt.w_dim,
            z_dim = opt.z_dim,
            s_dim = opt.state_dim
        )

        self.txt_parser = CompoundCFG(
            opt.vocab_size, opt.nt_states, opt.t_states, 
            h_dim = opt.h_dim,
            w_dim = opt.w_dim,
            z_dim = opt.z_dim,
            s_dim = opt.state_dim
        )

        word_emb = torch.nn.Embedding(len(vocab), opt.word_dim)
        torch.nn.init.xavier_uniform_(word_emb.weight)

        self.txt_enc = TextEncoder(opt, word_emb)
        self.img_enc = ImageEncoder2(opt)

        self.all_params = [] 
        self.all_params += list(self.img_parser.parameters())
        self.all_params += list(self.txt_parser.parameters())
        self.all_params += list(self.img_enc.parameters())
        self.all_params += list(self.txt_enc.parameters())

        self.optimizer = torch.optim.Adam(
            self.all_params, lr=opt.lr, betas=(opt.beta1, opt.beta2)
        ) 

        self.similarity = Similarity()
        self.contrastive = ContrastiveLoss2()

        if torch.cuda.is_available():
            cudnn.benchmark = False 
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.img_parser.cuda()
            self.txt_parser.cuda()

        logger.info(self.img_parser)
        logger.info(self.txt_parser)

    def train(self):
        self.img_enc.train()
        self.txt_enc.train()
        self.img_parser.train()
        self.txt_parser.train()

    def eval(self):
        self.img_enc.eval()
        self.txt_enc.eval()
        self.img_parser.eval()
        self.txt_parser.eval()

    def get_state_dict(self):
        state_dict = { 
            self.NS_IMG_PARSER: self.img_parser.state_dict(), 
            self.NS_TXT_PARSER: self.txt_parser.state_dict(),
            self.NS_IMG_ENCODER: self.img_enc.state_dict(), 
            self.NS_TXT_ENCODER: self.txt_enc.state_dict(),
            self.NS_OPTIMIZER: self.optimizer.state_dict(),
        } 
        return state_dict

    def set_state_dict(self, state_dict):
        self.img_parser.load_state_dict(state_dict[self.NS_IMG_PARSER])
        self.txt_parser.load_state_dict(state_dict[self.NS_TXT_PARSER])
        self.img_enc.load_state_dict(state_dict[self.NS_IMG_ENCODER])
        self.txt_enc.load_state_dict(state_dict[self.NS_TXT_ENCODER])
        self.optimizer.load_state_dict(state_dict[self.NS_OPTIMIZER])

    def norms(self):
        p_norm = sum([p.norm() ** 2 for p in self.all_params]).item() ** 0.5
        g_norm = sum([p.grad.norm() ** 2 for p in self.all_params if p.grad is not None]).item() ** 0.5
        return p_norm, g_norm

    def forward_img_parser(self, input, lengths):
        params, kl, features = self.img_parser(input)
        dist = SentCFG(params, lengths=lengths)

        the_spans = dist.argmax[-1]
        argmax_spans, trees, lprobs = utils.extract_parses(the_spans, lengths.tolist(), inc=0) 

        img_outputs = self.img_enc(features, lengths)

        ll, span_margs = dist.inside_im
        nll = -ll
        kl = torch.zeros_like(nll) if kl is None else kl
        return img_outputs, nll, kl, span_margs, argmax_spans, trees, lprobs

    def forward_txt_parser(self, input, lengths):
        params, kl = self.txt_parser(input)
        dist = SentCFG(params, lengths=lengths)

        the_spans = dist.argmax[-1]
        argmax_spans, trees, lprobs = utils.extract_parses(the_spans, lengths.tolist(), inc=0) 

        txt_outputs = self.txt_enc(input, lengths)

        ll, span_margs = dist.inside_im
        nll = -ll
        kl = torch.zeros_like(nll) if kl is None else kl
        return txt_outputs, nll, kl, span_margs, argmax_spans, trees, lprobs

    def forward_loss(self, img_span_features, cap_span_features, img_lengths, txt_lengths, img_span_bounds, txt_span_bounds, img_span_margs, txt_span_margs):
        b = img_span_features.size(0)
        N_txt = txt_lengths.max(0)[0]
        mstep_txt = (txt_lengths * 2).int()
        # focus on only short spans
        nstep_txt = int(mstep_txt.float().mean().item())

        N_img = img_lengths.max(0)[0]
        mstep_img = (img_lengths * 2).int()
        # focus on only short spans
        nstep_img = int(mstep_img.float().mean().item())

        matching_loss_matrix = torch.zeros(
            b, nstep_img, nstep_txt, device=img_span_features.device
        )
        similarity_matrix = torch.zeros(
            b, b, nstep_img, nstep_txt, device=img_span_features.device
        )

        for j in range(nstep_img):
            for k in range(nstep_txt):
                cap_emb = cap_span_features[:, k] 
                img_emb = img_span_features[:, j]
                cap_marg = txt_span_margs[:, k].softmax(-1).unsqueeze(-2)
                cap_emb = torch.matmul(cap_marg, cap_emb).squeeze(-2)

                img_marg = img_span_margs[:, j].softmax(-1).unsqueeze(-2)
                img_emb = torch.matmul(img_marg, img_emb).squeeze(-2)

                cap_emb = utils.l2norm(cap_emb) 
                img_emb = utils.l2norm(img_emb)
                similarity_matrix[:, :, j, k] = self.similarity(img_emb, cap_emb) 

        img_span_margs = img_span_margs.sum(-1).unsqueeze(2).unsqueeze(1)
        txt_span_margs = txt_span_margs.sum(-1).unsqueeze(1).unsqueeze(0)

        expected_similarity = img_span_margs[:, :, :nstep_img, :] * txt_span_margs[:, :, :, :nstep_txt] * similarity_matrix
        expected_similarity = expected_similarity.sum([-2,-1])
        
        expected_loss = self.contrastive(expected_similarity)
        return expected_loss

    def forward(self, images, captions, lengths, img_lengths, img_txts, img_spans, txt_spans, labels, ids=None, epoch=None, *args):
        self.niter += 1
        self.logger.update('Eit', self.niter)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        img_lengths = torch.tensor(img_lengths).long() if isinstance(img_lengths, list) else img_lengths
        lengths = torch.tensor(lengths).long() if isinstance(lengths, list) else lengths

        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            lengths = lengths.cuda()
            img_lengths = img_lengths.cuda()
        bsize = captions.size(0) 

        img_emb, nll_img, kl_img, span_margs_img, argmax_spans_img,  trees_img, lprobs_img = self.forward_img_parser(images, img_lengths)

        ll_loss_img = nll_img.sum()
        kl_loss_img = kl_img.sum()
        
        txt_emb, nll_txt, kl_txt, span_margs_txt, argmax_spans_txt,  trees_txt, lprobs_txt = self.forward_txt_parser(captions, lengths)

        ll_loss_txt = nll_txt.sum()
        kl_loss_txt = kl_txt.sum()
        
        contrastive_loss = self.forward_loss(img_emb, txt_emb, img_lengths, lengths, argmax_spans_img, argmax_spans_txt, span_margs_img, span_margs_txt)
        mt_loss = contrastive_loss.sum()

        loss_img = self.vse_lm_alpha * (ll_loss_img + kl_loss_img) / bsize
        loss_txt = self.vse_lm_alpha * (ll_loss_txt + kl_loss_txt) / bsize 
        loss_mt = self.vse_mt_alpha * mt_loss / bsize
        
        loss = loss_img + loss_txt + loss_mt

        self.optimizer.zero_grad()
        loss.backward()

        if self.grad_clip > 0:
            clip_grad_norm_(self.all_params, self.grad_clip)
        self.optimizer.step()
       
        self.logger.update('Loss_img', loss_img.item(), bsize)
        self.logger.update('Loss_txt', loss_txt.item(), bsize)
        self.logger.update('KL-Loss_img', kl_loss_img.item() / bsize, bsize)
        self.logger.update('KL-Loss_txt', kl_loss_txt.item() / bsize, bsize)
        self.logger.update('LL-Loss_img', ll_loss_img.item() / bsize, bsize)
        self.logger.update('LL-Loss_txt', ll_loss_txt.item() / bsize, bsize)

        self.n_word_img += (img_lengths + 1).sum().item()
        self.n_word_txt += (lengths + 1).sum().item()
        self.n_sent += bsize

        for b in range(bsize):
            max_img_len = img_lengths[b].item() 
            pred_img = [(a[0], a[1]) for a in argmax_spans_img[b] if a[0] != a[1]]
            pred_set_img = set(pred_img[:-1])
            gold_img = [(img_spans[b][i][0].item(), img_spans[b][i][1].item()) for i in range(max_img_len - 1)] 
            gold_set_img = set(gold_img[:-1])
            utils.update_stats(pred_set_img, [gold_set_img], self.all_stats_img) 

            max_txt_len = lengths[b].item() 
            pred_txt = [(a[0], a[1]) for a in argmax_spans_txt[b] if a[0] != a[1]]
            pred_set_txt = set(pred_txt[:-1])
            gold_txt = [(txt_spans[b][i][0].item(), txt_spans[b][i][1].item()) for i in range(max_txt_len - 1)] 
            gold_set_txt = set(gold_txt[:-1])
            utils.update_stats(pred_set_txt, [gold_set_txt], self.all_stats_txt)

        # if self.niter % self.log_step == 0:
        p_norm, g_norm = self.norms()
        all_f1_img = utils.get_f1(self.all_stats_img)
        all_f1_txt = utils.get_f1(self.all_stats_txt)
        train_kl_img = self.logger.meters["KL-Loss_img"].sum 
        train_ll_img = self.logger.meters["LL-Loss_img"].sum 
        train_kl_txt = self.logger.meters["KL-Loss_txt"].sum 
        train_ll_txt = self.logger.meters["LL-Loss_txt"].sum 

        info = '|Pnorm|: {:.6f}, |Gnorm|: {:.2f}, ReconPPL-Img: {:.2f}, KL-Img: {:.2f}, ' + \
                'PPLBound-Img: {:.2f}, CorpusF1-Img: {:.2f}, ' + \
                'ReconPPL-Txt: {:.2f}, KL-Txt: {:.2f}, ' + \
                'PPLBound-Txt: {:.2f}, CorpusF1-Txt: {:.2f}, ' + \
                'Speed: {:.2f} sents/sec'

        info = info.format(
            p_norm, g_norm, np.exp(train_ll_img / self.n_word_img), train_kl_img / self.n_sent,
            np.exp((train_ll_img + train_kl_img) / self.n_word_img), all_f1_img[0], 

            np.exp(train_ll_txt / self.n_word_txt), train_kl_txt / self.n_sent,
            np.exp((train_ll_txt + train_kl_txt) / self.n_word_txt), all_f1_txt[0], 
            self.n_sent / (time.time() - self.s_time)
        )

        pred_action_img = utils.get_actions(trees_img[0])
        sent_s_img = img_txts[0]
        pred_t_img = utils.get_tree(pred_action_img, sent_s_img)
        gold_t_img = utils.span_to_tree(img_spans[0].tolist(), img_lengths[0].item()) 
        gold_action_img = utils.get_actions(gold_t_img) 
        gold_t_img = utils.get_tree(gold_action_img, sent_s_img)
        info += "\nPred T Image: {}\nGold T Image: {}".format(pred_t_img, gold_t_img)

        pred_action_txt = utils.get_actions(trees_txt[0])
        sent_s_txt = [self.vocab.idx2word[wid] for wid in captions[0].cpu().tolist()]
        pred_t_txt = utils.get_tree(pred_action_txt, sent_s_txt)
        gold_t_txt = utils.span_to_tree(txt_spans[0].tolist(), lengths[0].item()) 
        gold_action_txt = utils.get_actions(gold_t_txt) 
        gold_t_txt = utils.get_tree(gold_action_txt, sent_s_txt)
        info += "\nPred T Text: {}\nGold T Text: {}".format(pred_t_txt, gold_t_txt)
        return info
