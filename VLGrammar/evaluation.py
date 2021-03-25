import time
import numpy as np
from collections import OrderedDict
from sklearn import metrics
import torch
import utils

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)

class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

def validate_parser(opt, data_loader, model, vocab, logger):
    batch_time = AverageMeter()
    val_logger = LogCollector()

    model.eval()
    end = time.time()
    nbatch = len(data_loader)

    n_word_img, n_word_txt, n_sent = 0, 0, 0
    sent_f1_img, sent_f1_txt, corpus_f1_img, corpus_f1_txt = [], [], [0., 0., 0.], [0., 0., 0.] 
    total_ll_img, total_kl_img, total_ll_txt, total_kl_txt = 0., 0., 0., 0.

    # clustering_targets = []
    # clustering_preds = []
    # clustering = model.img_parser.clustering

    for i, (images, captions, lengths, img_lengths, image_texts, img_spans, txt_spans, labels, ids) in enumerate(data_loader):
        model.logger = val_logger
        img_lengths = torch.tensor(img_lengths).long() if isinstance(img_lengths, list) else img_lengths
        lengths = torch.tensor(lengths).long() if isinstance(lengths, list) else lengths
        if torch.cuda.is_available():
            img_lengths = img_lengths.cuda()
            images = images.cuda()
            captions = captions.cuda()
            lengths = lengths.cuda()
        bsize = captions.size(0) 
        
        with torch.no_grad():
            txt_emb, nll_txt, kl_txt, span_margs_txt, argmax_spans_txt,  trees_txt, lprobs_txt = model.forward_txt_parser(captions, lengths)    
            img_emb, nll_img, kl_img, span_margs_img, argmax_spans_img,  trees_img, lprobs_img = model.forward_img_parser(images, img_lengths)
            contrastive_loss = model.forward_loss(img_emb, txt_emb, img_lengths, lengths, argmax_spans_img, argmax_spans_txt, span_margs_img, span_margs_txt)            

            # b,n = images.shape[0], images.shape[1]
            # images = images.reshape(b*n, images.shape[-3], images.shape[-2], images.shape[-1])
            # probs = clustering(images)[1]
            # probs = probs.reshape(b, n, -1)
            # preds = torch.argmax(probs, -1)
            # preds_flat = []
            # for i in range (0, b):
            #     preds_flat += preds[i, :img_lengths[i]].tolist()
            # targets = []
            # for label in labels:
            #     for l in label:
            #         targets.append(l)

            # clustering_targets += targets
            # clustering_preds += preds_flat

        batch_time.update(time.time() - end)
        end = time.time()

        total_ll_img += nll_img.sum().item()
        total_ll_txt += nll_txt.sum().item()
        total_kl_img += kl_img.sum().item()
        total_kl_txt += kl_txt.sum().item()
        n_word_img += (img_lengths + 1).sum().item()
        n_word_txt += (lengths + 1).sum().item()
        n_sent += bsize

        for b in range(bsize):
            max_img_len = img_lengths[b].item() 
            pred_img = [(a[0], a[1]) for a in argmax_spans_img[b] if a[0] != a[1]]
            pred_set_img = set(pred_img[:-1])
            gold_img = [(img_spans[b][i][0].item(), img_spans[b][i][1].item()) for i in range(max_img_len - 1)] 
            gold_set_img = set(gold_img[:-1])

            max_txt_len = lengths[b].item() 
            pred_txt = [(a[0], a[1]) for a in argmax_spans_txt[b] if a[0] != a[1]]
            pred_set_txt = set(pred_txt[:-1])
            gold_txt = [(txt_spans[b][i][0].item(), txt_spans[b][i][1].item()) for i in range(max_txt_len - 1)] 
            gold_set_txt = set(gold_txt[:-1])

            tp_img, fp_img, fn_img = utils.get_stats(pred_set_img, gold_set_img) 
            corpus_f1_img[0] += tp_img
            corpus_f1_img[1] += fp_img
            corpus_f1_img[2] += fn_img
            
            overlap_img = pred_set_img.intersection(gold_set_img)
            prec_img = float(len(overlap_img)) / (len(pred_set_img) + 1e-8)
            reca_img = float(len(overlap_img)) / (len(gold_set_img) + 1e-8)
            
            if len(gold_set_img) == 0:
                reca_img = 1. 
                if len(pred_set_img) == 0:
                    prec_img = 1.
            f1_img = 2 * prec_img * reca_img / (prec_img + reca_img + 1e-8)
            sent_f1_img.append(f1_img)

            tp_txt, fp_txt, fn_txt = utils.get_stats(pred_set_txt, gold_set_txt) 
            corpus_f1_txt[0] += tp_txt
            corpus_f1_txt[1] += fp_txt
            corpus_f1_txt[2] += fn_txt
            
            overlap_txt = pred_set_txt.intersection(gold_set_txt)
            prec_txt = float(len(overlap_txt)) / (len(pred_set_txt) + 1e-8)
            reca_txt = float(len(overlap_txt)) / (len(gold_set_txt) + 1e-8)

            if len(gold_set_txt) == 0:
                reca_txt = 1. 
                if len(pred_set_txt) == 0:
                    prec_txt = 1.
            f1_txt = 2 * prec_txt * reca_txt / (prec_txt + reca_txt + 1e-8)
            sent_f1_txt.append(f1_txt)

        if i % model.log_step == 0:
            logger.info(
                'Test: [{0}/{1}]\t{e_log}\t'
                .format(
                    i, nbatch, e_log=str(model.logger)
                )
            )
        del captions, lengths, images, txt_spans, img_lengths, ids, img_spans
        #if i > 10: break


    # #class_names = ['tabletop', 'drawer', 'cabinet_door', 'side_panel', 'bottom_panel', 'leg', 'leg_bar', 'central_support', 'pedestal', 'shelf']
    # class_names = ['chair_head', 'back_surface', 'back_frame_vertical_bar', 'back_frame_horizontal_bar',  'chair_seat', 'chair_arm', 'arm_sofa_style', 'arm_near_vertical_bar','arm_horizontal_bar', 'central_support', 'leg', 'leg_bar', 'pedestal']
    # label_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12]

    # targets = torch.as_tensor(clustering_targets).cuda()
    # predictions = torch.as_tensor(clustering_preds).cuda()
    
    # num_elems = targets.size(0)
    # match = utils._hungarian_match(predictions, targets, 13, 13)

    # reordered_preds = torch.zeros(num_elems).cuda()
    # for pred_i, target_i in match:
    #     reordered_preds[predictions == int(pred_i)] = int(target_i)

    # # Gather performance metrics
    # acc = int((reordered_preds == targets).sum()) / float(num_elems)
    # nmi = metrics.normalized_mutual_info_score(targets.cpu().numpy(), predictions.cpu().numpy())
    # ari = metrics.adjusted_rand_score(targets.cpu().numpy(), predictions.cpu().numpy())

    # report = metrics.classification_report(targets.cpu().numpy(), reordered_preds.cpu().numpy(), labels=label_ids, target_names=class_names)
    # print ("Here's clustering results! ")
    # print(report)

    # print ('ACC: {:.4f}, ARI: {:.2f}, NMI: {:.2f}'.format(acc, ari, nmi))

    # del reordered_preds
    # del targets
    # del predictions

    tp_img, fp_img, fn_img = corpus_f1_img  
    prec_img = tp_img / (tp_img + fp_img)
    recall_img = tp_img / (tp_img + fn_img)
    corpus_f1_img = 2 * prec_img * recall_img / (prec_img + recall_img) if prec_img + recall_img > 0 else 0.
    sent_f1_img = np.mean(np.array(sent_f1_img))
    recon_ppl_img = np.exp(total_ll_img / n_word_img)
    ppl_elbo_img = np.exp((total_ll_img + total_kl_img) / n_word_img) 
    kl_img = total_kl_img / n_sent

    tp_txt, fp_txt, fn_txt = corpus_f1_txt  
    prec_txt = tp_txt / (tp_txt + fp_txt)
    recall_txt = tp_txt / (tp_txt + fn_txt)
    corpus_f1_txt = 2 * prec_txt * recall_txt / (prec_txt + recall_txt) if prec_txt + recall_txt > 0 else 0.
    sent_f1_txt = np.mean(np.array(sent_f1_txt))
    recon_ppl_txt = np.exp(total_ll_txt / n_word_txt)
    ppl_elbo_txt = np.exp((total_ll_txt + total_kl_txt) / n_word_txt) 
    kl_txt = total_kl_txt / n_sent

    info = 'Corpus F1 Image: {:.2f}, Sentence F1 Image: {:.2f}' + \
        'Corpus F1 Text: {:.2f}, Sentence F1 Text: {:.2f}'
    info = info.format(
        corpus_f1_img * 100, sent_f1_img * 100,
        corpus_f1_txt * 100, sent_f1_txt * 100,
    )
    logger.info(info)
    return corpus_f1_img, corpus_f1_txt
