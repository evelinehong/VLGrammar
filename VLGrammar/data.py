import os, re, json
import numpy as np
import random
import torch
import torch.utils.data as data
import ast
from torchvision import transforms
import PIL.Image as Image
from PIL import ImageOps

IMG_TRANSFORM = transforms.Compose([
                    transforms.Resize(256),
                    transforms.ToTensor(),
                    transforms.Normalize([0], [1])
                ])

def set_rnd_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class SortedBlockSampler(data.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        all_sample = len(self.data_source)
        batch_size = data_source.batch_size
        nblock = all_sample // batch_size 
        residue = all_sample % batch_size
        nsample = all_sample - residue
        # https://numpy.org/doc/stable/reference/generated/numpy.array_split.html
        # it returns l % n sub-arrays of size l//n + 1 and the rest of size l//n.
        self.groups = np.array_split(range(nsample), nblock)
        self.strip_last = False
        if residue > 0:
            self.strip_last = True
            block = np.array(range(nsample, all_sample))
            self.groups.append(block)

    def __iter__(self):
        self.data_source._shuffle()
        end = -1 if self.strip_last else len(self.groups)
        groups = self.groups[:end]
        #random.shuffle(groups) 
        indice = torch.randperm(len(groups)).tolist() 
        groups = [groups[k] for k in indice]
        if self.strip_last:
            groups.append(self.groups[-1])
        indice = list()
        for i, group in enumerate(groups):
            indice.extend(group)
            #print(i, group)
        assert len(indice) == len(self.data_source)
        return iter(indice)

    def __len__(self):
        return len(self.data_source)

class SortedRandomSampler(data.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        self.data_source._shuffle()
        return iter(torch.randperm(len(self.data_source)).tolist())

    def __len__(self):
        return len(self.data_source)

class SortedSequentialSampler(data.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        self.data_source._shuffle()
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)

class DataLoader(data.Dataset):
    def __init__(self, data_path, data_split, vocab, 
                 type='chair', batch_size=1):
        self.batch_size = batch_size
        self.vocab = vocab
        self.captions = []
        self.image_texts = []
        self.image_spans = []
        self.spans = []
        self.images = []
        self.targets = []
        self.ids = []

        self.dirs = next(os.walk(os.path.join(data_path, data_split)))[1]
        self.dirs.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        if type == 'chair':
            self.parts = {'chair_head': 0, 'back_surface': 1, 'back_frame_vertical_bar': 2, 'back_frame_horizontal_bar': 3,  'chair_seat': 4, 'chair_arm': 5, 'arm_sofa_style': 6, 'arm_near_vertical_bar': 7,'arm_horizontal_bar': 8, 'central_support': 9, 'leg': 10, 'leg_bar': 11, 'pedestal': 12}
        elif type == 'table':
            self.parts = {'tabletop': 0, 'drawer': 1, 'cabinet_door': 2, 'side_panel': 3, 'bottom_panel': 4, 'leg': 5, 'leg_bar': 6, 'central_support': 7, 'pedestal': 8, 'shelf': 9}
        elif type == 'bed':
            self.parts = {'headboard': 0, 'bed_sleep_area': 1, 'bed_frame_horizontal_surface':2, 'bed_side_surface_panel': 3, 'bed_post': 4, 'leg': 5, 'surface_base': 6, 'ladder':7}
        elif type == 'bag':
            self.parts = {'bag_body': 0, 'handle': 1, 'shoulder_strap': 2}
        
        for di in self.dirs:
            image_each = []
            target_each = []
            
            f = open (os.path.join(data_path, data_split, di, "idx2cat.json"))
            part_dict = json.load(f)

            for val, t in part_dict.items(): 
                file_name = val + "occluded.png"
                if not os.path.exists(os.path.join(data_path, data_split, di, file_name)):
                    continue
                image_each.append(os.path.join(data_path, data_split, di, file_name))
                target_each.append(self.parts[t])

            f = open (os.path.join(data_path, data_split, di, "image_text.txt"))
            image_text = f.read()
            image_text = image_text[:-1].strip().split()

            f = open (os.path.join(data_path, data_split, di, "vis_spans.txt"))
            span = f.read()
            image_span = json.loads(span)

            f = open (os.path.join(data_path, data_split, di, "utterance.txt"))
            caption = f.read()
            caption = caption.replace(".", "").replace(",", " , ").replace(":", " : ").replace(";", " ; ").replace("/", " ").replace("\'", " \'").replace("\"", " \"").strip().split()

            f = open (os.path.join(data_path, data_split, di, "lan_spans.txt"))
            span = f.read()
            span = json.loads(span)

            if len(caption) > 45:
                continue

            if len(caption) != len(span) + 1:
                continue
            else:
                self.spans.append(span)
                self.captions.append(caption)

            self.image_spans.append(image_span)
            self.image_texts.append(image_text)        

            self.images.append(image_each)
            self.targets.append(target_each)  
            self.ids.append(di)  

        for (i, image) in reversed(list(enumerate(self.images))):
            if len(image) != len(self.image_spans[i]) + 1 or len(image) > 60:
                del self.images[i]
                del self.captions[i]
                del self.image_spans[i]
                del self.image_texts[i]
                del self.spans[i]
                del self.targets[i]

        self.length = len(self.captions)

    def _shuffle(self):
        indice = torch.randperm(self.length).tolist() 
        indice = sorted(indice, key=lambda k: len(self.captions[k]))
        self.captions = [self.captions[k] for k in indice]
        self.images = [self.images[k] for k in indice]
        self.image_spans = [self.image_spans[k] for k in indice]
        self.image_texts = [self.image_texts[k] for k in indice]
        self.targets = [self.targets[k] for k in indice]
        self.spans = [self.spans[k] for k in indice]

    def __getitem__(self, index):
        # image
        #img_id = index  // self.im_div
        image = []
        for path in self.images[index]:
            img = Image.open(path).convert('L')
            img = ImageOps.invert(img)
            img = IMG_TRANSFORM(img)
            image.append(img)
        # caption
        cap = []
        for token in self.captions[index]:
            if token in self.vocab.word2idx.keys():
                cap.append(self.vocab.word2idx[token])
            else:
                cap.append(0)
        caption = torch.tensor(cap)
        image_text = self.image_texts[index]
        image_span = self.image_spans[index]
        image_span = torch.tensor(image_span)
        span = self.spans[index]
        target = self.targets[index]
        span = torch.tensor(span)
        return image, caption, image_text, image_span, span, target, index

    def __len__(self):
        return self.length

def collate_fun(data):
    # sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    zipped_data = list(zip(*data))
    images, captions, image_texts, image_spans, spans, labels, ids = zipped_data
    img_lengths = [len(x) for x in images]

    max_img_len = max ([len(x) for x in images])
    img_indices = torch.zeros(len(captions), max_img_len, 2).long()
    for i, span in enumerate(image_spans):
        img_len = len(images[i])
        img_indices[i, :img_len - 1, :] = image_spans[i]

    zero_img = torch.ones((1, 256, 256))
    collate_images = []
    for image in images:
        image += [zero_img] * (max_img_len - len(image))
        image = torch.stack(image)
        collate_images.append(image)
    images = torch.stack(collate_images)

    max_lan_len = max([len(caption) for caption in captions]) 
    indices = torch.zeros(len(captions), max_lan_len, 2).long()
    targets = torch.zeros(len(captions), max_lan_len).long()
    lengths = [len(cap) for cap in captions]

    
    for i, cap in enumerate(captions):
        cap_len = len(cap)
        targets[i, : cap_len] = cap[: cap_len]
        indices[i, : cap_len - 1, :] = spans[i]
    return images, targets, lengths, img_lengths, image_texts, img_indices, indices, labels, ids
    # return images, targets, lengths, ids, indices 

def get_data_loader(data_path, data_split, vocab, 
                    batch_size=128, 
                    shuffle=True, 
                    nworker=2, 
                    type='chair',
                    sampler=None):
    dset = DataLoader(data_path, data_split, vocab, type, batch_size)
    if sampler:
        model = SortedRandomSampler
        if not isinstance(sampler, bool) and issubclass(sampler, data.Sampler):
            model = sampler
        #sampler = SortedRandomSampler(dset)
        sampler = model(dset)
    data_loader = torch.utils.data.DataLoader(
                    dataset=dset, 
                    batch_size=batch_size, 
                    shuffle=shuffle,
                    sampler=sampler,
                    pin_memory=True, 
                    collate_fn=collate_fun
    )
    return data_loader

def get_eval_iter(data_path, split_name, vocab, batch_size, 
                  nworker=4, 
                  shuffle=False,
                  type='chair',
                  sampler=None):
    eval_loader = get_data_loader(data_path, split_name, vocab, 
                  batch_size=batch_size, 
                  shuffle=shuffle,  
                  nworker=nworker, 
                  type=type, 
                  sampler=sampler
    )
    return eval_loader
