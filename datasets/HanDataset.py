import json
import torch
import pickle
import torchvision
from PIL import Image, ImageFile
from pathlib import Path
from easydict import EasyDict
from unicodedata import normalize
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

class augment_dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, token_id_dict):
        self.data_dir = Path(data_dir)
        self.label_file = self.data_dir / 'label.pkl'
        with self.label_file.open('rb') as f:
            self.label_dict = pickle.load(f)
        self.label_id = list(self.label_dict.keys())
        self.token_id_dict = token_id_dict

    def __len__(self):
        return len(self.label_id)

    def __getitem__(self, idx):
        image, gt = self.load_data(self.label_id[idx])

        return image, gt, self.label_id[idx]
    
    def load_data(self, file_name):
        img_path = self.data_dir / (file_name + '.png')

        gt = []

        for coord in self.label_dict[file_name]:
            label, lx, ly, w, h = coord
            label = normalize('NFKC', label.decode('utf-8')).encode('utf-8')
            
            if lx < 0 or ly < 0 or lx >= (lx+w) or ly >= (ly+h) or w < 10 or h < 10:
                continue
            if label not in self.token_id_dict['token2id']:
                _id = self.token_id_dict['token2id']['[OOV]']
            else:
                _id = self.token_id_dict['token2id'][label]

            gt.append([_id, lx + w//2, ly + h // 2, w, h])

        if len(gt) == 0:
            return None, None

        image = Image.open(img_path)

        return image, gt

class chinese_dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, data_dir, token_id_dict=None):
        super(chinese_dataset, self).__init__()
        self.cfg = EasyDict(cfg.copy())
        self.img_data_dir = Path(self.cfg.img_data_dir)
        self.json_data_dir = Path(self.cfg.json_data_dir)

        with open(data_dir, 'r') as f:
            self.file_names = f.readlines()
        try:
            with open(self.cfg.token_path, 'rb') as f:
                self.token_id_dict = pickle.load(f)
            if cfg.inference:
                print("Load Token Done")
        except FileNotFoundError:
            if cfg.inference:
                assert "Please check token.pkl file path"
            self.token_id_dict = self.make_token()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        image, gt = self.load_data(self.file_names[idx].strip())

        return image, gt, self.file_names[idx].strip()

    def load_data(self, file_name):
        json_path = self.json_data_dir / (file_name + '.json')
        img_path = self.img_data_dir / (file_name + '.jpg')

        with json_path.open('r') as f:
            data = json.load(f)

        gt = []
        for coords in data['Image_Text_Coord']:
            for coord in coords:
                lx, ly, w, h, _, _ = map(int, coord['bbox'])
                label = coord['label']
                label = normalize('NFKC', label).encode('utf-8')
                if self.cfg.inference:
                    if label not in self.token_id_dict['token2id']:
                        _id = self.token_id_dict['token2id']['[OOV]']
                    else:
                        _id = self.token_id_dict['token2id'][label]
                else:
                    if label not in self.token_id_dict['token2id']:
                        _id = len(self.token_id_dict['token2id'])
                        self.token_id_dict['token2id'][label] = _id
                        self.token_id_dict['id2token'][_id] = label
                    else:
                        _id = self.token_id_dict['token2id'][label]
                gt.append([_id, lx + w//2, ly + h // 2, w, h])
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            img_path = img_path.with_suffix('.JPG')
            image = Image.open(img_path).convert('RGB')
        return image, gt

    def get_token(self):
        return self.token_id_dict
        
    def make_token(self):
        
        token_id_dict = {
            'token2id': {
                '[BG]' : 0,
                '[OOV]': 1 
            },
            'id2token': {
                0 : '[BG]',
                1 : '[OOV]'
            }
        }
        for idx, file_name in enumerate(self.file_names):
            json_path = self.json_data_dir / (file_name.strip() + '.json')
            with json_path.open('r') as f:
                data = json.load(f)
            for coords in data['Image_Text_Coord']:
                for coord in coords:
                    label = coord['label']
                    label = normalize('NFKC', label).encode('utf-8')
                    if label not in token_id_dict['token2id']:
                        _id = len(token_id_dict['token2id'])
                        token_id_dict['token2id'][label] = _id
                        token_id_dict['id2token'][_id] = label
            print(f'\r Parsing Progress {idx} / {len(self.file_names)}', end='')
        with Path(self.cfg.token_path).open('wb') as f:
            pickle.dump(token_id_dict, f)
        print("Save Token Done! total token is "+str(len(token_id_dict['token2id'])))

        return token_id_dict
    

class chinese_collate(object):
    def __init__(self, cfg=None):
        if cfg is None:
            assert "Please Input setting information"
        self.cfg = EasyDict(cfg.copy())
        self.image_transform = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5,],[0.5,])]) 

    def __call__(self, batch):
        output_width, output_height = self.cfg.output_size
        crop_size = self.cfg.crop_size
        crop_ratio = self.cfg.crop_ratio
        batch_imgs, batch_labels, batch_imgs_size = [], [], []
        file_names = []
        # Train Process
        if not self.cfg.inference:
            for img, annotations,file_name in batch:
                if img is None or annotations is None:
                    continue
                file_names.append(file_name)
                pic_width, pic_height = img.size
                label = np.zeros((output_height, output_width, 7))      
        
                if pic_height < crop_size or pic_width < crop_size:
                    img = torchvision.transforms.functional.resize(img, (crop_size, crop_size))
        
                if np.random.randint(0, 101) < crop_ratio * 100 and pic_height > crop_size and pic_width > crop_size:
                    new_h, new_w = (crop_size, crop_size)
                    top = np.random.randint(0, pic_height - new_h)
                    left = np.random.randint(0, pic_width - new_w)
                    img = img.crop((left, top, left + new_w, top + new_h))
                else:
                    new_w, new_h = img.size
                    top = 0
                    left = 0
                    img = torchvision.transforms.functional.resize(img, (crop_size, crop_size))
            
                for annotation in annotations:

                    if annotation[1] < left or annotation[2] < top or annotation[1] >= (left + new_w) or annotation[2] >= (top + new_h):
                        continue
                
                    # ignore the character that exceed to much to the boundary
                    if (annotation[1] + (annotation[3] / 2)) >= (left + new_w) or (annotation[2] + (annotation[4] / 2)) >= (top + new_h): 
                        continue
            
                    x_c = (annotation[1] - left) * (output_width / new_w)
                    y_c = (annotation[2] - top) * (output_height / new_h)
                    width = annotation[3] * (output_width / new_w)
                    height = annotation[4] * (output_height / new_h)
                    if width / 10 <= 0 or height / 10 <= 0:
                        continue
                    heatmap=((np.exp(-(((np.arange(output_width) - x_c)/(width/10))**2)/2)).reshape(1,-1)
                            *(np.exp(-(((np.arange(output_height) - y_c)/(height/10))**2)/2)).reshape(-1,1))
                    
                    label[:,:,0] = np.maximum(label[:,:,0], heatmap[:,:])
                    label[int(y_c//1), int(x_c//1), 1] = 1
                    label[int(y_c//1), int(x_c//1), 2] = y_c % 1
                    label[int(y_c//1), int(x_c//1), 3] = x_c % 1
                    label[int(y_c//1), int(x_c//1), 4] = height / output_height 
                    label[int(y_c//1), int(x_c//1), 5] = width / output_width
                    label[int(y_c//1), int(x_c//1), 6] = annotation[0]
            
                if self.image_transform:
                    img = self.image_transform(img)

                batch_imgs.append(img)
                batch_labels.append(label)

            return torch.from_numpy(np.stack(batch_imgs)), torch.from_numpy(np.stack(batch_labels)), file_names

        # Valid or Test Process
        else:
            file_names = []
            for img, annotations, file_name in batch:
                pic_width, pic_height = img.size
                img = torchvision.transforms.functional.resize(img, (crop_size, crop_size))
                if self.image_transform:
                    img = self.image_transform(img)
                batch_imgs.append(img)
                batch_labels.append(annotations)
                batch_imgs_size.append((pic_width, pic_height))
                file_names.append(file_name)

            return torch.from_numpy(np.stack(batch_imgs)), batch_labels, batch_imgs_size, file_names
