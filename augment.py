from pathlib import Path
from PIL import Image
import json
import pickle
import cv2
import numpy as np
import random
from unicodedata import normalize
from tqdm import tqdm
import multiprocessing as mp

base = Path('/data/2022/ocr/1.원천데이터')
json_folder = Path('/data/2022/ocr/2.라벨링데이터')
augment_folder = Path('/data/2022/ocr/augment_goseo')

augment_folder.mkdir(exist_ok=True, parents=True)

def get_background(background):
    origin_img = cv2.imread(str(background))
    img_y,img_x,_ = np.where(origin_img!=255)
    lx,ly,rx,ry = min(img_x), min(img_y), max(img_x), max(img_y)
    img = origin_img[ly:ry, lx:rx, :].copy()
    mask = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    json_path = json_folder / background.parts[-3] / background.parts[-2] / (background.stem +'.json')
    with json_path.open('r') as f:
        data = json.load(f)

    for coords in data['Image_Text_Coord']:
        for coord in coords:
            x,y,w,h,_,_ = map(int, coord['bbox'])
            x, y = x-lx, y-ly
            if x < 0 or y < 0:
                continue
            img[y:y+h,x:x+w,:] = 0
            mask[y:y+h,x:x+w,:] = 255

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    background_img = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)

    background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2GRAY)
    if background.stem.split('_')[0] != 'ACKS':
        _, background_img = cv2.threshold(
                            background_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    return origin_img, background_img, lx, ly, rx, ry

def augment(aug_dict, rank):
    random.seed(rank)
    for d in base.glob('*'):
        if d.name not in aug_dict:
            continue
        aug_count = aug_dict[d.name]
        pbar = tqdm(list(d.glob('*')))
        for p in pbar:
            pbar.set_description(f"{rank}, {d.name}")
            file_list = sorted(list(p.glob('*.jpg')))
            file_list = file_list[1:]
            random.shuffle(file_list)
            for i in range(aug_count):
                if i % 20 == 0:
                    background = random.choice(file_list)
                    origin_img, background_img, lx, ly, rx, ry = get_background(background)
                    height, width = background_img.shape

                boolean_mask = np.zeros((height,width))

                output_img = origin_img.copy()
                augment_img = background_img.copy()
                out_bbox = []

                sample_img_path = random.choice(file_list)
                sample_img = cv2.imread(str(sample_img_path))

                sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
                if background.stem.split('_')[0] != 'ACKS':
                    _, sample_img = cv2.threshold(
                                        sample_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )
                json_path = json_folder / background.parts[-3] / background.parts[-2] / (sample_img_path.stem + '.json')
                with json_path.open('r') as f:
                    data = json.load(f)
                for coords in data['Image_Text_Coord']:
                    for coord in coords:
                        x,y,w,h,_,_ = map(int, coord['bbox'])
                        label = coord['label']
                        label = normalize('NFKC', label).encode('utf-8')
                        if x < 0 or y < 0:
                            continue
                        cropped_img = sample_img[y:y+h,x:x+w]
                        w = w + random.randint(-10,10) if w > 15 else w
                        h = h + random.randint(-10,10) if h > 15 else h
                        cropped_img = cv2.resize(cropped_img,(w,h))
                        # height_flag, width_flag = False, False
                        assign_flag = False
                        count = 0
                        while not assign_flag and count < 200:
                            new_height = random.randint(0, abs(height-h))
                            new_width = random.randint(0, abs(width-w))
                            if np.sum(boolean_mask[new_height:new_height+h, new_width:new_width+w]) == 0 and \
                                    new_height+h < height and new_width+w < width:
                                assign_flag = True
                                boolean_mask[new_height:new_height+h, new_width:new_width+w] = 1
                            count += 1
                        if assign_flag:
                            out_bbox.append([label, new_width, new_height, w, h])
                            augment_img[new_height:new_height+h, new_width:new_width+w] = cropped_img.copy()

                augment_img = cv2.cvtColor(augment_img, cv2.COLOR_GRAY2BGR)
                out_name = f'{background.stem}_{i}'
                if out_name in out_dict:
                    out_name = f'{background.stem}_{i+1000}_{rank}'
                out_img = augment_folder / f'{out_name}.png'
                output_img[ly:ry,lx:rx,:] = augment_img
                cv2.imwrite(str(out_img),output_img)
                out_dict[out_name] = out_bbox


if __name__ == '__main__':
    num_process = mp.cpu_count()

    total_aug_dict = {
        '해서' : 200,
        '행서' : 250,
        '초서' : 300
    }
    out_dict = mp.Manager().dict()

    process = []
    for rank in range(num_process):
        aug_dict = {
            '해서' : total_aug_dict['해서'] // num_process,
            '행서' : total_aug_dict['행서'] // num_process,
            '초서' : total_aug_dict['초서'] // num_process
        }
        p = mp.Process(target=augment, args=(aug_dict, rank,))
        p.start()
        process.append(p)
    
    for t in process:
        t.join()
    
    with (augment_folder / 'label.pkl').open('wb') as f:
        pickle.dump(out_dict.copy(), f)
