import cv2
import time
import torch
import pickle
import argparse
import datetime
from pathlib import Path
from utils.losses import calc_loss
from torch.utils.data import DataLoader
from models.HRCenterNet import _HRCenterNet
from utils.utility import load_setting, get_score
from datasets.HanDataset import chinese_dataset, augment_dataset, chinese_collate

import pytorch_lightning as pl

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
torch.cuda.manual_seed(42)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", "-s", type=str, default="settings/default.yaml")
    parser.add_argument("--test", "-t", default=False, action='store_true')
    parser.add_argument("--resume", "-r", default=False, action='store_true')
    parser.add_argument("--weight_fn", "-w", type=str, default="")
    parser.add_argument("--num_workers", "-nw", type=int, default=16)

    args = parser.parse_args()

    cfg = load_setting(args.setting)
    Path(cfg.weight_path).mkdir(exist_ok=True, parents=True)
    if not args.test:
        train_dataset = chinese_dataset(cfg, cfg.train_data_list_file)
        train_collate = chinese_collate(cfg)
        #aug_dataset = augment_dataset('/data/2022/ocr/augment_goseo', train_dataset.get_token())
        #train_dataloader = DataLoader(train_dataset+aug_dataset, pin_memory=True, shuffle = True, batch_size = cfg.batch_size, collate_fn = train_collate, num_workers=args.num_workers)
        train_dataloader = DataLoader(train_dataset, pin_memory=True, shuffle = True, batch_size = cfg.batch_size, collate_fn = train_collate, num_workers=args.num_workers)
        cfg.n_classes = len(train_dataset.get_token()['id2token'])
        print('Training data load Done!')

        cfg.inference = True
        cfg.crop_ratio = 0
        valid_dataset = chinese_dataset(cfg, cfg.valid_data_list_file)
        valid_collate = chinese_collate(cfg)
        valid_dataloader = DataLoader(valid_dataset, shuffle = False, batch_size = cfg.batch_size, collate_fn = valid_collate, num_workers=args.num_workers)
        print('Valid data load Done!')
        model = _HRCenterNet(cfg)
        if args.resume:
            if cfg.weight_fn == "":
                assert "Please check pretrained model path"
            model.load_state_dict(torch.load(cfg.weight_fn, map_location=device))
        model = model.to(device)
        print('Model load Done!')

        train(cfg, model, train_dataloader, valid_dataloader)

    else:
        cfg.inference = True
        cfg.crop_ratio = 0
        test_dataset = chinese_dataset(cfg, cfg.test_data_list_file)
        test_collate = chinese_collate(cfg)
        test_dataloader = DataLoader(test_dataset, shuffle = False, batch_size = cfg.batch_size, collate_fn = test_collate, num_workers=args.num_workers)
        if args.weight_fn == "":
            assert "Please check pretrained model path"
        cfg.n_classes = len(test_dataset.get_token()['id2token'])
        model = _HRCenterNet(cfg)
        model.load_state_dict(torch.load(args.weight_fn, map_location=device))
        model = model.to(device)

        evaluate(cfg, model, test_dataloader)


def train(cfg, model, train_dataloader, valid_dataloader):
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor='valid_f1',
        dirpath=f'{cfg.weight_path}',
        filename='{epoch:02d}-{valid_f1:.3f}',
        save_top_k=3,
        mode="max"
    )
    strategy = pl.strategies.DDPStrategy(find_unused_parameters=False)
    trainer = pl.Trainer(accelerator='gpu', devices=torch.cuda.device_count(), max_epochs=cfg.epochs, num_sanity_val_steps=1,
                         callbacks=[ckpt_callback, lr_callback],
                         strategy=strategy if torch.cuda.device_count() > 1 else None,
                         )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

@torch.no_grad()
def evaluate(cfg, model, dataloader):
    model.eval()
    precisions, recalls, f1_s = [], [], []
    print('\nStart Valid Process')
    for batch_idx, (imgs, labels, sizes, file_names) in enumerate(dataloader):
        inputs = imgs.to(device)
        outputs, class_preds = model(inputs)
        for i in range(inputs.shape[0]):
            label = torch.tensor(labels[i]).to(device)
            if len(label) == 0:
                print(label)
                continue
            precision, recall, f1 = get_score(outputs[i], class_preds[i], label, sizes[i], \
                                            cfg.output_size, nms_score=0.3, iou_threshold=0.1, iou_criterion=0.8)
            '''
            # print label, pred
            img = imgs[i].permute(1,2,0).cpu().numpy()*255
            width, height = sizes[i]
            img = cv2.resize(img, (width,height))
            for boxes in gt_boxes.cpu().tolist():
                lx, ly, rx, ry = map(int,boxes)
                #cx, cy, w, h = map(int,[cx,cy,w,h])
                #lx, ly, rx, ry = cx - w //2, cy - h // 2, cx + w // 2, cy + h // 2
                img = cv2.rectangle(img, (lx,ly), (rx,ry), (255,0,0), 3)
            for boxes in pred_boxes.cpu().tolist():
                lx, ly, rx, ry = map(int,boxes)
                img = cv2.rectangle(img, (lx,ly), (rx,ry), (0,0,255), 3)
            cv2.imwrite(f'{i}.png', img)
            '''
            precisions.append(precision)
            recalls.append(recall)
            f1_s.append(f1)
        print(f'\rValid Progress : {batch_idx}/{len(dataloader)}', end='')
    try:
        avg_precision = sum(precisions)/len(precisions)
        avg_recall = sum(recalls)/len(recalls)
        avg_f1 = sum(f1_s)/len(f1_s)
    except TypeError:
        print('precision',precisions)
        print('recall',recalls)
        print('f1',f1_s)
    print(f'\nprecision : {avg_precision:.4f} \n recall : {avg_recall:.4f} \n f1 : {avg_f1:.4f}')
    return avg_precision, avg_recall, avg_f1

if __name__ == "__main__":
    main()
