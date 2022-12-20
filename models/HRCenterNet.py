import torch
from torch import nn
import pytorch_lightning as pl
from utils.losses import calc_loss
from utils import utility as utils
from models.modules import BasicBlock, Bottleneck, init_weights

'''
def calc_attention(attention_list, feature_list):

    out = [ x + attn(x) for attn, x in zip(attention_list, feature_list) ]

    return out

class selfAttention(nn.Module):
    def __init__(self, feature_size):
        super(selfAttention, self).__init__()
        self.d_k = feature_size
        # self.W_Q = nn.Linear(feature_size, feature_size)
        # self.W_K = nn.Linear(feature_size, feature_size)
        # self.W_V = nn.Linear(feature_size, feature_size)
        self.W_Q = nn.Linear(feature_size**2, feature_size**2)
        self.W_K = nn.Linear(feature_size**2, feature_size**2)
        self.W_V = nn.Linear(feature_size**2, feature_size**2)

    def forward(self, x):
        b, c, width, height = x.shape
        x = x.contiguous().flatten(2, 3)
        # x = x.permute(0,2,1)
        Q = self.W_Q(x) / self.d_k
        K = self.W_K(x) / self.d_k
        V = self.W_V(x) / self.d_k
        # Q = self.W_Q(x) / (self.d_k ** 0.5)
        # K = self.W_K(x) / (self.d_k ** 0.5)
        # V = self.W_V(x) / (self.d_k ** 0.5)

        QK = torch.matmul(Q, K.transpose(-2,-1))
        QK = torch.clip(QK, -65504, 65504)
        QK = torch.nn.functional.softmax(QK, dim=2)
        # out = torch.matmul(QK, V)
        out = torch.bmm(QK, V)

        return out.contiguous().view(b, c, width, height)
'''

class StageModule(nn.Module):
    def __init__(self, stage, output_branches, c, bn_momentum):
        super(StageModule, self).__init__()
        self.stage = stage
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.stage):
            w = c * (2 ** i)
            branch = nn.Sequential(
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()
        # for each output_branches (i.e. each branch in all cases but the very last one)
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.stage):  # for each branch
                if i == j:
                    self.fuse_layers[-1].append(nn.Sequential())  # Used in place of "None" because it is callable
                elif i < j:
                    self.fuse_layers[-1].append(nn.Sequential(
                        nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=(1, 1), stride=(1, 1), bias=False),
                        nn.BatchNorm2d(c * (2 ** i), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Upsample(scale_factor=(2.0 ** (j - i)), mode='nearest'),
                    ))
                elif i > j:
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                      bias=False),
                            nn.BatchNorm2d(c * (2 ** j), eps=1e-05, momentum=0.1, affine=True,
                                           track_running_stats=True),
                            nn.ReLU(inplace=True),
                        ))
                    ops.append(nn.Sequential(
                        nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                  bias=False),
                        nn.BatchNorm2d(c * (2 ** i), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    ))
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        assert len(self.branches) == len(x)

        x = [branch(b) for branch, b in zip(self.branches, x)]

        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(0, len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    x_fused[i] = x_fused[i] + self.fuse_layers[i][j](x[j])

        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])

        return x_fused

class _HRCenterNet(pl.LightningModule):
    def __init__(self, cfg):
        super(_HRCenterNet, self).__init__()
        self.cfg = cfg
        c = self.cfg.channel
        nof_joints = self.cfg.nof_joints
        bn_momentum = self.cfg.bn_momentum
        n_classes = self.cfg.n_classes

        # self.attn1 = selfAttention(128)
        # self.attn2 = selfAttention(64)
        # self.attn3 = selfAttention(32)
        # self.attn4 = selfAttention(16)

        # self.attn1 = selfAttention(32)
        # self.attn2 = selfAttention(64)
        # self.attn3 = selfAttention(128)
        # self.attn4 = selfAttention(256)

        # Input (stem net)
        self.conv1 = nn.Conv2d(self.cfg.input_channel, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1 (layer1)      - First group of bottleneck (resnet) modules
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
        )

        # Fusion layer 1 (transition1)      - Creation of the first two branches (one full and one half resolution)
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv2d(256, c * (2 ** 1), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c * (2 ** 1), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),
        ])

        # Stage 2 (stage2)      - Second module with 1 group of bottleneck (resnet) modules. This has 2 branches
        self.stage2 = nn.Sequential(
            StageModule(stage=2, output_branches=2, c=c, bn_momentum=bn_momentum),
        )

        # Fusion layer 2 (transition2)      - Creation of the third branch (1/4 resolution)
        self.transition2 = nn.ModuleList([
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv2d(c * (2 ** 1), c * (2 ** 2), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c * (2 ** 2), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),  # ToDo Why the new branch derives from the "upper" branch only?
        ])

        # Stage 3 (stage3)      - Third module with 4 groups of bottleneck (resnet) modules. This has 3 branches
        self.stage3 = nn.Sequential(
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
        )

        # Fusion layer 3 (transition3)      - Creation of the fourth branch (1/8 resolution)
        self.transition3 = nn.ModuleList([
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv2d(c * (2 ** 2), c * (2 ** 3), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c * (2 ** 3), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),  # ToDo Why the new branch derives from the "upper" branch only?
        ])

        # Stage 4 (stage4)      - Fourth module with 3 groups of bottleneck (resnet) modules. This has 4 branches
        self.stage4 = nn.Sequential(
            StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum),
            StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum),
            StageModule(stage=4, output_branches=1, c=c, bn_momentum=bn_momentum),
        )

        # Final layer (final_layer)
        self.final_layer = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(32, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, nof_joints, kernel_size=(1, 1), stride=(1, 1), bias=False),
            #nn.Sigmoid()
        )
        # self.sigmoid = nn.Sigmoid()

        self.classifier = nn.Sequential(
            # nn.Conv2d(c, 32, kernel_size=(1,1), stride=(1,1)),
            # nn.BatchNorm2d(32, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(32, n_classes, kernel_size=(1,1), stride=(1,1))
            ################
            nn.Conv2d(c, 1024, kernel_size=(1,1), stride=(1,1)),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, n_classes, kernel_size=(1,1), stride=(1,1), bias=False)
        )

        if not cfg.inference:
            init_weights([self.conv1, self.bn1, self.conv2, self.bn2])
            init_weights(self.layer1.modules())
            init_weights(self.transition1.modules())
            init_weights(self.stage2.modules()) 
            init_weights(self.transition2.modules())
            init_weights(self.stage3.modules()) 
            init_weights(self.transition3.modules())
            init_weights(self.stage4.modules()) 
            init_weights(self.final_layer.modules()) 
            init_weights(self.classifier.modules()) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]  # Since now, x is a list (# == nof branches)
        # x = calc_attention([self.attn1, self.attn2], x)
        x = self.stage2(x)
        # x = [trans(x[-1]) for trans in self.transition2]    # New branch derives from the "upper" branch only
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]  # New branch derives from the "upper" branch only
        # x = calc_attention([self.attn1, self.attn2, self.attn3], x)

        x = self.stage3(x)
        # x = [trans(x) for trans in self.transition3]    # New branch derives from the "upper" branch only
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1])
        ]  # New branch derives from the "upper" branch only
        # x = calc_attention([self.attn1, self.attn2, self.attn3, self.attn4], x)

        final_x = self.stage4(x)
        # final_x = calc_attention([self.attn1], x)

        x = self.final_layer(final_x[0])
        x = torch.clamp(x, 0.0, 1.0)
        # bbox_x = self.sigmoid(x[:,:5,:,:])
        # class_x = final_x[0].permute(0,2,3,1)
        # class_x = class_x.contiguous().flatten(1,2)
        # b, c, h, w = final_x[0].shape
        # class_x = class_x.contiguous().view(b,h,w,-1).permute(0,3,1,2)

        # class_x = x[:,5:,:,:]
        class_x = self.classifier(final_x[0])
        return x, class_x

    def training_step(self, train_batch, batch_idx):
        imgs, labels, file_names = train_batch
        outputs, class_preds = self.forward(imgs)
        metrics = {}

        loss, class_loss = calc_loss(outputs, class_preds, labels, metrics)
        self.log('train_loss', loss, sync_dist=True)

        return {'loss':loss, 'heatmap_loss':metrics['heatmap'], 'size_loss':metrics['size'], 'offset_loss':metrics['offset'],
                'class_loss':metrics['class']}

    def validation_step(self, val_batch, batch_idx):
        imgs, labels, sizes, file_names = val_batch
        outputs, class_preds = self.forward(imgs)
        precisions, recalls, f1_s = [], [], []
        for i in range(imgs.shape[0]):
            label = torch.tensor(labels[i]).to(imgs.device)
            if len(label) == 0:
                continue
            precision, recall, f1 = utils.get_score(outputs[i], class_preds[i], label, sizes[i], \
                                                    self.cfg.output_size, nms_score=0.3, iou_threshold=0.1, iou_criterion=0.8)
            precisions.append(precision)
            recalls.append(recall)
            f1_s.append(f1)

        return {'precision':precisions, 'recall':recalls, 'f1':f1_s}

    def validation_step_end(self, outputs):
        precision = torch.tensor(outputs['precision']).mean()
        recall = torch.tensor(outputs['recall']).mean()
        f1 = torch.tensor(outputs['f1']).mean()

        return {'precision':precision, 'recall':recall, 'f1':f1}

    def validation_epoch_end(self, outputs):
        avg_precision = sum([x['precision'] for x in outputs]) / len(outputs)
        avg_recall = sum([x['recall'] for x in outputs]) / len(outputs)
        avg_f1 = sum([x['f1'] for x in outputs]) / len(outputs)
        self.log('valid_f1',avg_f1, sync_dist=True)
        self.log('valid_precision', avg_precision, sync_dist=True)
        self.log('valid_recall', avg_recall, sync_dist=True)
        print(f'valid_f1 : {avg_f1:.4f}\nvalid_precision : {avg_precision:.4f}\nvalid_recall : {avg_recall:.4f}')

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.cfg.optimizer)
        optimizer = optimizer(self.parameters(), lr=self.cfg.lr)

        if not hasattr(self.cfg, "scheduler") or not self.cfg.scheduler:
            return optimizer
        elif hasattr(torch.optim.lr_scheduler, self.cfg.scheduler):
            scheduler = getattr(torch.optim.lr_scheduler, self.cfg.scheduler)
        elif hasattr(utils, self.cfg.scheduler):
            scheduler = getattr(utils, self.cfg.scheduler)
        else:
            raise ModuleNotFoundError

        scheduler = {
                'scheduler': scheduler(optimizer, **self.cfg.scheduler_param),
                'interval': self.cfg.scheduler_interval,
                'name': "learning rate"
            }

        return [optimizer], [scheduler]
