import csv
import argparse
import logging
import os

from timm.utils import  ModelEmaV3

from models.LivNet import LivNet_Baseline, LivNet_no_moe,  LENet
from models.convnext_segmentaion import convnext_small,convnext_base
from models.focalnet import focalnet_base_lrf
from models.mpvit import mpvit_base
from utils.loss import FocalLoss

from utils.train_eval import CAM_visualization, evaluate, train_one_epoch, metric_info, visual_feature
from models.thynet import ThyNet
from torchvision.models import swin_b, Swin_B_Weights, vit_b_16, ViT_B_16_Weights, swin_s, Swin_S_Weights
from models.hiera_segmentation import Hiera
from timm.data import Mixup
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from torch import nn
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from dataset.data_loading import Liver_Dataset
from torchvision.models import resnet101, densenet201, resnext101_32x8d, ResNet101_Weights, DenseNet201_Weights, \
    ResNeXt101_32X8D_Weights
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, AUROC, F1Score ,Specificity
from models import cswin
from models.pvt_v2 import pvt_v2_b5
from models.swin_transformer_v2 import SwinTransformerV2
from models.davit import DaViT
def main(rank, k_fold, args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if args.model=='LENet':
        args.model='-'.join([args.model,args.serial_parallel,args.sparse_dense,f'{args.num_experts}experts',
                             f'top_{args.top_k}',args.head_type,*(['cat_moe_head'] if args.cat_moe_head else []),
                             'q_former_depths'+'_'.join(str(i) for i in args.q_former_depths),f'num_query_{args.num_query_tokens}',
                             f'query_dim_{args.query_dim}'
                             ])
    else:
        args.model = args.model
    seed = args.seed + k_fold
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Instantiate dataset
    if args.k_fold:
        train_df = pd.read_csv(r'/home/uax/SCY/Decouple_liver/dataset/train_{}.csv'.format(k_fold))
        valid_df = pd.read_csv(r'/home/uax/SCY/Decouple_liver/dataset/val_{}.csv'.format(k_fold))
        multi_center_df = pd.read_csv('/home/uax/SCY/Decouple_liver/dataset/multicenter_test.csv')#multi-center
        test_df = pd.read_csv(r'/home/uax/SCY/Decouple_liver/dataset/test.csv')#high-risk patient
        test3_df = pd.read_csv(r'/home/uax/SCY/Decouple_liver/dataset/test3.csv')# single-center
        train_dataset = Liver_Dataset(train_df, 'train')
        val_dataset = Liver_Dataset(valid_df, 'val')
        single_center_test_dataset = Liver_Dataset(test3_df, 'val')
        multi_center_test_dataset = Liver_Dataset(multi_center_df, 'val')
        high_risk_test_dataset = Liver_Dataset(test_df, 'val')
    else:
        raise Exception("Please use 5-fold cross validation")

    criterion_focal = FocalLoss(alpha=0.25,gamma=2)
    # criterion_focal = ClassBalancedLoss(samples_per_class=None, gamma=2, label_smoothing=args.smoothing,mixup=args.mixup)
    if args.criterion=='CrossEntropyLoss':
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    else:
        raise Exception("Please replace it with cross entropy")

    if args.weight_sampler:
        class_count = np.bincount(train_dataset['labels'])
        class_weights = 1.0 / class_count
        sample_weights = [class_weights[i] for i in train_dataset['labels'].values]
        sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights,num_samples=int(2*len(train_dataset)))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,sampler=sampler,
                                  num_workers=args.num_workers,pin_memory=True,drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,sampler=None,
                                  num_workers=args.num_workers,pin_memory=True,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    single_center_loader = DataLoader(single_center_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)
    multi_center_loader = DataLoader(multi_center_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)
    high_risk_loader = DataLoader(high_risk_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    if args.model == 'HSQformer_Baseline':
        print('HSQformer_Baseline')
        model = LivNet_Baseline()
    if args.model == 'HSQformer_no_moe':
        print('HSQformer_Baseline')
        model = LivNet_no_moe(aux_loss=args.aux_loss)

    if 'LENet' in args.model:
        print('LENet')
        model = LENet(args=args)

    if args.model == 'DaViT':
        print('DaViT')
        model=DaViT(patch_size=4, window_size=7, embed_dims=(128, 256, 512, 1024), num_heads=(4, 8, 16, 32),
            depths=(1, 1, 9, 1), mlp_ratio=4., overlapped_patch=False)
        pretrain_dict = torch.load('/home/uax/SCY/LiverClassification/Weights/DaViTmodel_best.pth.tar')
        model.load_state_dict(pretrain_dict['state_dict'])
        inchannel = model.head.in_features
        model.head = nn.Linear(in_features=inchannel, out_features=args.num_classes)
    if args.model == 'Mpvit_base':
        print('Mpvit_base')
        model = mpvit_base()
        weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/mpvit_base.pth', map_location="cpu")['model']
        model.load_state_dict(weight)
        model.cls_head.cls = nn.Linear(in_features=model.cls_head.cls.in_features,out_features=args.num_classes)
    if args.model == 'PVT_v2_b5':
        print('PVT_v2_b5')
        model = pvt_v2_b5()
        weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/pvt_v2_b5.pth', map_location="cpu")
        model.load_state_dict(weight)
        model.head = nn.Linear(in_features=model.head.in_features,out_features=args.num_classes)
    if args.model == 'FocalNet':
        print('FocalNet')
        model = focalnet_base_lrf(focal_levels=[3, 3, 3, 3],)
        weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/focalnet_base_lrf.pth', map_location="cpu")['model']
        model.load_state_dict(weight)
        model.head = nn.Linear(in_features=model.head.in_features,out_features=args.num_classes)
    if args.model == 'Resnet':
        print('Resnet')
        model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        model.fc= nn.Linear(in_features=model.fc.in_features,out_features=args.num_classes)
    if args.model == 'Densenet':
        print('Densenet')
        model = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(in_features=model.classifier.in_features, out_features=args.num_classes)
    if args.model == 'Resnext':
        print('Resnext')
        model = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=args.num_classes)
    if args.model == 'Thynet':
        print('Thynet')
        model = ThyNet(num_classes=args.num_classes)
    if args.model == 'Hiera':
        print('Hiera')
        model = Hiera(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3), drop_path_rate=0.)
        weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/hiera_base_224.pth', map_location="cpu")['model_state']
        model.load_state_dict(weight)
        model.head.projection = nn.Linear(
            in_features=model.head.projection.in_features,
            out_features=args.num_classes)
    if args.model == 'Cswin':
        print('Cswin')
        model = cswin.CSWin_96_24322_base_224( drop_path_rate=0.0)
        pretrain_dict = torch.load('/home/uax/SCY/LiverClassification/Weights/cswin_base_224.pth')
        model.load_state_dict(pretrain_dict['state_dict_ema'])
        inchannel = model.head.in_features
        model.head = nn.Linear(in_features=inchannel, out_features=args.num_classes)
    if args.model == 'Convnext':
        print('Convnext')
        model = convnext_small(pretrained=False, in_22k=False,return_feature=False)
        pretrain_dict = torch.load(r'/home/uax/SCY/LiverClassification/Weights/convnext_small_22k_224.pth',map_location="cpu")
        model.load_state_dict(pretrain_dict['model'],strict=False)
        inchannel = model.head.in_features
        model.head = nn.Linear(in_features=inchannel, out_features=args.num_classes)
    if args.model == 'Convnext_B':
        print('Convnext_B')
        model = convnext_base(pretrained=False, in_22k=False,drop_path_rate=0.,return_feature=False)
        pretrain_dict = torch.load(r'/home/uax/SCY/LiverClassification/Weights/convnext_base_22k_224.pth',map_location="cpu")
        model.load_state_dict(pretrain_dict['model'],strict=False)
        inchannel = model.head.in_features
        model.head = nn.Linear(in_features=inchannel, out_features=args.num_classes)
    if args.model == 'SwinTransformerV2':
        model = SwinTransformerV2(img_size=384, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                                  window_size=24, pretrained_window_sizes=[12, 12, 12, 6], drop_path_rate=0.2)
        pretrain_dict = torch.load(
            '/home/uax/SCY/LiverClassification/Weights/swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth')
        model.load_state_dict(pretrain_dict['model'])
        inchannel = model.head.in_features
        model.head = nn.Linear(in_features=inchannel, out_features=args.num_classes)
    if args.model == 'SwinTransformer':
        print('SwinTransformer')
        model=swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
        inchannel = model.head.in_features
        model.head = nn.Linear(in_features=inchannel, out_features=args.num_classes)
    if args.model == 'SwinTransformer_S':
        print('SwinTransformer_S')
        model=swin_s(weights=Swin_S_Weights.IMAGENET1K_V1)
        inchannel = model.head.in_features
        model.head = nn.Linear(in_features=inchannel, out_features=args.num_classes)
    if args.model == 'ViT':
        print('ViT')
        model=vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        inchannel = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features=inchannel, out_features=args.num_classes)
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(param)
    model.to(rank)
    metric_collection = MetricCollection([
        Accuracy(task=args.categories, num_classes=args.num_classes, average='macro'),
        Precision(task=args.categories, num_classes=args.num_classes, average='macro'),
        Recall(task=args.categories, num_classes=args.num_classes, average='macro'),
        Specificity(task=args.categories, num_classes=args.num_classes, average='macro'),
        F1Score(task=args.categories, num_classes=args.num_classes, average='macro'),
        AUROC(task=args.categories, num_classes=args.num_classes, average='macro',),
    ])
    if args.visual_feature:
        checkpoint = torch.load('/media/uax/CA4E64EFA9C3DA83/HCC/convnext/{}_val{}.pth'
        .format(args.model,k_fold),map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        visual_feature(model,metric_collection=metric_collection,device=rank, data_loader=val_loader)
    if args.cam_visualization:
        CAM_visualization(model, val_loader)
        print('exit')
        exit()
    if args.test:
        checkpoint = torch.load('/media/uax/CA4E64EFA9C3DA83/HCC/ablation/LENet-serial-sparse_token-4experts-top_1-linear-q_former_depths1_1_1_1_val{}.pth'
        .format(k_fold),map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        args.output_csv=f'{args.model}_multi'
        test_loss = evaluate(model=model, data_loader=single_center_loader, criterion=criterion,
                         metric_collection=metric_collection,device=rank,args=args,k_fold=k_fold)
        test_accuracy, test_precision, test_recall, test_specificity, test_f1, test_AUC = metric_collection.compute().values()
        print(
        " test_loss:{:.4f} test_accuracy:{:.4f} test_precision:{:.4f} test_recall:{:.4f} test_specificity:{:.4f} test_f1:{:.4f} test_AUC:{:.4f}" \
            .format(test_loss, test_accuracy, test_precision, test_recall,test_specificity, test_f1, test_AUC))
        args.output_csv = f'{args.model}_single'#写反了和high_risk，画图的时候注意
        test_loss = evaluate(model=model, data_loader=multi_center_loader, criterion=criterion,
                         metric_collection=metric_collection,device=rank,args=args,k_fold=k_fold)
        test_accuracy, test_precision, test_recall, test_specificity, test_f1, test_AUC = metric_collection.compute().values()
        print(
        " test_loss:{:.4f} test_accuracy:{:.4f} test_precision:{:.4f} test_recall:{:.4f} test_specificity:{:.4f} test_f1:{:.4f} test_AUC:{:.4f}" \
            .format(test_loss, test_accuracy, test_precision, test_recall,test_specificity, test_f1, test_AUC))
        args.output_csv = f'{args.model}_high_risk'
        test_loss = evaluate(model=model, data_loader=high_risk_loader, criterion=criterion,
                         metric_collection=metric_collection,device=rank,args=args,k_fold=k_fold)
        test_accuracy, test_precision, test_recall, test_specificity, test_f1, test_AUC = metric_collection.compute().values()
        print(
        " test_loss:{:.4f} test_accuracy:{:.4f} test_precision:{:.4f} test_recall:{:.4f} test_specificity:{:.4f} test_f1:{:.4f} test_AUC:{:.4f}" \
            .format(test_loss, test_accuracy, test_precision, test_recall,test_specificity, test_f1, test_AUC))
        return

    if args.model_ema:
        model_ema = ModelEmaV3(model,decay=args.model_ema_decay,
                use_warmup=args.model_ema_warmup,
                device='cpu' if args.model_ema_force_cpu else None,
            )
    else:
        model_ema=None

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "stage1" in name:
                print('freeze_layers: {}'.format(name))
                para.requires_grad = False
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                weight_decay=args.weight_decay)

    if args.lr_param_groups:

        swin_params = list(model.swintransformer.parameters())
        convnext_params = list(model.convnext.parameters())
        other_params= [p for n,p in model.named_parameters()
                             if 'swintransformer' not in n and 'convnext' not in n]
        assert len(list(model.parameters())) == len(swin_params) + len(convnext_params) + len(other_params), "学习率设置不完整"
        param_groups = [
            # self.swin 和 self.convnext 的参数
            {'params': swin_params},
            {'params': convnext_params},
            # 其他部分的参数
            {'params': other_params}
        ]

        optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=[args.lr, args.lr, args.lr*1.5 ],
                                            total_steps=args.epochs * len(train_loader), pct_start=0.05,
                                            div_factor=float('inf'), final_div_factor=1000)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,total_steps=args.epochs * len(train_loader), pct_start=0.05,
                                            div_factor=float('inf'), final_div_factor=1000)
        # optimizer = Lion(
        #     [{'params': [param for name, param in model.named_parameters() if 'backbone' in name], 'lr': args.lr},
        #      {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
        #       'lr': args.lr * 2}], weight_decay=args.weight_decay)
        #optimizer = Lion(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.apm:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    if args.mixup is True:
        mixup_fn=Mixup(mixup_alpha=0.8, cutmix_alpha=1., prob=1.,switch_prob=0.5, num_classes=args.num_classes)
    else:
        mixup_fn = None

    best_test_auc = 0.
    best_val_auc = 0.
    start_epoch = 0
    num_updates =0
    metric_collection.to('cpu')

    if args.resume:
        checkpoint = torch.load('/home/uax/SCY/Decouple_liver/checkpoints/{}_ema{}.pth'
                                .format(args.model, k_fold), map_location='cpu')

        model.load_state_dict(checkpoint['state_dict'])

    if args.test==False:
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.checkpoints, exist_ok=True)
        log_path = os.path.join(args.log_dir, '{}_training.log'.format(args.model))
        logger = logging.getLogger()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        if k_fold == 0:
            logger.addHandler(file_handler)
            logging.info("Model Name:{}".format(args.model))
            logging.info("Model Configuration:\n{}".format(model))
            for attribute, value in vars(args).items():
                logger.info(f"{attribute}: {value}")
            logging.info("param:\n{}".format(param))
    for epoch in range(start_epoch,args.epochs):
        if args.lr_param_groups:
            logging.info(f' k_fold {k_fold}  '
                         f' epoch {epoch}  '
                         f"LR backbone: {optimizer.param_groups[0]['lr']} "
                         f"LR other: {optimizer.param_groups[1]['lr']} "
                         f"LR other: {optimizer.param_groups[2]['lr']} "
                         )
        else:
            logging.info(f' k_fold {k_fold}  '
                         f' epoch {epoch}  '
                         f"LR backbone: {optimizer.param_groups[0]['lr']} "
                         )


        train_loss = train_one_epoch(model, optimizer, metric_collection,num_updates=num_updates,epoch=epoch,scheduler=scheduler,criterion_focal=criterion_focal,
                                            mixup_fn=mixup_fn,data_loader=train_loader, device=rank,ema_updata_epoch=args.ema_updata_epoch,
                                            criterion=criterion,scaler=scaler,aux_loss=args.aux_loss,model_ema=model_ema)
        train_metric=metric_info(metric_collection)
        logger.info(f"Train: loss:{train_loss:.4f}, {train_metric}")
        if epoch>args.ema_updata_epoch:
            val_loss = evaluate(model=model, data_loader=val_loader, criterion=criterion,
                                 metric_collection=metric_collection,device=rank,args=args)
            val_metric=metric_info(metric_collection)
            logger.info(f"Val: loss:{val_loss:.4f}, {val_metric}")

            test_loss = evaluate(model=model, data_loader=single_center_loader, criterion=criterion,
                                 metric_collection=metric_collection,device=rank,args=args)
            test_metric=metric_info(metric_collection)
            logger.info(f"Single_center_test: loss:{test_loss:.4f}, {test_metric}")

            test_loss = evaluate(model=model, data_loader=multi_center_loader, criterion=criterion,
                                 metric_collection=metric_collection,device=rank,args=args)
            test_metric=metric_info(metric_collection)
            logger.info(f"Multi_center_test: loss:{test_loss:.4f}, {test_metric}")

            test_loss = evaluate(model=model, data_loader=high_risk_loader, criterion=criterion,
                                 metric_collection=metric_collection,device=rank,args=args)
            test_metric=metric_info(metric_collection)
            logger.info(f"High_risk_test 3: loss:{test_loss:.4f}, {test_metric}")
            if val_metric['AUROC'] > best_val_auc:
                best_val_auc = val_metric['AUROC']
                num_updates += 1
                model_ema.update(model, step=num_updates)
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_schedule': scheduler.state_dict()}
                torch.save(checkpoint, os.path.join(args.checkpoints,
                                                    args.model + '_val{}.pth'.format(k_fold)))

                checkpoint = {
                    'epoch': epoch,
                    'state_dict': model_ema.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_schedule': scheduler.state_dict()}
                torch.save(checkpoint, os.path.join(args.checkpoints,
                                                    args.model + '_ema{}.pth'.format(k_fold)))
        else:

            val_loss = evaluate(model=model, data_loader=val_loader, criterion=criterion,
                                metric_collection=metric_collection, device=rank,args=args)
            val_metric = metric_info(metric_collection)
            logger.info(f"Val: loss:{val_loss:.4f}, {val_metric}")

            test_loss = evaluate(model=model, data_loader=single_center_loader, criterion=criterion,
                                 metric_collection=metric_collection,device=rank,args=args)
            test_metric=metric_info(metric_collection)
            logger.info(f"Single_center_test: loss:{test_loss:.4f}, {test_metric}")

            test_loss = evaluate(model=model, data_loader=multi_center_loader, criterion=criterion,
                                 metric_collection=metric_collection,device=rank,args=args)
            test_metric=metric_info(metric_collection)
            logger.info(f"Multi_center_test: loss:{test_loss:.4f}, {test_metric}")

            test_loss = evaluate(model=model, data_loader=high_risk_loader, criterion=criterion,
                                 metric_collection=metric_collection,device=rank,args=args)
            test_metric=metric_info(metric_collection)
            logger.info(f"High_risk_test 3: loss:{test_loss:.4f}, {test_metric}")

            if val_metric['AUROC'] > best_val_auc:
                best_val_auc = val_metric['AUROC']
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_schedule': scheduler.state_dict()}
                torch.save(checkpoint, os.path.join(args.checkpoints,
                                                args.model + '_val{}.pth'.format(k_fold)))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 是否启用SyncBatchNorm
    parser.add_argument('--log_dir', type=str, default='/home/uax/SCY/Decouple_liver/logs')
    parser.add_argument('--mixup', default=True,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--num_workers', type=float, default=24)

    parser.add_argument('--categories', type=str, default='binary', choices=['binary', 'multiclass', 'multilabel'])
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--optimizer', default='AdamW', help='optimizer')
    parser.add_argument('--seed', default=42, help='set seed')
    parser.add_argument('--aux_loss', default=False, help='aux_loss')
    parser.add_argument('--resume', type=bool, default=False, help='put the path to resuming file if needed')
    parser.add_argument('--criterion', default='CrossEntropyLoss', help='criterion')
    parser.add_argument('--lr_param_groups', default=True, help='lr_param_groups')
    parser.add_argument('--weight_sampler', default=True, help='lr_param_groups')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                       help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                       help='Decay factor for model weights moving average (default: 0.9998)')
    parser.add_argument('--model-ema-warmup', action='store_true',
                       help='Enable warmup for model EMA decay.')
    parser.add_argument('--ema_updata_epoch', type=int, default=4)
    parser.add_argument('--model_ema', type=bool, default=True)

    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--freeze_layers', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--apm', default=True, help='Automatic Mixed Precision')
    parser.add_argument('--distributed', default=False, help='distributed')
    parser.add_argument('--cam_visualization', type=bool, default=False)
    parser.add_argument('--visual_feature', type=bool, default=False)
    parser.add_argument('--k_fold', type=bool, default=True)
    parser.add_argument('--statistics', type=bool, default=False)
    parser.add_argument('--test', type=bool, default=True)
    parser.add_argument('--output_csv', type=str, default=None)
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    #ModelArguments
    parser.add_argument('--model', type=str, default='LENet',
                        choices=['Resnet', 'Mobilenet', 'Cswin', 'Convnext', 'SwinTransformerV2',
                                 'SwinTransformer','Load_contrastive','DaViT','Liver','Cswin',
                                 'Hiera','Densenet','Resnext','Thynet','ViT','LivNet_Baseline'
                                 'SwinTransformer_S','LivNet_no_moe','LivNet_MOE','LENet','Convnext_B'
                                 'PVT_v2_b5','FocalNet','Mpvit_base'
                                 ])
    parser.add_argument('--serial_parallel', type=str, default='serial',
                        choices=['parallel', 'serial','invert_serial'],help='Types of attention')
    parser.add_argument('--sparse_dense', type=str, default='sparse_token',
                        choices=['sparse_token', 'dense_token','sparse_expert','dense_expert','mlp'],
                        help='Types of MOE')
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--head_type', type=str, default='linear',
                        choices=['moe_head', 'linear','mlp' ],
                        )
    parser.add_argument('--cat_moe_head', type=bool, default=False)
    parser.add_argument('--q_former_depths', type=list, default=[1,1,1,1],
    choices=[[1,1,1,1],[1,1,9,1],[1, 2, 7, 2],[2,2,6,2],[3,4,6,3],[1,3,6,3],[1],[1,1],[1,1,1]])
    parser.add_argument('--stage_dims', type=list, default=[96,192,384,768],
    choices=[[96, 192, 384, 768],[192, 384, 768],[384, 768],[768]])
    parser.add_argument('--q_former_head_num', type=list, default=[3,6,12,24],
    choices=[[3, 6, 12, 24],[24],[12, 24],[6, 12, 24]])
    parser.add_argument('--num_query_tokens', type=int, default=200,
    choices=[50,100,200,300])
    parser.add_argument('--query_dim', type=int, default=384,
    choices=[96,192,384,768])

    args = parser.parse_args()

    if args.k_fold:
        k_fold=5
        for i in range(0,k_fold):
            print('第{}折'.format(i))
            main(0, i, args)
    else:
        main(0, 0, args)

