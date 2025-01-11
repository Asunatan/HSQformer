import logging

import math
import os.path
import sys
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import torchvision
from PIL import Image
from torchvision.models import resnet50
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F
import cv2
from pytorch_grad_cam import GradCAM, AblationCAM, ScoreCAM, HiResCAM, GradCAMPlusPlus, XGradCAM
#from mmcls.core import f1_score
import numpy as np
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn
from tqdm import tqdm
import torch
from .distributed_utils import reduce_value,get_rank
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,roc_auc_score,auc
from torchvision.utils import make_grid
def metric_info(metric_collection):
    metric = {}
    for k,v in metric_collection.compute().items():
        metric[k.replace('Binary', '')]= np.around(v.numpy(), decimals=4)
    return metric
def KD_loss(logits_student, logits_teacher, temperature=1.0):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="batchmean")#todo reduction='none'.sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd
def DKD_loss(logits_student, logits_teacher, target, alpha=1.0, beta=8.0, temperature=1.0):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


def train_one_epoch(model, optimizer,metric_collection=None, data_loader=None, device=0,num_updates=0,epoch=0,criterion_focal=None,
                    scheduler=None,criterion=None,mixup_fn=None,scaler=None,aux_loss=None,model_ema=None,ema_updata_epoch=None):
    metric_collection.reset()
    model.train()
    total_loss = 0.
    data_loader = tqdm(data_loader, desc=' train')
    for step, (img, labels) in enumerate(data_loader):
        target=labels
        img=img.cuda()
        labels=labels.cuda()
        if mixup_fn is not None:
            img, labels = mixup_fn(img, labels)
        with torch.cuda.amp.autocast():
            output = model(img)
            if aux_loss:
                loss = criterion(output, labels)+criterion_focal(output,labels,target)
            else:
                loss = criterion(output, labels)
        if not math.isfinite(loss): # this could trigger if using AMP
            logging.info(f'Loss is {loss}, stopping training')
            assert math.isfinite(loss)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            total_loss = total_loss + loss.item()
            data_loader.set_description('loss:{:.6f}'.format(loss.item()))
            pred = torch.softmax(output, 1)
            metric_collection.update(pred[:,1].detach().cpu(), target.detach().cpu())
        scheduler.step()
    return total_loss

@torch.no_grad()
def evaluate(model, metric_collection=None, data_loader=None, device=0,criterion=None,
             args=None,k_fold=None):
    metric_collection.reset()
    model.eval()
    total_loss=0
    data_loader = tqdm(data_loader, desc='val',file=sys.stdout)
    id_list,M_result_list,label_list=[],[],[]
    for step, (img, labels,img_path) in enumerate(data_loader):

        img=img.to(device)
        labels=labels.to(device)
        with torch.cuda.amp.autocast():
            pred = model(img)
            loss = criterion(pred, labels)
        total_loss = total_loss + loss.item()
        data_loader.set_description('loss: {:.4f} '.format(loss.item()))
        pred=pred.softmax(1).detach().cpu()
        metric_collection.update(pred[:,1], labels.detach().cpu())
        if args.output_csv is not None:
            id_list.extend(img_path)
            M_result_list.extend(pred.numpy()[:,1])
            label_list.extend(labels.detach().cpu().numpy())
    if args.output_csv is not None:
        PathDF = pd.DataFrame({'id': id_list, 'M': M_result_list,'label': label_list})
        PathDF.to_csv(f"{args.output_csv}{k_fold}.csv", index=False)

    return total_loss

@torch.no_grad()
def test_prob(model,metric_collection=None,num_classes=2, data_loader=None, device=0,optimizer=None,
                           criterion=None,mixup_fn=None,scaler=None,aux_loss=None,model_ema=None,criterion_focal=None,
                            output_excel=False,k_flod=None
                                   ):
    model.eval()
    metric_collection.reset()
    total_loss=0
    TTA_result = 0.
    TTA_number=1
    for i in range(TTA_number):
        data_loader = tqdm(data_loader, desc='test',file=sys.stdout)
        running_pred = []
        running_label = []
        running_id=[]
        for step, (img, labels,id) in enumerate(data_loader):
            img = img.to(device)
            labels = labels.to(device)
            loss = 0.
            pred=0.
            # pred_list=[]
            with torch.cuda.amp.autocast():
                outputs_list = model(img)
                if isinstance(outputs_list, list):
                    for out in outputs_list:
                        loss = loss + criterion(out, labels)
                        pred += out#torch.argmax(out, 1).detach().cpu()  # .numpy()
                else :
                    pred=outputs_list
            running_label.append(labels)
            running_id.extend(id)
            running_pred.append(pred)
        one_epoch_pred = torch.cat(running_pred, dim=0)
        one_epoch_label = torch.cat(running_label, dim=0)
        # one_epoch_id = torch.cat(running_id, dim=0)
        TTA_result += one_epoch_pred
        # pred = pred.softmax(1).detach().cpu().numpy()
    TTA_result /= TTA_number
    TTA_result = TTA_result.softmax(1).detach().cpu()
    one_epoch_label = one_epoch_label.detach().cpu()

    PathDF = pd.DataFrame({'id':running_id,'B': TTA_result[:,0].numpy(),'M':TTA_result[:, 1].numpy(),'AI_label':TTA_result.argmax(1),'label':one_epoch_label})
    PathDF.to_csv("LivNet_3020_test{}.csv".format(k_flod), index=True)#4565
    TTA_result=TTA_result.argmax(1)
    metric_collection.update(TTA_result.float(), one_epoch_label)
    return loss



@torch.no_grad()
def liver_2classification_test(model,metric_collection=None,num_classes=2, data_loader=None, device=0,
                           criterion=None,mixup_fn=None,scaler=None,aux_loss=None,model_ema=None,criterion_focal=None,
                            output_excel=False,
                                   ):
    model.eval()
    metric_collection.reset()
    total_loss=0
    TTA_result = 0.
    TTA_number=5
    for i in range(TTA_number):
        data_loader = tqdm(data_loader, desc='test',file=sys.stdout)
        running_pred = []
        running_label = []
        for step, (img, labels) in enumerate(data_loader):
            img = img.to(device)
            labels = labels.to(device)
            loss = 0.
            pred=0.
            pred_list=[]
            with torch.cuda.amp.autocast():
                outputs_list = model(img)

                for out in outputs_list:
                    loss = loss + criterion(out, labels)
                    pred += out#torch.argmax(out, 1).detach().cpu()  # .numpy()
                running_label.append(labels)
                running_pred.append(pred)
        one_epoch_pred=torch.cat(running_pred,dim=0)
        one_epoch_label=torch.cat(running_label,dim=0)

        TTA_result += one_epoch_pred#.detach().cpu()#.numpy()

    one_epoch_label = one_epoch_label.detach().cpu()
    TTA_result/=TTA_number
    TTA_result = TTA_result.softmax(1).detach().cpu().numpy()

    # running_pred.append(pred.softmax(1).detach().cpu())

    # one_epoch_pred=torch.cat(running_pred,dim=0)


    # PathDF = pd.DataFrame(TTA_result)
    # PathDF.to_csv("1800.csv", index=True)
    TTA_result=TTA_result.argmax(1)
    metric_collection.update(TTA_result.float(), one_epoch_label)

    # model.eval()
    # total_loss = 0
    # metric_collection.reset()
    # data_loader = tqdm(data_loader, desc='test', file=sys.stdout)
    # for step, (img, labels) in enumerate(data_loader):
    #     img=img.to(device)
    #     labels=labels.to(device)
    #     with torch.cuda.amp.autocast():
    #         outputs = model(img)
    #
    #         loss = torch.zeros(1).to(device)
    #         target = labels.detach().cpu()#.numpy()
    #         result = 0.
    #         for out in outputs:
    #             loss =loss+criterion(out, labels)
    #             #加权
    #             pred = torch.argmax(out, 1).detach().cpu()#.numpy()
    #             weights = accuracy_score(target, pred)
    #             result += weights * out
    #     result=result.argmax(1).cpu()
    #     metric_collection.update(result, target)
    return total_loss#,val_pred,val_label




@torch.no_grad()
def rl_extract_feature(model,metric_collection=None,data_df_len=None,num_classes=2, data_loader=None, device=0,
                           criterion=None,mixup_fn=None,scaler=None,aux_loss=None,model_ema=None,criterion_focal=None,
                            output_excel=False,
                                   ):
    model.eval()
    metric_collection.reset()
    total_loss=0
    TTA_feature = 0.
    for i in range(1):
        data_loader = tqdm(data_loader, desc='test',file=sys.stdout)
        running_feature = []
        # for step, (id,img, labels,AFP,size) in enumerate(data_loader):
        #     img ,AFP,size= img.to(device),AFP.to(device),size.to(device)
        #     labels=labels.to(device).float().unsqueeze(1)
        for step, (img, labels) in enumerate(data_loader):
            img = img.to(device)
            labels = labels.to(device)
            target = labels.float().unsqueeze(1)
            with torch.cuda.amp.autocast():
                probability,multi_feature = model(img)
                if aux_loss:
                    probability_all=0
                    for i in range(8):
                        multi_feature[i]=multi_feature[i].mean([-2, -1])
                        probability_all+=probability[i]
                    multi_feature=torch.cat(multi_feature,dim=1)
                    features = torch.cat((multi_feature,probability_all.softmax(1),target),dim=1)
                    running_feature.append(features)
            metric_collection.update(probability_all.argmax(1), labels)
        one_epoch_feature=torch.cat(running_feature,dim=0)
        TTA_feature+=one_epoch_feature.detach().cpu().numpy()


    # TTA_feature/=1.


    # np.save('test_features.npy',TTA_feature)
    # a=0

        #     loss = torch.zeros(1).to(device)
        #     target = labels.detach().cpu()#.numpy()
        #     result = torch.zeros_like(outputs[0])
        #     for out in outputs:
        #         loss =loss+criterion(out, labels)
        #         #加权
        #         pred = torch.argmax(out, 1).detach().cpu()#.numpy()
        #         weights = accuracy_score(target, pred)
        #         result += weights * out
        # result=result.softmax(1)

        # #category_list.extend(category.detach().cpu().numpy())
        #
        #
        # total_loss = total_loss + loss.item()
        # data_loader.set_description('test_loss: {:.4f} '.format(loss.item()))


        #pred_class = result.argmax(1).detach().cpu()

        #metric_collection.update(pred_class.float(), target)
    return TTA_feature#,val_pred,val_label
@torch.no_grad()
def visual_feature(model,metric_collection=None,data_df_len=None,num_classes=2, data_loader=None, device=0,
                           criterion=None,mixup_fn=None,scaler=None,aux_loss=None,model_ema=None,criterion_focal=None,
                            output_excel=False,
                                   ):
    model.eval()
    metric_collection.reset()
    data_loader = tqdm(data_loader, desc='test', file=sys.stdout)
    running_pred = []
    running_label = []
    total_loss=0
    for step, (img, labels,img_path) in enumerate(data_loader):
        img=img.to(device)
        labels=labels.to(device)
        with torch.cuda.amp.autocast():
            feature_list = model(img)
        for stage,feature in enumerate(feature_list):
            if len(feature.shape)==4:
                grid_img = make_grid(feature[:,:,:,:48].permute(3, 0, 1, 2), normalize=False, scale_each=True, nrow=8)
                plt.figure(figsize=(20, 20))  # 你可能需要调整这个大小以适应你的显示需求
                plt.imshow(grid_img.detach().cpu().numpy().transpose(1, 2, 0),cmap='gray' )#,cmap='gray'  # 调整通道顺序以适应 matplotlib 的要求
                plt.axis('off')
                plt.savefig(f'/home/uax/SCY/Decouple_liver/visual_feature/combine/swin_{stage}.png',bbox_inches='tight', pad_inches=0)
                plt.show()
                plt.close()
            if len(feature.shape)==3:
                B, N, C = feature.shape
                H= W = int(np.sqrt(N))
                feature = feature.view(B, H, W, C).contiguous()
                grid_img = make_grid(feature[:,:,:,:48].permute(3, 0, 1, 2), normalize=False, scale_each=False, nrow=8)#.permute(3, 0, 1, 2)
                plt.figure(figsize=(20, 20))  # 你可能需要调整这个大小以适应你的显示需求
                plt.imshow(grid_img.detach().cpu().numpy().transpose(1, 2, 0))  # 调整通道顺序以适应 matplotlib 的要求
                plt.axis('off')
                plt.savefig(f'/home/uax/SCY/Decouple_liver/visual_feature/combine/conv_all_{stage}.png',bbox_inches='tight', pad_inches=0)
                plt.show()






def reshape_transform(tensor, height=20, width=20):
    # return tensor.permute(0, 3, 1, 2)
    result = tensor.reshape(tensor.size(0),
        height, width, tensor.size(2))

    result = result.permute(0, 3, 1, 2)
    return result
    # return tensor.permute(0,3,1,2)
def CAM_visualization(model, data_loader,device=0):
    output_path='/home/uax/SCY/Decouple_liver/visual_feature/cam/LIVER'
    checkpoint = torch.load('/media/uax/CA4E64EFA9C3DA83/HCC/ablation/LENet-serial-sparse_token-4experts-top_2-linear-q_former_depths2_2_6_2-num_query_400-query_dim_768_val0.pth'
                            ,map_location='cpu')
    #/media/uax/CA4E64EFA9C3DA83/HCC/convnext/Convnext_val0.pth
    #/home/uax/SCY/Decouple_liver/checkpoints/SwinTransformer_S_val0.pth
    #/media/uax/CA4E64EFA9C3DA83/HCC/ablation/LENet-serial-sparse_token-4experts-top_1-linear-q_former_depths2_2_6_2-num_query_100-query_dim_768_val0.pth
    model.load_state_dict(checkpoint['state_dict'])
    os.makedirs(output_path,exist_ok=True)
    model.eval()
    target_layers = [model.sc_att_mlp[-1]]
    # targets = [ClassifierOutputTarget(1)]
    targets = None
    # cam=GradCAM(model=model, target_layers=target_layers,reshape_transform=reshape_transform)
    cam =GradCAM(model=model, target_layers=target_layers,reshape_transform=reshape_transform)
    data_loader = tqdm(data_loader, desc='cam',file=sys.stdout)

    for step, (images,_, rgb_img_path) in enumerate(data_loader):
        images = images.to(device)
        grayscale_cam = cam(input_tensor=images,targets=targets,eigen_smooth=False,aug_smooth=False,)
        grayscale_cam = grayscale_cam[0, :]
        rgb_img = cv2.imread(rgb_img_path[0], 1)#[:, :, ::-1]
        # rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img = np.float32(rgb_img) / 255
        cam_image = show_cam_on_image(rgb_img, grayscale_cam,False)
        img_name=rgb_img_path[0].split('/')[-1].split('_')[0]
        # cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_path,img_name+'.png'), cam_image)
        A=0
    A=0




