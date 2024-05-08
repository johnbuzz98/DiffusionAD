import json
import os
from collections import defaultdict

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from data.dataset_beta_thresh import (
    DAGMTestDataset,
    DAGMTrainDataset,
    MPDDTestDataset,
    MPDDTrainDataset,
    MVTecTestDataset,
    MVTecTrainDataset,
    VisATestDataset,
    VisATrainDataset,
)
from models.DDPM import GaussianDiffusionModel, get_beta_schedule
from models.Recon_subnetwork import UNetModel
from models.Seg_subnetwork import SegmentationSubNetwork
from piq import psnr, ssim
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
import argparse

# Change the current working directory
os.chdir("/workspace/DiffusionAD")
logger = get_logger(__name__)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def defaultdict_from_json(jsonDict):
    func = lambda: defaultdict(str)
    dd = func()
    dd.update(jsonDict)
    return dd


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=4, logits=False, reduce=True):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def train(
    training_dataset_loader,
    testing_dataset_loader,
    unet_model,
    seg_model,
    optimizer_ddpm,
    optimizer_seg,
    scheduler_seg,
    args,
    data_len,
    sub_class,
    class_type,
    accelerator,
):
    betas = get_beta_schedule(args["T"], args["beta_schedule"])
    ddpm_sample = GaussianDiffusionModel(
        args["img_size"],
        betas,
        loss_weight=args["loss_weight"],
        loss_type=args["loss-type"],
        noise=args["noise_fn"],
        img_channels=args["channels"],
    )
    # tqdm_epoch = range(0, args["EPOCHS"])
    tqdm_epoch = tqdm(range(0, args["EPOCHS"]), disable=not accelerator.is_local_main_process)
    # dataset loop
    best_image_auroc = 0.0
    best_pixel_auroc = 0.0
    best_epoch = 0
    image_auroc_list = []
    pixel_auroc_list = []
    performance_x_list = []

    loss_focal = BinaryFocalLoss().to(accelerator.device)
    loss_smL1 = nn.SmoothL1Loss().to(accelerator.device)
    unet_model.to(accelerator.device)
    seg_model.to(accelerator.device)

    logger.info("***** Running training *****")
    step = 0
    for epoch in tqdm_epoch:
        unet_model.train()
        seg_model.train()
        tbar = tqdm(training_dataset_loader, disable=not accelerator.is_local_main_process)
        # tbar = tqdm(training_dataset_loader)
        for i, sample in enumerate(tbar):
            aug_image = sample["augmented_image"]  # .to(accelerator.device)
            anomaly_mask = sample["anomaly_mask"]  # .to(accelerator.device)
            anomaly_label = sample["has_anomaly"].squeeze()  # .to(accelerator.device).squeeze()

            noise_loss, pred_x0, normal_t, x_normal_t, x_noiser_t = ddpm_sample.norm_guided_one_step_denoising(
                unet_model, aug_image, anomaly_label, args
            )
            pred_mask = seg_model(torch.cat((aug_image, pred_x0), dim=1))

            # loss
            focal_loss = loss_focal(pred_mask, anomaly_mask)
            smL1_loss = loss_smL1(pred_mask, anomaly_mask)
            loss = noise_loss + 5 * focal_loss + smL1_loss

            accelerator.backward(loss)
            optimizer_ddpm.step()
            optimizer_seg.step()
            scheduler_seg.step()
            optimizer_ddpm.zero_grad()
            optimizer_seg.zero_grad()
            if accelerator.sync_gradients:
                step += 1
            logger.info("Epoch:%d, Step:%d, Train Loss: %.3f" % (epoch, step, loss.item()))
            step += 1
            accelerator.log(
                {
                    "Train Loss": loss.item(),
                    "Train Focal Loss": focal_loss.item(),
                    "Train smL1 Loss": smL1_loss.item(),
                    "Train Noise Loss": noise_loss.item(),
                    "Epoch": epoch,
                },
                step=step,
            )
            logger.info("Epoch:%d, Step:%d, Train Loss: %.3f" % (epoch, step, loss.item()))

        if (epoch + 1) % 100 == 0 and epoch > 0:  # evaluation
            total_image_pred, total_image_gt, total_pixel_pred, total_pixel_gt = eval(
                testing_dataset_loader, args, unet_model, seg_model, accelerator, sub_class
            )
            normal_aug_image, normal_pred_x0, anomaly_aug_image, anomaly_pred_x0 = (
                eval_denoise(  # TODO 여기서 고장남 RuntimeError: Tensors must be CUDA and dense
                    training_dataset_loader,
                    ddpm_sample,
                    unet_model,
                    args,
                    accelerator,
                    200 / accelerator.num_processes,
                )
            )
            total_image_pred, total_image_gt, total_pixel_pred, total_pixel_gt = accelerator.gather_for_metrics(
                (total_image_pred, total_image_gt, total_pixel_pred, total_pixel_gt)
            )
            normal_aug_image, normal_pred_x0, anomaly_aug_image, anomaly_pred_x0 = accelerator.gather_for_metrics(
                (normal_aug_image, normal_pred_x0, anomaly_aug_image, anomaly_pred_x0)
            )
            normal_aug_image = torch.clamp(normal_aug_image, max=1.0)
            normal_pred_x0 = torch.clamp(normal_pred_x0, max=1.0)
            anomaly_aug_image = torch.clamp(anomaly_aug_image, max=1.0)
            anomaly_pred_x0 = torch.clamp(anomaly_pred_x0, max=1.0)

            ssim_score_0 = ssim(normal_aug_image, normal_pred_x0, data_range=1.0, reduction="mean")
            ssim_score_1 = ssim(anomaly_aug_image, anomaly_pred_x0, data_range=1.0, reduction="mean")
            psnr_score_0 = psnr(normal_aug_image, normal_pred_x0, data_range=1.0)
            psnr_score_1 = psnr(anomaly_aug_image, anomaly_pred_x0, data_range=1.0)
            auroc_image = (
                round(roc_auc_score(total_image_gt.detach().cpu().numpy(), total_image_pred.detach().cpu().numpy()), 6)
                * 100
            )
            auroc_pixel = (
                round(
                    roc_auc_score(
                        total_pixel_gt.detach().cpu().numpy().astype(bool).astype(int),
                        total_pixel_pred.detach().cpu().numpy(),
                    ),
                    6,
                )
                * 100
            )
            accelerator.log(
                {
                    "Image AUROC": auroc_image,
                    "Pixel AUROC": auroc_pixel,
                    "Normal SSIM": ssim_score_0,
                    "Abnormal SSIM": ssim_score_1,
                    "Normal PSNR": psnr_score_0,
                    "Abnormal PSNR": psnr_score_1,
                    "Original Image(Normal)": [wandb.Image(to_pil_image(image)) for image in normal_aug_image],
                    "Reconstructed Image(Normal)": [wandb.Image(to_pil_image(image)) for image in normal_pred_x0],
                    "Original Image(Abnormal)": [wandb.Image(to_pil_image(image)) for image in anomaly_aug_image],
                    "Reconstructed Image(Abnormal)": [wandb.Image(to_pil_image(image)) for image in anomaly_pred_x0],
                    "Epoch": epoch,
                },
                step=step,
            )
            logger.info(
                "Normal SSIM: %.3f, Abnormal SSIM: %.3f, Normal PSNR: %.3f, Abnormal PSNR: %.3f"
                % (ssim_score_0, ssim_score_1, psnr_score_0, psnr_score_1)
            )
            image_auroc_list.append(auroc_image)
            pixel_auroc_list.append(auroc_pixel)
            performance_x_list.append(int(epoch))
            if auroc_image + auroc_pixel >= best_image_auroc + best_pixel_auroc:
                if auroc_image >= best_image_auroc:
                    save(unet_model, seg_model, args=args, final="best", epoch=epoch, sub_class=sub_class)
                    best_image_auroc = auroc_image
                    best_pixel_auroc = auroc_pixel
                    best_epoch = epoch

    save(unet_model, seg_model, args=args, final="last", epoch=args["EPOCHS"], sub_class=sub_class)

    temp = {
        "classname": [sub_class],
        "Image-AUROC": [best_image_auroc],
        "Pixel-AUROC": [best_pixel_auroc],
        "epoch": best_epoch,
    }
    df_class = pd.DataFrame(temp)
    df_class.to_csv(
        f"{args['output_path']}/metrics/ARGS={args['arg_num']}/{args['eval_normal_t']}_{args['eval_noisier_t']}t_{args['condition_w']}_{class_type}_image_pixel_auroc_train.csv",
        mode="a",
        header=False,
        index=False,
    )


def eval_denoise(training_dataset_loader, ddpm_sample, unet_model, args, accelerator, num_images=100):
    unet_model.eval()
    tbar = tqdm(training_dataset_loader, disable=not accelerator.is_local_main_process)
    with torch.no_grad():
        for idx, sample in enumerate(tbar):
            if idx == 0:
                aug_image = sample["augmented_image"]
                anomaly_label = sample["has_anomaly"].squeeze()
                _, pred_x0, _, _, _ = ddpm_sample.norm_guided_one_step_denoising(
                    unet_model, sample["augmented_image"], sample["has_anomaly"].squeeze(), args
                )
            else:
                aug_image = torch.cat((aug_image, sample["augmented_image"]), dim=0)
                anomaly_label = torch.cat((anomaly_label, sample["has_anomaly"].squeeze()), dim=0)
                _, pred_x0_, _, _, _ = ddpm_sample.norm_guided_one_step_denoising(
                    unet_model, sample["augmented_image"], sample["has_anomaly"].squeeze(), args
                )
                pred_x0 = torch.cat((pred_x0, pred_x0_), dim=0)
            if aug_image.shape[0] >= num_images:
                break
    pred_x0 += 1
    pred_x0 /= 2  # torch.clamp(pred_x0[idx], 0, 1)

    normal_indices = anomaly_label == 0
    anomaly_indices = anomaly_label == 1

    normal_aug_image = aug_image[normal_indices]
    anomaly_aug_image = aug_image[anomaly_indices]
    normal_pred_x0 = pred_x0[normal_indices]
    anomaly_pred_x0 = pred_x0[anomaly_indices]

    return normal_aug_image, normal_pred_x0, anomaly_aug_image, anomaly_pred_x0


def eval(testing_dataset_loader: DataLoader, args: dict, unet_model, seg_model, accelerator, sub_class: str):
    unet_model.eval()
    seg_model.eval()
    os.makedirs(f'{args["output_path"]}/metrics/ARGS={args["arg_num"]}/{sub_class}/', exist_ok=True)
    in_channels = args["channels"]
    betas = get_beta_schedule(args["T"], args["beta_schedule"])

    ddpm_sample = GaussianDiffusionModel(
        args["img_size"],
        betas,
        loss_weight=args["loss_weight"],
        loss_type=args["loss_type"],
        noise=args["noise_fn"],
        img_channels=in_channels,
    )

    total_image_pred = torch.empty(0, device=accelerator.device)  # Modified to PyTorch tensor
    total_image_gt = torch.empty(0, device=accelerator.device)  # Modified to PyTorch tensor
    total_pixel_gt = torch.empty(0, device=accelerator.device)  # Modified to PyTorch tensor
    total_pixel_pred = torch.empty(0, device=accelerator.device)  # Modified to PyTorch tensor
    tbar = tqdm(testing_dataset_loader, disable=not accelerator.is_local_main_process)
    with torch.no_grad():
        for i, sample in enumerate(tbar):
            image = sample["image"]  # Assuming .to(device) if needed
            target = sample["has_anomaly"]  # Assuming .to(device) if needed
            gt_mask = sample["mask"]  # Assuming .to(device) if needed

            normal_t_tensor = torch.tensor([args["eval_normal_t"]], device=accelerator.device).repeat(image.shape[0])
            noiser_t_tensor = torch.tensor([args["eval_noisier_t"]], device=accelerator.device).repeat(image.shape[0])
            _, pred_x_0_condition, _, _, _, _, _ = ddpm_sample.norm_guided_one_step_denoising_eval(
                unet_model, image, normal_t_tensor, noiser_t_tensor, args
            )
            pred_mask = seg_model(torch.cat((image, pred_x_0_condition), dim=1))

            out_mask = pred_mask

            topk_out_mask = torch.flatten(out_mask[0], start_dim=1)
            topk_out_mask = torch.topk(topk_out_mask, 50, dim=1, largest=True)[0]
            image_score = torch.mean(topk_out_mask)

            total_image_pred = torch.cat((total_image_pred, image_score.unsqueeze(0)))  # Modified
            total_image_gt = torch.cat((total_image_gt, target[0].unsqueeze(0)))  # Modified

            flatten_pred_mask = out_mask[0].flatten()
            flatten_gt_mask = gt_mask[0].flatten()  # .to(torch.int32)  # .astype(int) changed to .to()

            total_pixel_gt = torch.cat((total_pixel_gt, flatten_gt_mask))  # Modified
            total_pixel_pred = torch.cat((total_pixel_pred, flatten_pred_mask))  # Modified

    return (
        total_image_pred,
        total_image_gt,
        total_pixel_pred,
        total_pixel_gt,
    )


def save(unet_model, seg_model, args, final, epoch, sub_class):
    if final == "last":
        torch.save(
            {
                "n_epoch": epoch,
                "unet_model_state_dict": unet_model.state_dict(),
                "seg_model_state_dict": seg_model.state_dict(),
                "args": args,
            },
            f'{args["output_path"]}/model/diff-params-ARGS={args["arg_num"]}/{sub_class}/params-{final}.pt',
        )

    else:
        torch.save(
            {
                "n_epoch": epoch,
                "unet_model_state_dict": unet_model.state_dict(),
                "seg_model_state_dict": seg_model.state_dict(),
                "args": args,
            },
            f'{args["output_path"]}/model/diff-params-ARGS={args["arg_num"]}/{sub_class}/params-{final}.pt',
        )


def main(sub_class):
    # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    # read file from argument
    file = "args1.json"
    # load the json args
    with open(f"./args/{file}", "r") as f:
        args = json.load(f)
    args["arg_num"] = file[4:-5]
    args = defaultdict_from_json(args)
    set_seed(args["seed"])
    mvtec_classes = [
        "carpet",
        "grid",
        "leather",
        "tile",
        "wood",
        "bottle",
        "cable",
        "capsule",
        "hazelnut",
        "metal_nut",
        "pill",
        "screw",
        "toothbrush",
        "transistor",
        "zipper",
    ]

    visa_classes = [
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum",
    ]

    mpdd_classes = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
    dagm_class = ["Class1", "Class2", "Class3", "Class4", "Class5", "Class6", "Class7", "Class8", "Class9", "Class10"]

    current_classes = visa_classes

    class_type = ""
    #for sub_class in current_classes:
    print("class", sub_class)
    if sub_class in visa_classes:
        subclass_path = os.path.join(args["visa_root_path"], sub_class)
        print(subclass_path)
        training_dataset = VisATrainDataset(subclass_path, sub_class, img_size=args["img_size"], args=args)
        testing_dataset = VisATestDataset(
            subclass_path,
            sub_class,
            img_size=args["img_size"],
        )
        class_type = "VisA"
    elif sub_class in mpdd_classes:
        subclass_path = os.path.join(args["mpdd_root_path"], sub_class)
        training_dataset = MPDDTrainDataset(subclass_path, sub_class, img_size=args["img_size"], args=args)
        testing_dataset = MPDDTestDataset(
            subclass_path,
            sub_class,
            img_size=args["img_size"],
        )
        class_type = "MPDD"
    elif sub_class in mvtec_classes:
        subclass_path = os.path.join(args["mvtec_root_path"], sub_class)
        training_dataset = MVTecTrainDataset(subclass_path, sub_class, img_size=args["img_size"], args=args)
        testing_dataset = MVTecTestDataset(
            subclass_path,
            sub_class,
            img_size=args["img_size"],
        )
        class_type = "MVTec"
    elif sub_class in dagm_class:
        subclass_path = os.path.join(args["dagm_root_path"], sub_class)
        training_dataset = DAGMTrainDataset(subclass_path, sub_class, img_size=args["img_size"], args=args)
        testing_dataset = DAGMTestDataset(
            subclass_path,
            sub_class,
            img_size=args["img_size"],
        )
        class_type = "DAGM"

    print(file, args)

    data_len = len(testing_dataset)
    training_dataset_loader = DataLoader(
        training_dataset,
        batch_size=args["Batch_Size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(testing_dataset, batch_size=1, shuffle=False, num_workers=4)

    # make arg specific directories
    for i in [
        f'{args["output_path"]}/model/diff-params-ARGS={args["arg_num"]}/{sub_class}',
        f'{args["output_path"]}/diffusion-training-images/ARGS={args["arg_num"]}/{sub_class}',
        f'{args["output_path"]}/metrics/ARGS={args["arg_num"]}/{sub_class}',
    ]:
        try:
            os.makedirs(i)
        except OSError:
            pass
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        log_with="wandb",
        kwargs_handlers=[ddp_kwargs],
    )
    # wandb.init(project="DiffusionAD", entity="woojun_lee", config=args, name=f"{class_type}_{sub_class}")

    accelerator.init_trackers(
        project_name="DiffusionAD",
        config=args,
        init_kwargs={
            "wandb": {
                "entity": "woojun_lee",
                "name": f"{class_type}_{sub_class}",
                "dir": "/workspace3/diffusion_ad",
            }
        },
    )

    unet_model = UNetModel(
        args["img_size"][0],
        args["base_channels"],
        channel_mults=args["channel_mults"],
        dropout=args["dropout"],
        n_heads=args["num_heads"],
        n_head_channels=args["num_head_channels"],
        in_channels=args["channels"],
    )

    seg_model = SegmentationSubNetwork(in_channels=6, out_channels=1)

    optimizer_ddpm = optim.Adam(
        unet_model.parameters(), lr=args["diffusion_lr"], weight_decay=args["weight_decay"]
    )

    optimizer_seg = optim.Adam(seg_model.parameters(), lr=args["seg_lr"], weight_decay=args["weight_decay"])

    scheduler_seg = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_seg, T_max=10, eta_min=0, last_epoch=-1, verbose=False
    )

    training_dataset_loader, test_loader, unet_model, seg_model, optimizer_ddpm, optimizer_seg, scheduler_seg = (
        accelerator.prepare(
            training_dataset_loader,
            test_loader,
            unet_model,
            seg_model,
            optimizer_ddpm,
            optimizer_seg,
            scheduler_seg,
        )
    )

    train(
        training_dataset_loader,
        test_loader,
        unet_model,
        seg_model,
        optimizer_ddpm,
        optimizer_seg,
        scheduler_seg,
        args,
        data_len,
        sub_class,
        class_type,
        accelerator,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub_class", type=str, default="candle")
    args = parser.parse_args()
    main(args.sub_class)