import torch
import os
import argparse
import json
import copy
import math
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import nn
from pathlib import Path
from torchvision import transforms as pth_transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import utils
import models
from vmamba_models.dino_vmamba import (
    dinov2_vmamba_small,
    Backbone_DINOv2_VSSM_2,
)
from vmamba_models.MambaDecoder import MambaDecoder
from torchvision import transforms as pth_transforms
from dataset.dataset_busbra import BUSBRADataset
from dataset.dataset_tn3k import TN3KDataset
from dataset.dataset_custom_coco import CocoMultiLabelDataset
from dataset.transforms import get_transforms, get_transforms_multilabel
from prepare_cross_validation import write_json
from segmentation_metrics import METRIC_NAMES, finite_mean, metrics_by_class


OPENUS_ROOT = Path(__file__).resolve().parent
DEBUT_ROOT = OPENUS_ROOT.parents[1]
DEBUT_DATA_ROOT = DEBUT_ROOT / 'data'
CUSTOM_COCO_JSON = DEBUT_DATA_ROOT / 'COCO/labels/cavite/annotations.json'
CUSTOM_IMAGES_ROOT = DEBUT_DATA_ROOT / 'COCO/images/cavite'
CUSTOM_SPLIT_FILE = OPENUS_ROOT / 'data/splits.json'
CUSTOM_METADATA_FILE = DEBUT_ROOT / 'utils/correspondance_echo_final.xlsx'


def configure_custom_classes(args, *datasets):
    """Infer the decoder width from COCO, validating an explicit override."""
    class_names = datasets[0].class_names
    if any(dataset.class_names != class_names for dataset in datasets[1:]):
        raise ValueError('CUSTOM datasets do not expose the same class order')
    detected = len(class_names)
    if args.num_classes is not None and args.num_classes != detected:
        raise ValueError(f'--num_classes={args.num_classes}, but {args.coco_json} defines '
                         f'{detected} classes: {list(class_names)}')
    args.num_classes = detected
    args.class_names = list(class_names)
    print(f"Detected classes ({detected}): {', '.join(class_names)}")

def checkpoint_improved(metric, miou, dice, best_miou, best_dice, selected):
    return (not selected or (metric in ("either", "miou") and miou > best_miou)
            or (metric in ("either", "dice") and dice > best_dice))


class SegmentationHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_dim, num_classes, kernel_size=1)

    def forward(self, tokens, orig_hw):
        # tokens: [B, N, D] where N = h*w patches
        B, N, D = tokens.shape
        h = w = int(math.sqrt(N))
        feat = tokens.permute(0,2,1).view(B, D, h, w)
        logits = self.conv1x1(feat)  # [B, C, h, w]
        # upsample to original resolution
        return F.interpolate(logits, size=orig_hw, mode='bilinear', align_corners=False)


class MambaDecoderHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.decoder = MambaDecoder(num_classes=num_classes)

    def forward(self, encoder_out, orig_hw):
        logits = self.decoder.forward(encoder_out)
  
        return F.interpolate(logits, size=orig_hw, mode='bilinear', align_corners=False)


def compute_iou(pred, target, eps=1e-6):
    pred_flat   = pred.view(-1).byte()
    target_flat = target.view(-1).byte()

    # binary masks for foreground
    pred_fg   = pred_flat == 1
    target_fg = target_flat == 1

    # compute intersection and union
    intersection = (pred_fg & target_fg).sum().float()
    union = (pred_fg | target_fg).sum().float()

    # handle empty union (no foreground in either pred or target) as IoU=1
    if union.item() == 0:
        return 1.0

    return ((intersection + eps) / (union + eps)).item()


def compute_dice(pred, target, eps=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    # flatten
    pred_flat   = pred.view(-1)
    target_flat = target.view(-1)
    # binary masks for foreground class
    pred_fg   = pred_flat == 1
    target_fg = target_flat == 1

    intersection = (pred_fg & target_fg).sum().float()
    dice = (2. * intersection + eps) / (pred_fg.sum().float() + target_fg.sum().float() + eps)
    return dice.item()


def compute_iou_multilabel(pred, target, eps=1e-6):
    """Per-class IoU for multi-label masks. pred/target: [B, C, H, W] in {0, 1}."""
    num_classes = pred.shape[1]
    ious = []
    for c in range(num_classes):
        pred_fg = pred[:, c].reshape(-1).bool()
        target_fg = target[:, c].reshape(-1).bool()
        intersection = (pred_fg & target_fg).sum().float()
        union = (pred_fg | target_fg).sum().float()
        if union.item() == 0:
            ious.append(1.0)  # no foreground in either, matching compute_iou convention
        else:
            ious.append(((intersection + eps) / (union + eps)).item())
    return ious


def compute_dice_multilabel(pred, target, eps=1e-6):
    """Per-class Dice for multi-label masks. pred/target: [B, C, H, W] in {0, 1}."""
    num_classes = pred.shape[1]
    dices = []
    for c in range(num_classes):
        pred_fg = pred[:, c].reshape(-1).bool()
        target_fg = target[:, c].reshape(-1).bool()
        intersection = (pred_fg & target_fg).sum().float()
        dice = (2. * intersection + eps) / (pred_fg.sum().float() + target_fg.sum().float() + eps)
        dices.append(dice.item())
    return dices


def train_seg(model, head, optimizer, loader, epoch, criterion):
    model.eval()
    head.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    for imgs, masks, mask_filenames in metric_logger.log_every(loader, 20, header):
        imgs = imgs.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)
        
        with torch.no_grad():
            if args.arch == 'vmamba_small':
                output = model(imgs)
                # output = output[:, 1:]
        
        logits = head(output, orig_hw=imgs.shape[2:])
        loss = criterion(logits, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # log 
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: float(meter.global_avg) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def validate_seg(val_loader, model, head, criterion, args):
    model.eval()
    head.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    total_iou = 0.0
    total_dice = 0.0
    total_iou_per_class = None
    total_dice_per_class = None
    totals = {name: [] for name in METRIC_NAMES}
    count = 0
    for imgs, masks, mask_filenames in metric_logger.log_every(val_loader, 20, 'Test:'):
        imgs = imgs.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)

        with torch.no_grad():
            if args.arch == 'vmamba_small':
                output = model(imgs)
                # output = output[:, 1:]

        logits = head(output, orig_hw=imgs.shape[2:])

        loss = criterion(logits, masks)
        if args.multilabel:
            preds = (torch.sigmoid(logits) > 0.5).float()
            if args.dataset_name == 'CUSTOM':
                sample_metrics = []
                for sample_pred, file_name in zip(preds, mask_filenames):
                    height, width = val_loader.dataset.original_size(file_name)
                    native_pred = F.interpolate(
                        sample_pred[None], size=(height, width), mode='nearest'
                    )[0].bool().cpu()
                    native_true = val_loader.dataset.native_mask(file_name)
                    values = metrics_by_class(
                        native_pred, native_true,
                        val_loader.dataset.scale_cm(file_name), height,
                    )
                    sample_metrics.append(values)
                    for name in METRIC_NAMES:
                        totals[name].append(values[name].cpu())
                iou_pc = torch.stack([value['iou'] for value in sample_metrics]).mean(0).tolist()
                dice_pc = torch.stack([value['dice'] for value in sample_metrics]).mean(0).tolist()
            else:
                iou_pc = compute_iou_multilabel(preds, masks)
                dice_pc = compute_dice_multilabel(preds, masks)
            if total_iou_per_class is None:
                total_iou_per_class = np.zeros(len(iou_pc))
                total_dice_per_class = np.zeros(len(dice_pc))
            total_iou_per_class += np.array(iou_pc) * preds.shape[0]
            total_dice_per_class += np.array(dice_pc) * preds.shape[0]
            batch_iou = float(np.mean(iou_pc))
            batch_dice = float(np.mean(dice_pc))
        else:
            preds = logits.argmax(dim=1)
            batch_iou = compute_iou(preds, masks)
            batch_dice = compute_dice(preds, masks)
        metric_weight = preds.shape[0] if args.multilabel else 1
        total_iou += batch_iou * metric_weight
        total_dice += batch_dice * metric_weight
        count += metric_weight

        metric_logger.update(loss=loss.item())
    metric_logger.synchronize_between_processes()
    metrics = {'loss': metric_logger.meters['loss'].global_avg,
               'miou': total_iou / max(count, 1),
               'dice': total_dice / max(count, 1)}
    if args.multilabel and total_iou_per_class is not None:
        metrics['iou_per_class'] = (total_iou_per_class / max(count, 1)).tolist()
        metrics['dice_per_class'] = (total_dice_per_class / max(count, 1)).tolist()
        metrics['class_names'] = args.class_names
        if args.dataset_name == 'CUSTOM' and totals['iou']:
            stacked_totals = {name: torch.stack(values) for name, values in totals.items()}
            for name in ('precision', 'recall', 'diff_surface'):
                values = stacked_totals[name]
                means = [finite_mean(values[:, c]) for c in range(values.shape[1])]
                metrics[f'{name}_per_class'] = [value if math.isfinite(value) else None
                                                for value in means]
                overall = finite_mean(values)
                metrics[name] = overall if math.isfinite(overall) else None
    return metrics


def eval_seg(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join(f"{k}: {v}" for k,v in sorted(vars(args).items())))
    cudnn.benchmark = True
    utils.fix_random_seeds(args.seed)

    if args.img_size is None:
        args.img_size = 512 if args.dataset_name == 'CUSTOM' else 224
    print(f"Input image size: {args.img_size}x{args.img_size}")

    # ============ preparing data ... ============
    train_transform, val_transform = get_transforms(img_size=args.img_size)

    if args.dataset_name == 'BUSBRA':
        organized_data_dir = "."
        fold = '0'  # Use fold 1
        train_ds = BUSBRADataset(organized_data_dir, fold, split='train', transform=train_transform)
        val_ds = BUSBRADataset(organized_data_dir, fold, split='validation', transform=val_transform)
    elif args.dataset_name == 'TN3K':
        train_ds = TN3KDataset(image_dir=args.data_root, mask_dir=args.data_root2, json_file=args.json_file, split='train', transform=train_transform)
        val_ds   = TN3KDataset(image_dir=args.data_root, mask_dir=args.data_root2, json_file=args.json_file,   split='val',   transform=val_transform)
    elif args.dataset_name == 'CUSTOM':
        train_transform, val_transform = get_transforms_multilabel(img_size=args.img_size)
        train_ds = CocoMultiLabelDataset(coco_json=args.coco_json, images_dir=args.images_root, split_file=args.split_file, split='train', transform=train_transform, metadata_file=args.metadata_file)
        val_ds   = CocoMultiLabelDataset(coco_json=args.coco_json, images_dir=args.images_root, split_file=args.split_file, split='val',   transform=val_transform, metadata_file=args.metadata_file)
        configure_custom_classes(args, train_ds, val_ds)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    if args.num_classes is None:
        args.num_classes = 2

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    train_loader  = DataLoader(train_ds, batch_size=args.batch_size_per_gpu, sampler=train_sampler,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader    = DataLoader(val_ds,   batch_size=args.batch_size_per_gpu, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)
    print(f"Data loaded: {len(train_ds)} train, {len(val_ds)} val samples.")

    # ============ building network ... ============
    if 'swin' in args.arch:
        args.patch_size = 4
        model = models.__dict__[args.arch](window_size=args.window_size,
                                            patch_size=args.patch_size,
                                            num_classes=0)
        embed_dim = model.num_features
    elif args.arch == 'vmamba_small':
        if args.pretrained_vmamba:
            model = Backbone_DINOv2_VSSM_2(pretrained='./pretrained/vmamba/vssm_small_0229_ckpt_epoch_222.pth',
                                          seg_head=True)
        else:
            model = dinov2_vmamba_small(patch_size=args.patch_size,
                                        return_all_tokens=True,
                                        masked_im_modeling=False)
        embed_dim = model.dims[-1]
    else:
        model = models.__dict__[args.arch](patch_size=args.patch_size,
                                           num_classes=0,
                                           use_mean_pooling=args.avgpool_patchtokens==1)
        embed_dim = model.embed_dim
    model.cuda()
    print(f"Backbone {args.arch} built.")

    # load pretrained backbone
    if args.arch == 'vmamba_small':
        if os.path.isfile(args.pretrained_weights):
            print(f"Loading pretrained weights from {args.pretrained_weights}")
            checkpoint = torch.load(args.pretrained_weights, map_location="cpu", weights_only=False)
            print("Keys in checkpoint:", checkpoint.keys())

            if args.checkpoint_key is not None and args.checkpoint_key in checkpoint:
                print(f"Taking key {args.checkpoint_key} in checkpoint")
                state_dict = checkpoint[args.checkpoint_key]
                print("State dict keys example (first 5):", list(state_dict.keys())[:5])
                
                clean_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('backbone.'):
                        new_key = k[9:]  # Remove 'backbone.'
                        clean_state_dict[new_key] = v
                
                try:
                    msg = model.load_state_dict(clean_state_dict, strict=False)
                    print(f"Successfully loaded pretrained weights with msg: {msg}")
                except Exception as e:
                    print(f"Error during loading: {e}")
                    print("WARNING: Could not load pretrained weights. Training with random init.")
            else:
                print(f"No key '{args.checkpoint_key}' found. Available keys:", list(checkpoint.keys()))
                print("WARNING: Could not load pretrained weights. Training with random init.")
        else:
            print(f"No pretrained weights found at {args.pretrained_weights}")
    else:
        utils.load_pretrained_weights(model, args.pretrained_weights,
                                      args.checkpoint_key, args.arch, args.patch_size)

    # freeze backbone
    for p in model.parameters(): p.requires_grad = False

    # segmentation head
    # head = SegmentationHead(embed_dim, args.num_classes).cuda() #simple head
    head = MambaDecoderHead(num_classes=args.num_classes).cuda()
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
    if args.multilabel:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index)

    # optionally resume
    to_restore = {"epoch": 0, "best_miou": 0., "best_dice": 0.}
    if args.load_from:
        utils.restart_from_checkpoint(os.path.join(args.output_dir, args.load_from),
                                      run_variables=to_restore,
                                      state_dict=head,
                                      optimizer=optimizer,
                                      scheduler=scheduler)
    start_epoch = to_restore["epoch"]
    best_miou   = to_restore["best_miou"]
    best_dice   = to_restore["best_dice"]
    selected = start_epoch > 0
    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        train_loss = train_seg(model, head, optimizer, train_loader, epoch, criterion)
        scheduler.step()

        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            metrics = validate_seg(val_loader, model, head, criterion, args)
            val_loss, miou, dice = metrics['loss'], metrics['miou'], metrics['dice']
            print(f"Epoch {epoch} Val Loss {val_loss:.4f} mIoU {miou:.4f} Dice {dice:.4f}")
            if 'dice_per_class' in metrics:
                print(f"  Per-class IoU:  {[f'{v:.4f}' for v in metrics['iou_per_class']]}")
                print(f"  Per-class Dice: {[f'{v:.4f}' for v in metrics['dice_per_class']]}")
                for name in ('precision', 'recall', 'diff_surface'):
                    values = metrics.get(f'{name}_per_class')
                    if values is not None:
                        print(f"  Per-class {name}: {values}")
            
            # Check if validation performance improved
            improved = checkpoint_improved(args.selection_metric, miou, dice, best_miou, best_dice, selected)
            if miou > best_miou:
                best_miou = miou
            if dice > best_dice:
                best_dice = dice
            
            # log
            log_stats = {**{'epoch': epoch, 'train_loss': train_loss['loss'] if isinstance(train_loss, dict) else train_loss},
                         **{'val_loss': float(val_loss), 'miou': float(miou), 'dice': float(dice), 'best_miou': float(best_miou), 'best_dice': float(best_dice)}}
            for name in ('class_names', 'iou_per_class', 'dice_per_class',
                         'precision_per_class', 'recall_per_class', 'diff_surface_per_class'):
                if name in metrics:
                    log_stats[name] = metrics[name]
            path_txt = os.path.join(args.output_dir, f"log_{args.log_name}.txt")
            with open(path_txt, 'a') as f:
                f.write(json.dumps(log_stats) + "\n")
            
            if improved:
                save_dict = {
                    'epoch': epoch+1,
                    'state_dict': head.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_miou': best_miou,
                    'best_dice': best_dice
                }
                save_dict.update(selection_metric=args.selection_metric, validation_metrics=metrics)
                torch.save(save_dict, os.path.join(args.output_dir, f"checkpoint_{args.checkpoint_key}_seg_best.pth"))
                write_json(Path(args.output_dir) / f"validation_results_{args.checkpoint_key}_best.json",
                           {"epoch": epoch + 1, "selection_metric": args.selection_metric, "validation_metrics": metrics})
                selected = True
                print(f"Saved best checkpoint at epoch {epoch} with mIoU: {miou:.4f}, Dice: {dice:.4f}")
            
            if (epoch + 1) % 10 == 0:
                save_dict = {
                    'epoch': epoch+1,
                    'state_dict': head.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_miou': best_miou,
                    'best_dice': best_dice
                }
                torch.save(save_dict, os.path.join(args.output_dir, f"checkpoint_{args.checkpoint_key}_seg_epoch_{epoch+1}.pth"))
                print(f"Saved checkpoint at epoch {epoch+1}")
            
            print(f"Max mIoU so far: {best_miou:.4f}")
            print(f"Max Dice so far: {best_dice:.4f}")
    print(f"Segmentation training completed. Best mIoU: {best_miou:.4f}, Best Dice: {best_dice:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with segmentation on medical images')
    parser.add_argument('--train_csv', default='./BUSBRA/BUSBRA/5-fold-cv.csv', type=str)
    parser.add_argument('--val_csv', default='./BUSBRA/BUSBRA/5-fold-cv.csv', type=str)
    parser.add_argument('--json_file', default='./tn3k/tn3k-trainval-fold0.json', type=str)
    parser.add_argument('--data_root', default='./BUSBRA/BUSBRA/Images/', type=str)
    parser.add_argument('--data_root2', default='./BUSBRA/BUSBRA/Masks/', type=str)
    parser.add_argument('--dataset_name', default='BUSBRA', type=str)
    parser.add_argument('--coco_json', default=str(CUSTOM_COCO_JSON), type=str,
                        help='COCO annotation file (dataset_name=CUSTOM)')
    parser.add_argument('--images_root', default=str(CUSTOM_IMAGES_ROOT), type=str,
                        help='Image directory (dataset_name=CUSTOM)')
    parser.add_argument('--split_file', default=str(CUSTOM_SPLIT_FILE), type=str,
                        help='train/val/test split json (dataset_name=CUSTOM)')
    parser.add_argument('--metadata_file', default=str(CUSTOM_METADATA_FILE), type=str,
                        help='Spreadsheet containing image_id and physical echelle')
    parser.add_argument('--multilabel', default=False, type=utils.bool_flag,
                        help='Multi-label segmentation: per-class sigmoid channels + BCE loss')
    parser.add_argument('--img_size', default=None, type=int,
                        help='Input image size (default: 512 for CUSTOM, 224 otherwise)')
    parser.add_argument('--n_last_blocks', default=4, type=int)
    parser.add_argument('--avgpool_patchtokens', default=0, choices=[0,1,2], type=int)
    parser.add_argument('--arch', default='vmamba_small', type=str,
                        choices=['vit_tiny','vit_small','vit_base','vit_large',
                                 'swin_tiny','swin_small','swin_base','swin_large',
                                 'resnet50','resnet101','vmamba_small'])
    parser.add_argument('--patch_size', default=4, type=int)
    parser.add_argument('--window_size', default=7, type=int)
    parser.add_argument('--pretrained_weights', default='', type=str)
    parser.add_argument('--checkpoint_key', default='teacher', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch_size_per_gpu', default=16, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--val_freq', default=1, type=int)
    parser.add_argument('--selection_metric', choices=['dice', 'miou', 'either'], default='either')
    parser.add_argument('--output_dir', default='.', type=str)
    parser.add_argument('--num_classes', default=None, type=int,
                        help='Decoder channels; inferred from CUSTOM COCO categories by default')
    parser.add_argument('--ignore_index', default=255, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--load_from', default=None, type=str)
    parser.add_argument('--log_name', default='seg', type=str)
    parser.add_argument("--pretrained_vmamba", default=False, type=utils.bool_flag)
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for ck in args.checkpoint_key.split(','):
        args_copy = copy.deepcopy(args)
        args_copy.checkpoint_key = ck
        print(f"Starting segmentation eval for key: {ck}")
        eval_seg(args_copy)
