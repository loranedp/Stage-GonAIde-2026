#!/usr/bin/env python3
import os
import argparse
import csv
import json
import copy
import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import nn
from pathlib import Path
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
from dataset.dataset_tn3k import TN3KDataset, TN3KTestDataset
from dataset.dataset_custom_coco import CocoMultiLabelDataset
from dataset.transforms import get_transforms, get_transforms_multilabel


class MambaDecoderHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.decoder = MambaDecoder(num_classes=num_classes)

    def forward(self, encoder_out, orig_hw):
        logits = self.decoder.forward(encoder_out)
        return F.interpolate(logits, size=orig_hw, mode='bilinear', align_corners=False)


def save_predicted_masks(preds, save_dir, mask_filenames, batch_idx, sample_offset=0, prefix="", max_samples=8,
                         images_root=None):
    batch_size = preds.shape[0]
    num_samples = min(batch_size, max_samples)
    
    for i in range(num_samples):
        try:
            if preds[i].ndim == 3:
                import cv2

                stem = (Path(mask_filenames[i]).stem if i < len(mask_filenames)
                        else f"mask_{sample_offset + i:04d}")
                target_size = (510, 380)
                if images_root and i < len(mask_filenames):
                    with Image.open(Path(images_root) / mask_filenames[i]) as image:
                        target_size = image.size
                width, height = target_size
                lines = []
                for class_id, channel in enumerate(preds[i]):
                    mask = Image.fromarray((channel.cpu().numpy() > 0.5).astype(np.uint8) * 255)
                    mask = np.array(mask.resize(target_size, Image.NEAREST))
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    for contour in contours:
                        if len(contour) < 3:
                            continue
                        coordinates = " ".join(f"{x / width:.6f} {y / height:.6f}"
                                               for x, y in contour.reshape(-1, 2))
                        lines.append(f"{class_id} {coordinates}")
                Path(save_dir, f"pred_{stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""))
                continue

            # Convert prediction to numpy and scale to 0-255 for saving as image
            pred_mask = preds[i].cpu().numpy()
            pred_mask = (pred_mask * 255).astype(np.uint8)  # Convert from 0-1 to 0-255

            # Create PIL Image and save
            if pred_mask.ndim == 3:
                # Multi-label prediction [C, H, W]: one class per RGB channel
                mask_image = Image.fromarray(pred_mask.transpose(1, 2, 0), mode='RGB')
            else:
                mask_image = Image.fromarray(pred_mask, mode='L')  # 'L' for grayscale
            
            # Use original mask filename (remove extension and add pred_ prefix)
            if i < len(mask_filenames):
                original_filename = mask_filenames[i]
                # Remove extension and add pred_ prefix
                filename_without_ext = os.path.splitext(original_filename)[0]
                filename = f"{filename_without_ext}.png"
            else:
                # Fallback to generic naming if filename not available
                global_sample_idx = sample_offset + i
                filename = f"pred_mask_{global_sample_idx:04d}.png"
            
            save_path = os.path.join(save_dir, filename)
            mask_image.save(save_path)
            
        except Exception as e:
            print(f"Warning: Failed to save mask for batch {batch_idx}, sample {i}: {e}")
            continue


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# index 0 = Cavite -> dark blue, index 1 = Gonade -> light blue, index 2 = Intestin -> violet
OVERLAY_COLORS = [(0, 0, 139), (135, 206, 250), (148, 0, 211), (255, 255, 0), (255, 0, 255), (0, 255, 255)]


def save_overlay_images(imgs, preds, save_dir, mask_filenames, batch_idx, sample_offset=0, alpha=0.4,
                         images_root=None):
    """Blend predicted masks over the input images and save as PNG.

    imgs: [B, 3, H, W] normalized image tensors (cpu), at the (square, resized) model input resolution.
    preds: [B, C, H, W] binary multi-label masks, or [B, H, W] class-index maps, same resolution as imgs.
    Class colors are indexed by class channel/index, see OVERLAY_COLORS above.

    When images_root is given and the original file for a sample can be found there, the overlay is
    drawn on that original (native-resolution, undistorted) image instead of the resized `imgs` tensor,
    with the prediction upsampled back to the original size via nearest-neighbor interpolation.
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    for i in range(preds.shape[0]):
        try:
            orig_image = None
            if images_root and i < len(mask_filenames):
                img_path = os.path.join(images_root, mask_filenames[i])
                if os.path.isfile(img_path):
                    orig_image = Image.open(img_path).convert('RGB')

            pred_i = preds[i:i + 1]  # keep batch dim for interpolate
            if orig_image is not None:
                target_hw = (orig_image.height, orig_image.width)
                if pred_i.dim() == 4:  # multi-label [1, C, H, W]
                    pred = F.interpolate(pred_i.float(), size=target_hw, mode='nearest')[0].numpy()
                else:  # argmax [1, H, W]
                    pred = F.interpolate(pred_i.unsqueeze(1).float(), size=target_hw, mode='nearest')[0, 0].numpy()
                image = np.asarray(orig_image, dtype=np.float64) / 255.0
            else:
                image = (imgs[i] * std + mean).clamp(0, 1).numpy().transpose(1, 2, 0)
                pred = preds[i].numpy()

            overlay = image.copy()

            if pred.ndim == 3:
                # Multi-label [C, H, W]: one binary mask per class
                class_masks = [(c, pred[c] > 0.5) for c in range(pred.shape[0])]
            else:
                # Argmax map [H, W]: skip background class 0
                class_masks = [(int(c) - 1, pred == c) for c in np.unique(pred) if c != 0]

            for c, m in class_masks:
                color = np.array(OVERLAY_COLORS[c % len(OVERLAY_COLORS)], dtype=np.float64) / 255.0
                overlay[m] = (1 - alpha) * overlay[m] + alpha * color

            out_image = Image.fromarray((overlay * 255).astype(np.uint8))

            if i < len(mask_filenames):
                filename = f"{os.path.splitext(mask_filenames[i])[0]}.png"
            else:
                filename = f"overlay_{sample_offset + i:04d}.png"
            out_image.save(os.path.join(save_dir, filename))

        except Exception as e:
            print(f"Warning: Failed to save overlay for batch {batch_idx}, sample {i}: {e}")
            continue


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


@torch.no_grad()
def test_seg(test_loader, model, head, criterion, args, save_predictions=False):
    model.eval()
    head.eval()
    
    total_iou = 0.0
    total_dice = 0.0
    total_loss = 0.0
    total_samples = 0  # Count samples, not batches
    total_iou_per_class = None
    total_dice_per_class = None
    
    all_predictions = []
    all_targets = []
    per_image_results = []

    mask_dir = None
    if args.save_masks:
        mask_dir = os.path.join(args.output_dir, f"predicted_masks_{args.checkpoint_key}")
        os.makedirs(mask_dir, exist_ok=True)
        print(f"Predicted masks will be saved to: {mask_dir}")
        print(f"NOTE: ALL test samples will have masks saved (estimated {len(test_loader) * args.batch_size_per_gpu} images)")

    overlay_dir = None
    if args.save_overlays:
        overlay_dir = os.path.join(args.output_dir, f"overlay_masks_{args.checkpoint_key}")
        os.makedirs(overlay_dir, exist_ok=True)
        print(f"Overlay images will be saved to: {overlay_dir}")

    print("Starting test evaluation...")
    
    for i, (imgs, masks, mask_filenames) in enumerate(test_loader):
        imgs = imgs.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)
        
        if args.arch == 'vmamba_small':
            output = model(imgs)
            # output = output[:, 1:]
        
        logits = head(output, orig_hw=imgs.shape[2:])
        
        loss = criterion(logits, masks)
        if args.multilabel:
            preds = (torch.sigmoid(logits) > 0.5).float()
        else:
            preds = logits.argmax(dim=1)

        batch_size = preds.shape[0]
        for j in range(batch_size):
            if args.multilabel:
                iou_pc = compute_iou_multilabel(preds[j:j+1], masks[j:j+1])
                dice_pc = compute_dice_multilabel(preds[j:j+1], masks[j:j+1])
                if total_iou_per_class is None:
                    total_iou_per_class = np.zeros(len(iou_pc))
                    total_dice_per_class = np.zeros(len(dice_pc))
                total_iou_per_class += np.array(iou_pc)
                total_dice_per_class += np.array(dice_pc)
                sample_iou = float(np.mean(iou_pc))
                sample_dice = float(np.mean(dice_pc))
            else:
                sample_iou = compute_iou(preds[j:j+1], masks[j:j+1])
                sample_dice = compute_dice(preds[j:j+1], masks[j:j+1])

            entry = {'file_name': mask_filenames[j] if j < len(mask_filenames) else f"sample_{total_samples:04d}",
                     'iou': sample_iou, 'dice': sample_dice}
            if args.multilabel:
                entry['iou_per_class'] = iou_pc
                entry['dice_per_class'] = dice_pc
            per_image_results.append(entry)

            total_iou += sample_iou
            total_dice += sample_dice
            total_samples += 1
        
        total_loss += loss.item() * batch_size
        
        if save_predictions:
            all_predictions.append(preds.cpu())
            all_targets.append(masks.cpu())
        
        if args.save_masks and mask_dir is not None:
            try:
                save_predicted_masks(
                    preds.cpu(), 
                    mask_dir, 
                    mask_filenames,  
                    i, 
                    sample_offset=total_samples - batch_size,  
                    prefix=f"{args.checkpoint_key}_",
                    max_samples=preds.shape[0],
                    images_root=getattr(args, "images_root", None),
                )
                
            except Exception as e:
                print(f"Warning: Failed to save masks for batch {i}: {e}")

        if overlay_dir is not None:
            save_overlay_images(
                imgs.cpu(),
                preds.cpu(),
                overlay_dir,
                mask_filenames,
                i,
                sample_offset=total_samples - batch_size,
                alpha=args.overlay_alpha,
                images_root=getattr(args, 'images_root', None),
            )

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(test_loader)} batches, "
                  f"Running IoU: {total_iou/total_samples:.4f}, "
                  f"Running Dice: {total_dice/total_samples:.4f}")
    
    avg_iou = total_iou / max(total_samples, 1)
    avg_dice = total_dice / max(total_samples, 1)
    avg_loss = total_loss / max(total_samples, 1)

    print(f"\n=== Test Results ===")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test mIoU: {avg_iou:.4f}")
    print(f"Test Dice: {avg_dice:.4f}")
    if args.multilabel and total_iou_per_class is not None:
        iou_per_class = (total_iou_per_class / max(total_samples, 1)).tolist()
        dice_per_class = (total_dice_per_class / max(total_samples, 1)).tolist()
        print(f"Per-class IoU:  {[f'{v:.4f}' for v in iou_per_class]}")
        print(f"Per-class Dice: {[f'{v:.4f}' for v in dice_per_class]}")
    print(f"Total samples: {total_samples}")
    
    if save_predictions and len(all_predictions) > 0:
        pred_save_path = os.path.join(args.output_dir, f"test_predictions_{args.checkpoint_key}.pt")
        torch.save({
            'predictions': torch.cat(all_predictions, dim=0),
            'targets': torch.cat(all_targets, dim=0),
            'metrics': {'loss': avg_loss, 'miou': avg_iou, 'dice': avg_dice}
        }, pred_save_path)
        print(f"Predictions saved to: {pred_save_path}")
    
    if args.save_masks and mask_dir is not None:
        print(f"Predicted masks saved to: {mask_dir}")
    if overlay_dir is not None:
        print(f"Overlay images saved to: {overlay_dir}")

    # per-image metrics CSV
    csv_path = os.path.join(args.output_dir, f"test_results_per_image_{args.checkpoint_key}_{args.cpk_name}.csv")
    num_pc = len(per_image_results[0].get('iou_per_class', [])) if per_image_results else 0
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file_name', 'iou', 'dice']
                        + [f'iou_class{c+1}' for c in range(num_pc)]
                        + [f'dice_class{c+1}' for c in range(num_pc)])
        for e in per_image_results:
            writer.writerow([e['file_name'], f"{e['iou']:.6f}", f"{e['dice']:.6f}"]
                            + [f"{v:.6f}" for v in e.get('iou_per_class', [])]
                            + [f"{v:.6f}" for v in e.get('dice_per_class', [])])
    print(f"Per-image results saved to: {csv_path}")

    metrics = {'loss': avg_loss, 'miou': avg_iou, 'dice': avg_dice}
    if args.multilabel and total_iou_per_class is not None:
        metrics['iou_per_class'] = iou_per_class
        metrics['dice_per_class'] = dice_per_class
    metrics['per_image'] = per_image_results
    return metrics


def load_model(args):
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
    return model, embed_dim


def load_pretrained_weights(model, args):
    if args.arch == 'vmamba_small':
        if os.path.isfile(args.pretrained_weights):
            print(f"Loading pretrained weights from {args.pretrained_weights}")
            checkpoint = torch.load(args.pretrained_weights, map_location="cpu", weights_only=False)
            
            if args.checkpoint_key is not None and args.checkpoint_key in checkpoint:
                print(f"Taking key {args.checkpoint_key} in checkpoint")
                state_dict = checkpoint[args.checkpoint_key]
                
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
                    print("WARNING: Could not load pretrained weights.")
            else:
                print(f"No key '{args.checkpoint_key}' found. Available keys:", list(checkpoint.keys()))
        else:
            print(f"No pretrained weights found at {args.pretrained_weights}")
    else:
        utils.load_pretrained_weights(model, args.pretrained_weights,
                                      args.checkpoint_key, args.arch, args.patch_size)


def load_complete_checkpoint(model, head, args):
    checkpoint_path = os.path.join(args.output_dir, f"checkpoint_{args.checkpoint_key}_seg_{args.cpk_name}.pth")
    if os.path.isfile(checkpoint_path):
        print(f"Loading complete model checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        if 'model_state_dict' in checkpoint and 'head_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            head.load_state_dict(checkpoint['head_state_dict'])
            print("Loaded complete model (backbone + head) from checkpoint")
        elif 'complete_model_state_dict' in checkpoint:
            complete_state_dict = checkpoint['complete_model_state_dict']
            
            backbone_state_dict = {}
            head_state_dict = {}
            
            for key, value in complete_state_dict.items():
                if key.startswith('backbone.'):
                    backbone_key = key[9:]  # Remove 'backbone.' prefix
                    backbone_state_dict[backbone_key] = value
                elif key.startswith('head.') or key.startswith('decoder.'):
                    head_key = key.replace('head.', '').replace('decoder.', 'decoder.')
                    head_state_dict[head_key] = value
            
            model.load_state_dict(backbone_state_dict, strict=False)
            head.load_state_dict(head_state_dict, strict=False)
            print("Loaded complete model from unified state dict")
        else:
            head.load_state_dict(checkpoint['state_dict'])
            print("Warning: Checkpoint only contains head weights, backbone uses pretrained weights")
            if args.pretrained_weights:
                load_pretrained_weights(model, args)
        
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
        print(f"Best training mIoU: {checkpoint.get('best_miou', 'N/A'):.4f}")
        print(f"Best training Dice: {checkpoint.get('best_dice', 'N/A'):.4f}")
        return checkpoint
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return None
    
    
def load_test_data(args):
    if args.img_size is None:
        args.img_size = 512 if args.dataset_name == 'CUSTOM' else 224
    print(f"Input image size: {args.img_size}x{args.img_size}")
    _, test_transform = get_transforms(img_size=args.img_size)

    if args.dataset_name == 'BUSBRA':
        organized_data_dir = "./US_DownStreamTask_datasets/BUSBRA/BUSBRA/organized_data"
        fold = '1_60p'  # Use fold 1
        test_ds = BUSBRADataset(organized_data_dir, fold, split='test', transform=test_transform)
    elif args.dataset_name == 'TN3K':
        test_ds = TN3KTestDataset(test_image_dir=args.data_root, test_mask_dir=args.data_root2, transform=test_transform)
    elif args.dataset_name == 'CUSTOM':
        _, test_transform = get_transforms_multilabel(img_size=args.img_size)
        test_ds = CocoMultiLabelDataset(coco_json=args.coco_json, images_dir=args.images_root, split_file=args.split_file, split=args.eval_split, transform=test_transform)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    
    test_loader = DataLoader(test_ds, batch_size=args.batch_size_per_gpu, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"Test data loaded: {len(test_ds)} samples.")
    return test_loader, test_ds


def main():
    """Main testing function"""
    parser = argparse.ArgumentParser('Test segmentation model on medical images')
    
    # Data arguments
    parser.add_argument('--test_csv', default='./BUSBRA/BUSBRA/5-fold-cv.csv', type=str)
    parser.add_argument('--data_root', default='./BUSBRA/BUSBRA/Images/', type=str)
    parser.add_argument('--data_root2', default='./BUSBRA/BUSBRA/Masks/', type=str)
    parser.add_argument('--dataset_name', default='BUSBRA', type=str, choices=['BUSBRA', 'TN3K', 'CUSTOM'])
    parser.add_argument('--coco_json', default='', type=str, help='COCO annotation file (dataset_name=CUSTOM)')
    parser.add_argument('--images_root', default='', type=str, help='Image directory (dataset_name=CUSTOM)')
    parser.add_argument('--split_file', default='', type=str, help='train/val/test split json (dataset_name=CUSTOM)')
    parser.add_argument('--eval_split', choices=['test', 'val'], default='test', help='CUSTOM split to evaluate')
    parser.add_argument('--results_dir', default='', help='Output directory separate from checkpoint directory')
    parser.add_argument('--multilabel', default=False, type=utils.bool_flag,
                        help='Multi-label segmentation: per-class sigmoid channels + BCE loss')
    parser.add_argument('--img_size', default=None, type=int,
                        help='Input image size (default: 512 for CUSTOM, 224 otherwise)')

    # Model arguments
    parser.add_argument('--arch', default='vmamba_small', type=str,
                        choices=['vit_tiny','vit_small','vit_base','vit_large',
                                 'swin_tiny','swin_small','swin_base','swin_large',
                                 'resnet50','resnet101','vmamba_small'])
    parser.add_argument('--patch_size', default=4, type=int)
    parser.add_argument('--window_size', default=7, type=int)
    parser.add_argument('--avgpool_patchtokens', default=0, choices=[0,1,2], type=int)
    parser.add_argument('--pretrained_weights', default='', type=str)
    parser.add_argument('--checkpoint_key', default='teacher', type=str)
    parser.add_argument("--pretrained_vmamba", default=False, type=utils.bool_flag)
    parser.add_argument('--load_complete_checkpoint', default=False, type=utils.bool_flag, 
                        help='Load complete model checkpoint (backbone+head). If False, loads backbone and head separately.')
    
    # Test arguments
    parser.add_argument('--output_dir', default='.', type=str, help='Directory containing the trained checkpoint')
    parser.add_argument('--batch_size_per_gpu', default=16, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--ignore_index', default=255, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--save_predictions', default=False, type=utils.bool_flag, help='Save test predictions to file')
    parser.add_argument('--save_masks', default=True, type=utils.bool_flag, help='Save predicted masks to file')
    parser.add_argument('--save_overlays', default=True, type=utils.bool_flag,
                        help='Save test images with predicted masks blended on top')
    parser.add_argument('--overlay_alpha', default=0.4, type=float,
                        help='Mask opacity in overlay images (0=invisible, 1=opaque)')
    parser.add_argument('--cpk_name', default='best', type=str, help='Name of the checkpoint')
    # Distributed arguments
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    
    args = parser.parse_args()
    
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join(f"{k}: {v}" for k,v in sorted(vars(args).items())))
    cudnn.benchmark = True
    utils.fix_random_seeds(args.seed)
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    test_loader, test_ds = load_test_data(args)

    model, embed_dim = load_model(args)

    head = MambaDecoderHead(num_classes=args.num_classes).cuda()
    if args.multilabel:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index)

    results = {}
    for ck in args.checkpoint_key.split(','):
        args_copy = copy.deepcopy(args)
        args_copy.checkpoint_key = ck.strip()
        
        print(f"\n{'='*60}")
        print(f"Testing checkpoint key: {args_copy.checkpoint_key}")
        print(f"{'='*60}")
        
        if args_copy.load_complete_checkpoint:
            checkpoint = load_complete_checkpoint(model, head, args_copy)
            if checkpoint is None:
                print(f"Skipping checkpoint key: {args_copy.checkpoint_key}")
                continue
        else:
            load_pretrained_weights(model, args_copy)
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_{args_copy.checkpoint_key}_seg_{args.cpk_name}.pth")
            if os.path.isfile(checkpoint_path):
                print(f"Loading head checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                head.load_state_dict(checkpoint['state_dict'])
                print(f"Loaded head checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
            else:
                print(f"No head checkpoint found at {checkpoint_path}")
                continue

        # Freeze all model parameters for testing
        for p in model.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = False
        print("All model parameters (backbone + head) frozen for testing.")

        checkpoint_directory = args_copy.output_dir
        if args.results_dir:
            args_copy.output_dir = args.results_dir
            Path(args_copy.output_dir).mkdir(parents=True, exist_ok=True)
        test_metrics = test_seg(test_loader, model, head, criterion, args_copy,
                               save_predictions=args.save_predictions)

        per_image = test_metrics.pop('per_image', None)
        results[args_copy.checkpoint_key] = test_metrics

        test_results = {
            'dataset': args.dataset_name,
            'checkpoint_key': args_copy.checkpoint_key,
            'checkpoint_path': os.path.join(checkpoint_directory, f"checkpoint_{args_copy.checkpoint_key}_seg_{args.cpk_name}.pth"),
            'eval_split': args.eval_split,
            'test_metrics': test_metrics,
            'num_test_samples': len(test_ds),
            'training_info': {
                'best_miou': checkpoint.get('best_miou', 'N/A'),
                'best_dice': checkpoint.get('best_dice', 'N/A'),
                'epoch': checkpoint.get('epoch', 'N/A')
            },
            'per_image': per_image
        }
        
        results_path = os.path.join(args_copy.output_dir, f"test_results_{args_copy.checkpoint_key}_{args.cpk_name}.json")
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"Test results saved to: {results_path}")
    
    print(f"\n{'='*60}")
    print("TESTING SUMMARY")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Total test samples: {len(test_ds)}")
    print(f"Architecture: {args.arch}")
    
    for ck, metrics in results.items():
        print(f"\nCheckpoint '{ck}':")
        print(f"  Test mIoU: {metrics['miou']:.4f}")
        print(f"  Test Dice: {metrics['dice']:.4f}")
        print(f"  Test Loss: {metrics['loss']:.4f}")
    
    if len(results) > 1:
        best_ck = max(results.keys(), key=lambda x: results[x]['miou'])
        print(f"\nBest checkpoint by mIoU: '{best_ck}' ({results[best_ck]['miou']:.4f})")
    
    print(f"\nAll results saved to: {args.results_dir or args.output_dir}")


if __name__ == '__main__':
    main() 
