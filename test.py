import os
import cv2
import time
import torch

import numpy as np
import albumentations as A
import torch.nn.functional as F
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from patchify import patchify, unpatchify
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_pil_image
from scipy.ndimage import label, binary_dilation, binary_erosion

class CellSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = os.listdir(images_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx]).replace('\\', '/')
        mask_filename = self.images[idx].replace("input", "gt")
        mask_path = os.path.join(self.masks_dir, mask_filename)
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = np.expand_dims(image, axis=0)
            image = torch.tensor(image, dtype=torch.float)
            mask = torch.tensor(mask, dtype=torch.long)

        return image, mask

def compute_metrics(preds, gt, num_classes):
    eps = 1e-8
    metrics = {}

    preds = preds.to(torch.int64)
    gt = gt.to(torch.int64)

    preds_one_hot = F.one_hot(preds, num_classes=num_classes).permute(0, 3, 1, 2).float()
    gt_one_hot = F.one_hot(gt, num_classes=num_classes).permute(0, 3, 1, 2).float()

    intersection = (preds_one_hot * gt_one_hot).sum(dim=(0, 2, 3))
    union = preds_one_hot.sum(dim=(0, 2, 3)) + gt_one_hot.sum(dim=(0, 2, 3))

    dice_scores = (2. * intersection + eps) / (union + eps)
    iou_scores = intersection / (union - intersection + eps)
    precision = intersection / (preds_one_hot.sum(dim=(0, 2, 3)) + eps)
    recall = intersection / (gt_one_hot.sum(dim=(0, 2, 3)) + eps)

    # Separating the metrics by class category
    metrics['iou'] = iou_scores.cpu().numpy().tolist()
    metrics['dice'] = dice_scores.cpu().numpy().tolist()
    metrics['precision'] = precision.cpu().numpy().tolist()
    metrics['recall'] = recall.cpu().numpy().tolist()

    return metrics

def compute_cell_counts(preds, gt, class_labels):
    cell_counts = {'predicted': {}, 'actual': {}}
    for label in class_labels:
        cell_counts['predicted'][label] = (preds == label).sum().item()
        cell_counts['actual'][label] = (gt == label).sum().item()
    return cell_counts

def compute_count_based_metrics(pred_counts, gt_counts):
    precision = {}
    recall = {}
    f1_score = {}

    for label in range(len(pred_counts)):
        tp = min(pred_counts[label], gt_counts[label])
        fp = pred_counts[label] - tp
        fn = gt_counts[label] - tp
        precision[label] = tp / (tp + fp + 1e-8)
        recall[label] = tp / (tp + fn + 1e-8)
        f1_score[label] = 2 * (precision[label] * recall[label]) / (precision[label] + recall[label] + 1e-8)

    return {'precision': np.mean(list(precision.values())), 
            'recall': np.mean(list(recall.values())), 
            'f1_score': np.mean(list(f1_score.values()))}

def format_metrics(metrics):
    return np.mean(metrics)

def print_metrics(test_set_size, avg_pixel_metrics, avg_count_metrics, cell_classes):
    print(f"Tested on {test_set_size} images\n")
    print("Average Pixel-level Metrics:")
    class_groups = {
        'Segmentation (Class 0)': [0],
        'Cell Classification (Classes 1 to 4)': [1, 2, 3, 4],
        'Borders (Class 5)': [5]
    }
    for metric, values in avg_pixel_metrics.items():
        print(f"{metric.capitalize()}:")
        for group_name, classes in class_groups.items():
            group_metrics = format_metrics([values[i] for i in classes])
            print(f"  {group_name}: {group_metrics:.4f}")
    
    if avg_count_metrics:
        print("\nAverage Count-based Metrics:")
        for metric, class_dict in avg_count_metrics.items():
            print(f"{metric.capitalize()}:")
            for group_name, classes in class_groups.items():
                if classes == cell_calsses:
                    if metric in ['precision', 'recall', 'f1_score']:  # These are count-based metrics
                        # Ensure only to access classes that are in the cell_classes
                        group_metrics = {f"Class {class_idx}": round(class_dict[class_idx], 4) 
                                        for class_idx in classes if class_idx in cell_classes and class_idx in class_dict}
                        print(f"  {group_name}: {group_metrics}")

def test_model(model, device, test_loader, num_classes, cell_classes):
    model.eval()
    overall_metrics = []
    overall_counts = []
    overall_count_metrics = []

    total_images = 0
    total_time = 0

    with torch.no_grad():
        for inputs, true_masks in test_loader:
            start_time = time.time()
            
            inputs = inputs.to(device)
            true_masks = true_masks.to(device)
            outputs = model(inputs)
            
            end_time = time.time()
            batch_time = end_time - start_time
            total_time += batch_time
            total_images += inputs.size(0)

            preds = torch.argmax(outputs, dim=1)

            pixel_metrics = compute_metrics(preds, true_masks, num_classes)
            overall_metrics.append(pixel_metrics)

            if cell_classes:
                counts = compute_cell_counts(preds, true_masks, cell_classes)
                count_metrics = compute_count_based_metrics(counts['predicted'], counts['actual'])
                overall_counts.append(counts)
                overall_count_metrics.append(count_metrics)

    # Aggregate metrics across all batches and for each class category
    avg_pixel_metrics = {key: np.mean([m[key] for m in overall_metrics], axis=0) for key in overall_metrics[0]}
    avg_count_metrics = {}
    if cell_classes:
        avg_count_metrics = {key: {} for key in overall_count_metrics[0]}
        for metric in avg_count_metrics:
            for class_key in overall_count_metrics[0][metric]:
                avg_count_metrics[metric][class_key] = np.mean([m[metric][class_key] for m in overall_count_metrics])

    test_set_size = len(test_loader.dataset)
    print_metrics(test_set_size, avg_pixel_metrics, avg_count_metrics, cell_classes)

    # Compute and print inference time in images per second
    images_per_second = total_images / total_time
    print(f'Inference Time: {images_per_second:.2f} images/second')

def apply_color_map(image_tensor, color_map):
    """ Apply a color map to a segmentation map. """
    colors = np.array([color_map[i] for i in range(len(color_map))])
    image_color = colors[image_tensor]  # Maps each pixel to its respective color
    return image_color

def relabel_regions_with_borders(preds, num_classes, color_map, border_color=[1, 0, 1]):
    labeled_array, num_features = label(preds)
    relabeled_preds = np.zeros_like(preds)

    for region in range(1, num_features + 1):
        mask = (labeled_array == region)
        if mask.sum() == 0:
            continue
        majority_class = np.bincount(preds[mask], minlength=num_classes).argmax()
        relabeled_preds[mask] = majority_class
    
    # Create a border mask using dilation and erosion
    border_mask = np.zeros_like(relabeled_preds, dtype=bool)
    for region in range(1, num_features + 1):
        mask = (labeled_array == region)
        dilated_mask = binary_dilation(mask, structure=np.ones((3, 3)))
        eroded_mask = binary_erosion(mask, structure=np.ones((3, 3)))
        border_mask |= (dilated_mask & ~eroded_mask)

    # Create an RGB version of the relabeled predictions and apply the border
    relabeled_preds_rgb = np.zeros((preds.shape[0], preds.shape[1], 3))
    for i in range(num_classes):
        relabeled_preds_rgb[relabeled_preds == i] = color_map[i]
    
    # Apply the magenta color to the borders
    relabeled_preds_rgb[border_mask] = border_color

    # Normalize the image to [0, 1] range
    relabeled_preds_rgb = np.clip(relabeled_preds_rgb, 0, 1)

    return relabeled_preds_rgb


def visualize_and_save_segmentation_samples(model, device, test_loader, num_samples, color_map, compute_metrics_fn, output_dir, file_name):
    model.eval()
    sample_indices = np.random.choice(len(test_loader.dataset), num_samples, replace=False)
    fig, axs = plt.subplots(nrows=num_samples, ncols=3, figsize=(15, 5 * num_samples))

    metrics_list = []
    metrics_text = ""

    with torch.no_grad():
        for idx, sample_idx in enumerate(sample_indices):
            inputs, true_masks = test_loader.dataset[sample_idx]
            inputs, true_masks = inputs.to(device), true_masks.to(device)
            outputs = model(inputs.unsqueeze(0))
            preds = torch.argmax(outputs, dim=1).squeeze(0)
            
            # Apply connected component labeling, relabel regions, and add borders
            relabeled_preds_with_borders = relabel_regions_with_borders(preds.cpu().numpy(), len(color_map), color_map)

            # Convert tensors to PIL images for easy visualization
            input_image = to_pil_image(inputs.cpu())
            gt_image = apply_color_map(true_masks.cpu().numpy(), color_map)

            # Compute metrics
            pred_counts = np.bincount(preds.cpu().numpy().flatten(), minlength=len(color_map))
            gt_counts = np.bincount(true_masks.cpu().numpy().flatten(), minlength=len(color_map))
            metrics = compute_metrics_fn(pred_counts, gt_counts)
            metrics_list.append(metrics)

            # Prepare metrics text for saving
            metrics_text += f"Sample {idx + 1} Metrics:\n"
            for metric, value in metrics.items():
                metrics_text += f"  {metric}: {value:.4f}\n"
            metrics_text += "\n"

            # Plot images
            axs[idx, 0].imshow(input_image, cmap='gray')
            axs[idx, 0].set_title('Input Image')
            axs[idx, 0].axis('off')

            axs[idx, 1].imshow(relabeled_preds_with_borders)
            axs[idx, 1].set_title('Prediction with Borders')
            axs[idx, 1].axis('off')

            axs[idx, 2].imshow(gt_image)
            axs[idx, 2].set_title('Ground Truth')
            axs[idx, 2].axis('off')

        plt.tight_layout()

        # Save plot as PNG
        plot_path = os.path.join(output_dir, f"{file_name}.png")
        plt.savefig(plot_path)
        plt.close(fig)

    # Save metrics to TXT file
    metrics_path = os.path.join(output_dir, f"{file_name}.txt")
    with open(metrics_path, 'w') as f:
        f.write(metrics_text)

    return metrics_list



def visualize_segmentation_samples_patches(model, device, test_loader, num_samples, color_map):
    model.eval()
    sample_indices = np.random.choice(len(test_loader.dataset), num_samples, replace=False)
    fig, axs = plt.subplots(nrows=num_samples, ncols=3, figsize=(15, 5 * num_samples))

    with torch.no_grad():
        for idx, sample_idx in enumerate(sample_indices):
            inputs, true_masks = test_loader.dataset[sample_idx]
            inputs, true_masks = inputs.to(device), true_masks.to(device)

            # Patchify the image
            patch_size = 512  # Define the size of each patch
            patches = patchify(inputs.cpu().numpy(), (1, patch_size, patch_size), step=patch_size//2)
            num_patches_height, num_patches_width = patches.shape[1], patches.shape[2]
            patches = patches.reshape(-1, 1, patch_size, patch_size)

            # Prepare tensor to store patch predictions
            pred_patches = []

            for i, patch in enumerate(patches):
                patch_tensor = torch.tensor(patch, dtype=torch.float).unsqueeze(0).to(device)
                output = model(patch_tensor)
                pred = torch.argmax(output, dim=1).squeeze(0)
                pred_patches.append(pred.cpu().numpy())

            # Reshape pred_patches to match the dimensions expected by unpatchify
            pred_patches = np.array(pred_patches)
            pred_patches = pred_patches.reshape((1, num_patches_height, num_patches_width, patch_size, patch_size))

            # Unpatchify predictions
            pred_image = unpatchify(pred_patches[0], inputs.cpu().numpy().shape[1:])  # Adjust indexing for batch dimension

            # Convert tensors to PIL images for visualization
            input_image = to_pil_image(inputs.cpu())
            pred_image = apply_color_map(pred_image, color_map)
            gt_image = apply_color_map(true_masks.cpu().numpy(), color_map)

            # Plot images
            axs[idx, 0].imshow(input_image, cmap='gray')
            axs[idx, 0].set_title('Input Image')
            axs[idx, 0].axis('off')

            axs[idx, 1].imshow(pred_image)
            axs[idx, 1].set_title('Prediction')
            axs[idx, 1].axis('off')

            axs[idx, 2].imshow(gt_image)
            axs[idx, 2].set_title('Ground Truth')
            axs[idx, 2].axis('off')

        plt.tight_layout()
        plt.show()

encoders_list = [("efficientnet-b5", 1024)]
for encoder, res in encoders_list:
    test_input_dir = "dataset/test/input"
    test_mask_dir = "dataset/test/gt"

    test_transform = A.Compose([
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()
    ])
    test_dataset = CellSegmentationDataset(test_input_dir, test_mask_dir, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, pin_memory=True, num_workers=0, shuffle=False)
    print("---------------------------------")
    print(f"Testing encoder: {encoder}")
    print("---------------------------------")
    model_path = f"models/good_models/UNET_{encoder}_CrossEntropyLoss_Adam_0.0001_{res}/best_model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=1, classes=6)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    
    model.to(device)

    num_classes = 6
    cell_calsses = [1, 2, 3, 4]

    indices_to_colors = {
            0: (0, 0, 0),       # Background/Other
            1: (255, 0, 0),     # Red
            2: (0, 255, 0),     # Green
            3: (255, 255, 0),   # Yellow
            4: (0, 0, 255),     # Blue
            5: (255, 0, 255),   # Magenta
        }

    # test_model(model, device, test_loader, num_classes, cell_calsses)
    output_dir = 'output'  # Replace with your desired directory
    file_name = 'segmentation_results'

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # visualize_segmentation_samples(model, device, test_loader, num_samples=2, color_map=indices_to_colors)
    metrics_list = visualize_and_save_segmentation_samples(model, device, test_loader, 2, indices_to_colors, compute_count_based_metrics, output_dir, file_name)

    # Print metrics for each sample
    for i, metrics in enumerate(metrics_list):
        print(f"Sample {i + 1} Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print("\n")


