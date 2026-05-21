import os
import cv2
import time
import torch
import random

import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from tqdm import tqdm
from albumentations.pytorch import ToTensorV2

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset

class CellSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = os.listdir(images_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx]).replace(os.pathsep, '/')
        mask_filename = self.images[idx].replace("input", "gt")
        mask_path = os.path.join(self.masks_dir, mask_filename).replace(os.pathsep, '/')

        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Read image in grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # Read mask as is
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = np.expand_dims(image, axis=0)  # Add channel dimension if needed
            image = torch.tensor(image, dtype=torch.float)  # Convert to float and scale
            mask = torch.tensor(mask, dtype=torch.long)  # Ensure mask is long for targets

        return image, mask
    
def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # Numpy module
    torch.manual_seed(seed_value)  # PyTorch
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # Python hash seed
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)  # Sets the seed for generating random numbers for the current GPU
        torch.cuda.manual_seed_all(seed_value)  # Sets the seed for generating random numbers on all GPUs
        torch.backends.cudnn.deterministic = True  # Ensures that CUDA selects the same convolution algorithm each time
        torch.backends.cudnn.benchmark = False  # False for reproducibility, True may improve performance

def visualize_sample(dataset):
    idx = random.randint(0, len(dataset) - 1)  # Select a random index
    image, mask = dataset[idx]  # Load the image and mask

    # If using ToTensorV2() in transformations, the image and mask tensors are CxHxW. Convert them to HxWxC for plotting.
    image = image.squeeze().cpu().detach().numpy()  # Assuming grayscale, remove channel dim and convert to numpy
    mask = mask.squeeze().cpu().detach().numpy()  # Convert to numpy, assuming single-channel mask

    if image.ndim == 3:
        image = np.transpose(image, (1, 2, 0))  # If the image has channels, move channels to the last dimension

    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image, cmap='gray')  # Display the image
    ax[0].set_title('Input Image')
    ax[0].axis('off')

    ax[1].imshow(mask, cmap='gray')  # Display the mask
    ax[1].set_title('GT Mask')
    ax[1].axis('off')

    plt.show()

def apply_color_map(mask, mapping):
    # Initialize an RGB image with the same height and width as the mask
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    for index, color in mapping.items():
        color_mask[mask == index] = color
    return color_mask

def plot_and_save_images(input_image, pred_mask, gt_mask, epoch, iou_scores, class_names, output_dir="output_images"):
    indices_to_colors = {
        0: (0, 0, 0),       # Background/Other
        1: (255, 0, 0),     # Red
        2: (0, 255, 0),     # Green
        3: (255, 255, 0),   # Yellow
        4: (0, 0, 255),     # Blue
        5: (255, 0, 255),   # Magenta
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    iou_str = ", ".join([f"{class_name}: {iou_scores[i]:.4f}" for i, class_name in enumerate(class_names)])
    
    # Input Image
    axs[0].imshow(input_image.squeeze(), cmap='gray')
    axs[0].set_title('Input Image')
    axs[0].axis('off')
    
    if pred_mask.ndim == 3:
        pred_mask = pred_mask.argmax(axis=0)
                                     
    pred_color_mask = apply_color_map(pred_mask, indices_to_colors)
    axs[1].imshow(pred_color_mask)
    axs[1].set_title('Predicted Mask')
    axs[1].axis('off')
    
    gt_color_mask = apply_color_map(gt_mask, indices_to_colors)
    axs[2].imshow(gt_color_mask)
    axs[2].set_title('GT Mask')
    axs[2].axis('off')
    
    plt.suptitle(f"IoU Scores by Class: {iou_str}", fontsize=10)
    plt.savefig(os.path.join(output_dir, f'epoch_{epoch}.png'))
    plt.close()

def dice_score_multiclass(preds, gt, num_classes):
    eps = 1e-8  # Small epsilon to avoid division by zero
    
    # Create a one-hot encoded version of preds and gt for all classes
    preds_one_hot = F.one_hot(preds, num_classes=num_classes).permute(0, 3, 1, 2).float()
    gt_one_hot = F.one_hot(gt, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    # Calculate intersection and union with broadcasting
    intersection = (preds_one_hot * gt_one_hot).sum(dim=(0, 2, 3))
    union = preds_one_hot.sum(dim=(0, 2, 3)) + gt_one_hot.sum(dim=(0, 2, 3))
    
    # Compute Dice score for each class
    dice_scores = (2. * intersection + eps) / (union + eps)
    
    # Return the average Dice score across all classes
    return dice_scores.mean()

def iou_per_class(y_true, y_pred, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_true, y_pred = y_true.to(device), y_pred.to(device)
    true_positive = torch.zeros(num_classes, dtype=torch.float32, device=device)
    false_positive = torch.zeros(num_classes, dtype=torch.float32, device=device)
    false_negative = torch.zeros(num_classes, dtype=torch.float32, device=device)
    
    for cls in range(num_classes):
        true_positive[cls] = torch.sum((y_pred == cls) & (y_true == cls))
        false_positive[cls] = torch.sum((y_pred == cls) & (y_true != cls))
        false_negative[cls] = torch.sum((y_pred != cls) & (y_true == cls))
    
    iou = true_positive / (true_positive + false_positive + false_negative + 1e-6)
    return iou.cpu()

def train(model, optimizer, loss_fn, train_loader, valid_loader, folder_name, num_epochs, device):
    writer = SummaryWriter(f'runs/good_runs/{folder_name}')

    best_epoch_ckpt = 0
    best_valid_metric = float('-inf')

    # Early stopping
    epochs_since_improvement = 0
    patience = 5

    start_time = time.time()

    for epoch in range(num_epochs):
        train_loss = []
        train_dice_score = []
        loop = tqdm(enumerate(train_loader),total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, (data, targets) in loop:
            if 0:
                print(f"Epoch {epoch+1}:")
                print(f"Input shape: {data.shape}")
                print(f"Input range: {data.min().item()} to {data.max().item()}")
                print(f"GT mask shape: {targets.shape}")            
                print(f"GT mask range: {targets.min().item()} to {targets.max().item()}")
            data = data.float()
            data = data.to(device)
            targets = targets.to(device)
            targets = targets.type(torch.long)

            # forward
            optimizer.zero_grad()
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            
            # backward
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            preds = torch.argmax(predictions, dim=1)
            batch_dice_score = dice_score_multiclass(preds, targets, num_classes=6)
            # batch_dice_score = dice_score_multiclass_v2(preds, targets, num_classes=6)
            train_dice_score.append(batch_dice_score.item())
            # update tqdm loop
            loop.set_postfix(train_loss=np.mean(train_loss), train_dice=np.mean(train_dice_score))
        
        avg_train_loss = sum(train_loss) / len(train_loss)
        avg_train_dice = sum(train_dice_score) / len(train_dice_score)
        
        # Validation phase
        model.eval()  # Set model to evaluation mode
        valid_loss = []
        valid_dice_scores = []
        # Choose a random batch to visualize
        random_batch_index = random.randint(0, len(valid_loader)-1)
        valid_loop = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
        
        with torch.no_grad():
            for batch_idx, (data, targets) in valid_loop:
                data = data.float()
                data = data.to(device)
                targets = targets.to(device)
                targets = targets.type(torch.long)
                
                # Squeeze the channel dimension from targets if necessary
                if targets.dim() == 4 and targets.shape[1] == 1:
                    targets = targets.squeeze(1)  # This changes shape from [B, 1, H, W] to [B, H, W]

                predictions = model(data)
                loss = loss_fn(predictions, targets)

                valid_loss.append(loss.item())

                preds = torch.argmax(predictions, dim=1)
                
                # Calculate and store metrics for this batch
                batch_dice_score = dice_score_multiclass(preds, targets, num_classes=6)
                # batch_dice_score = dice_score_multiclass_v2(preds, targets, num_classes=6)
                valid_dice_scores.append(batch_dice_score.item())

                # Update tqdm loop for validation with current average loss
                valid_loop.set_postfix(valid_loss=np.mean(valid_loss), valid_dice=np.mean(valid_dice_scores))
                
                # Visualize and save images for the randomly selected batch
                if batch_idx == random_batch_index:
                    input_image_np = data[0].cpu().squeeze().numpy()
                    pred_mask_np = preds[0].cpu().numpy()
                    gt_mask_np = targets[0].cpu().numpy()
                    
                    class_names = ['Background', 'Red', 'Green', 'Yellow', 'Blue', 'Magenta']

                    iou_scores = iou_per_class(targets[0], preds[0], num_classes=6)
                    print("\nIoU scores by Class:")
                    for i in range(len(class_names)):  # Iterate directly over class names
                        print(f"{class_names[i]}: {iou_scores[i].item():.4f}")  # Access scores using class names as keys
                    print("---------------------------")

                    # Adjust function as necessary for visualization
                    plot_and_save_images(input_image_np, pred_mask_np, gt_mask_np, epoch, 
                                        iou_scores, class_names, output_dir=f'output_images/{folder_name}')
        
        # Compute average validation loss
        avg_valid_loss = sum(valid_loss) / len(valid_loss)

        # Compute overall metrics for the validation set
        avg_valid_dice = sum(valid_dice_scores) / len(valid_dice_scores)

        # Update and save the best model based on validation dice score
        if avg_valid_dice > best_valid_metric:
            best_valid_metric = avg_valid_dice
            best_epoch_ckpt = epoch
            epochs_since_improvement = 0 
            # Save model
            model_save_directory = os.path.join('models', folder_name)
            os.makedirs(model_save_directory, exist_ok=True)
            model_save_path = os.path.join(model_save_directory, f'best_model.pth')
            torch.save(model.state_dict(), model_save_path)
        else:
            epochs_since_improvement += 1
        
        # Check if early stopping condition is met
        if epochs_since_improvement >= patience:
            model_save_path = os.path.join(model_save_directory, f'last_model.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"Early stopping triggered after {epoch + 1} epochs due to no improvement in validation Dice score for {patience} consecutive epochs.")
            break

        print("#### Loss and Metrics ####")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_valid_loss:.4f}")
        print(f"Average Training Dice Score: {avg_train_dice:.4f}")
        print(f"Average Validation Dice Score: {avg_valid_dice:.4f}")
        print("---------------------------")

        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_valid_loss, epoch)
        writer.add_scalar('DiceScore/Train', avg_train_dice, epoch)
        writer.add_scalar('DiceScore/Validation', avg_valid_dice, epoch)

    writer.close()
    end_time = time.time()  # End time
    total_time = end_time - start_time  # Calculate total time taken

    # Convert seconds to hours, minutes, and seconds
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = total_time % 60

    print(f"Training completed in: {int(hours)}h:{int(minutes)}min:{int(seconds)}s")
    print(f"Best model checkpoint saved in epoch {best_epoch_ckpt} with {best_valid_metric:.4f} of dice score")

def compute_class_weights(dataloader):
    """
    Compute class weights inversely proportional to the frequency of each class
    for a dataset with a known number of classes, ensuring all class indices are valid.
    
    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        
    Returns:
        torch.Tensor: Tensor of class weights.
    """
    num_classes = 6  # Adjust based on your dataset specifics
    class_counts = torch.zeros(num_classes, dtype=torch.long)

    for batch_index, (_, mask) in enumerate(dataloader):
        mask = mask.squeeze()  # Remove batch dimension as batch_size=1
        labels, counts = torch.unique(mask, return_counts=True)

        # Convert labels to the correct data type for indexing
        labels = labels.long()  # Ensure labels are long for indexing

        # print(f"Batch {batch_index}: Labels found - {labels}")
        # print(f"Batch {batch_index}: Counts for each label - {counts}")

        class_counts[labels] += counts

    # Avoid division by zero for classes that do not appear
    class_counts[class_counts == 0] = 1
    total_counts = class_counts.sum()
    
    # Compute weights as inverse of frequency
    class_weights = total_counts / class_counts
    
    # Normalize weights such that the smallest weight is 1
    class_weights = class_weights / class_weights.min()
    
    return class_weights

def main():
    set_seed(200823)

    train_transform = A.Compose([
        A.OneOf([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1)
        ], p=0.5),
        A.Perspective(scale=(0.05, 0.1), p=0.5),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})
    
    valid_transform = A.Compose([
        A.Normalize(mean=0.0, std=1.0),  # Normalizes the image with mean 0 and standard deviation 1
        ToTensorV2()  # Converts the image and mask to PyTorch tensors
    ])

    train_input_dir = "dataset/train/input"
    train_mask_dir = "dataset/train/gt"

    valid_input_dir = "dataset/valid/input"
    valid_mask_dir = "dataset/valid/gt"

    ENCODER = 'efficientnet-b4'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['Background', 'Red', 'Green', 'Yellow', 'Blue', 'Borders']

    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(CLASSES), 
        in_channels=1,                  
    )

    train_dataset = CellSegmentationDataset(
        train_input_dir, 
        train_mask_dir, 
        transform=train_transform,
    )

    valid_dataset = CellSegmentationDataset(
        valid_input_dir, 
        valid_mask_dir, 
        transform=valid_transform,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=8, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=8, shuffle=False)

    # weights = compute_class_weights(train_loader)
    # print("Computed Class Weights:", weights)

    # Visualize input and gt data
    # visualize_sample(train_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the weighted loss function
    # loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    loss_fn = torch.nn.CrossEntropyLoss()

    num_epochs = 100
    lr = 0.0001
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    model.to(device)

    loss_fn_name = type(loss_fn).__name__
    optimizer_name = type(optimizer).__name__

    # Create the folder name
    folder_name = f"UNET_{ENCODER}_{loss_fn_name}_{optimizer_name}_{lr}_1024" 

    train(model, optimizer, loss_fn, train_loader, valid_loader, folder_name, num_epochs, device)

if __name__ == '__main__':
    main()