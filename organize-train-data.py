import torch
import os 
from tqdm import tqdm
import numpy as np
import cv2
import random
import time
from torch.optim import Adam, SGD, RMSprop
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import segmentation_models_pytorch as smp
import optuna
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

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

class CombinedLoss(torch.nn.Module):
    def __init__(self, loss_a, loss_b, weight_a=1.0, weight_b=1.0):
        super(CombinedLoss, self).__init__()
        self.loss_a = loss_a
        self.loss_b = loss_b
        self.weight_a = weight_a
        self.weight_b = weight_b
    
    def forward(self, inputs, targets):
        return self.weight_a * self.loss_a(inputs, targets) + self.weight_b * self.loss_b(inputs, targets)

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
    return dice_scores[1:].mean()

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
    return iou.cpu().mean()

def train(model, optimizer, loss_fn, scheduler, train_loader, valid_loader, folder_name, num_epochs, device):
    writer = SummaryWriter(f'runs/{folder_name}')

    best_epoch_ckpt = 0
    best_valid_metric = float('-inf')
    best_valid_iou = float('-inf')
    best_iou_epoch = 0
    # Early stopping
    epochs_since_improvement = 0
    patience = 10

    start_time = time.time()
    model.to(device)
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
            train_dice_score.append(batch_dice_score.item())
            # update tqdm loop
            loop.set_postfix(train_loss=np.mean(train_loss), train_dice=np.mean(train_dice_score))
        
        avg_train_loss = sum(train_loss) / len(train_loss)
        avg_train_dice = sum(train_dice_score) / len(train_dice_score)
        
        # Validation phase
        model.eval()  # Set model to evaluation mode
        valid_loss = []
        valid_dice_scores = []
        valid_iou_scores = []
        # Choose a random batch to visualize
        valid_loop = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
        
        with torch.no_grad():
            for batch_idx, (data, targets) in valid_loop:
                data = data.float()
                data = data.to(device)
                targets = targets.to(device)
                targets = targets.type(torch.long)
                
                predictions = model(data)
                loss = loss_fn(predictions, targets)

                valid_loss.append(loss.item())

                preds = torch.argmax(predictions, dim=1)
                
                # Calculate and store metrics for this batch
                batch_dice_score = dice_score_multiclass(preds, targets, num_classes=6)
                valid_dice_scores.append(batch_dice_score.item())

                batch_iou_score = iou_per_class(preds, targets, num_classes=6)
                valid_iou_scores.append(batch_iou_score.item())

                # Update tqdm loop for validation with current average loss
                valid_loop.set_postfix(valid_loss=np.mean(valid_loss), valid_dice=np.mean(valid_dice_scores), valid_iou=np.mean(valid_iou_scores))
        
        # Compute average validation loss
        avg_valid_loss = sum(valid_loss) / len(valid_loss)

        # Compute overall metrics for the validation set
        avg_valid_dice = sum(valid_dice_scores) / len(valid_dice_scores)

        avg_valid_iou = sum(valid_iou_scores) / len(valid_iou_scores)

        scheduler.step(avg_valid_dice)

        if avg_valid_iou > best_valid_iou:
            best_valid_iou = avg_valid_iou
            best_iou_epoch = epoch

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
            print(f"Early stopping triggered after {epoch + 1} epochs due to no improvement in validation Dice score for {patience} consecutive epochs.")
            break

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
    print(f"Best iou score was {best_valid_iou:.4f} at the epoch {best_iou_epoch}")

    return best_valid_metric


def objective(trial):
    num_epochs = 10

    # Define encoder and pre-trained weights pairs
    encoder_weights_pairs = [
        "efficientnet-b5:imagenet",
        "timm-efficientnet-b5:imagenet",
        "timm-efficientnet-b5:advprop",
        "timm-efficientnet-b5:noisy-student"
    ]


    # Define the loss functions to try
    loss_functions = {
        'CrossEntropy': torch.nn.CrossEntropyLoss(),
        'DiceLoss': smp.losses.DiceLoss(mode='multiclass'),
        'FocalLoss': smp.losses.FocalLoss(mode='multiclass'),
        'ComboLoss': CombinedLoss(torch.nn.CrossEntropyLoss(), smp.losses.DiceLoss(mode='multiclass'))
    }

    # Define optimizers and their respective learning rates to try
    optimizer_configs = [
        "Adam_lr_0.001",
        "Adam_lr_0.0001",
        "SGD_lr_0.01",
        "SGD_lr_0.001",
        "RMSprop_lr_0.001",
        "RMSprop_lr_0.0001"
    ]

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    train_input_dir = "processed_dataset_with_borders/augmented_set/train/input"
    train_mask_dir = "processed_dataset_with_borders/augmented_set/train/gt"

    valid_input_dir = "processed_dataset_with_borders/augmented_set/valid/input"
    valid_mask_dir = "processed_dataset_with_borders/augmented_set/valid/gt"

    # Prepare dataset and dataloader
    train_dataset = CellSegmentationDataset(images_dir=train_input_dir, masks_dir=train_mask_dir, transform=train_transform)
    valid_dataset = CellSegmentationDataset(images_dir=valid_input_dir, masks_dir=valid_mask_dir, transform=valid_transform)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=10, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=10, pin_memory=True)

    encoder_weights_pair = trial.suggest_categorical('encoder_weights_pair', encoder_weights_pairs)
    # Splitting the string back into encoder name and weights
    encoder, weights = encoder_weights_pair.split(':')
    
    # Creating the model with the selected encoder
    model = smp.Unet(
        encoder_name=encoder, 
        encoder_weights=weights,
        in_channels=1,
        classes=6,
    )
    
    # Selecting loss function
    loss_fn_name = trial.suggest_categorical('loss_function', list(loss_functions.keys()))
    loss_fn = loss_functions[loss_fn_name]
    
    optimizer_config_str = trial.suggest_categorical('optimizer_config', optimizer_configs)
    
    # Now, safely split this string
    optimizer_name, lr_str = optimizer_config_str.split('_lr_')
    lr = float(lr_str)
    
    # Map the optimizer name to the actual optimizer class
    optimizer_class = {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
        "RMSprop": torch.optim.RMSprop
    }.get(optimizer_name)
    
    if optimizer_class is None:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    optimizer = optimizer_class(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)
    folder_name = f"{encoder_weights_pair}_{loss_fn_name}_{optimizer_name}_lr{lr}".replace(':', '_').replace(',', '_').replace(' ', '')

    print(f"""
        Encoder: {encoder} with {weights} weights.
        Loss function: {loss_fn_name}.
        Optimizer: {optimizer_name} with {lr_str} of learning rate.
        Using {device} device.
    """)
    # Running the training and validation loops
    dice_score = train(model, optimizer, loss_fn, scheduler, train_loader, valid_loader, folder_name, num_epochs, device)
    
    return dice_score

def main():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=5)  # Adjust n_trials to fit your computational budget

    best_trial = study.best_trial
    print(f"Best trial: Loss function {best_trial.params['loss_function']}, Encoder and weights {best_trial.params['encoder_weights_pair']}, Optimizer {best_trial.params['optimizer_config']['optimizer'].__name__}, Learning rate {best_trial.params['lr']}")

if __name__ == "__main__":
    main()