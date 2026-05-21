import os
import cv2
from tqdm import tqdm
import albumentations as A

def augment_and_save(input_image_path, mask_image_path, save_input_path, save_mask_path, augmentor):
    """
    Apply augmentations to an input image and its corresponding mask, then save both.
    """
    input_image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_UNCHANGED)
    
    # Apply the augmentation to both the input image and mask
    augmented = augmentor(image=input_image, mask=mask_image)
    augmented_input_image = augmented['image']
    augmented_mask = augmented['mask']
    
    # Save the augmented input image and mask
    cv2.imwrite(save_input_path, augmented_input_image)
    cv2.imwrite(save_mask_path, augmented_mask)

def create_augmented_dataset(dataset_dir, augmented_dataset_dir, subset='train'):
    """
    Create a new dataset by applying specified augmentations, keeping input images and masks aligned.
    """
    augmentor = A.Compose([
        A.Rotate(limit=180, p=1),  # Apply rotation to all images
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
        A.PiecewiseAffine(p=0.5),
    ], additional_targets={'mask': 'mask'})  # Specify that the mask should be treated as an image for augmentation purposes

    input_dir = os.path.join(dataset_dir, subset, 'input')
    gt_dir = os.path.join(dataset_dir, subset, 'gt')
    augmented_input_dir = os.path.join(augmented_dataset_dir, subset, 'input')
    augmented_gt_dir = os.path.join(augmented_dataset_dir, subset, 'gt')
    
    os.makedirs(augmented_input_dir, exist_ok=True)
    os.makedirs(augmented_gt_dir, exist_ok=True)
    
    # Iterate over input images and find corresponding masks
    for input_image_name in tqdm(os.listdir(input_dir), desc=f"Processing {subset}/input"):
        input_image_path = os.path.join(input_dir, input_image_name)
        mask_image_name = input_image_name.replace("_input.tif", "_gt.tif")
        mask_image_path = os.path.join(gt_dir, mask_image_name)
        
        save_input_path = os.path.join(augmented_input_dir, f"aug_{input_image_name}")
        save_mask_path = os.path.join(augmented_gt_dir, f"aug_{mask_image_name}")  # Corrected line
        
        augment_and_save(input_image_path, mask_image_path, save_input_path, save_mask_path, augmentor)

if __name__ == "__main__":
    dataset_dir = "dataset"
    augmented_dataset_dir = "dataset"  

    create_augmented_dataset(dataset_dir, augmented_dataset_dir)
    print("Augmentation completed and saved to:", augmented_dataset_dir)
