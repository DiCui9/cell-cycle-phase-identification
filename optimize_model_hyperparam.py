import os
import sys
import random
import cv2
import torch
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
from scipy import ndimage
import matplotlib.pyplot as plt

# Define the model loading and prediction function
def load_model(model_path):
    model = smp.Unet(encoder_name="efficientnet-b5", encoder_weights=None, in_channels=1, classes=6)
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, device

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Load image in grayscale
    input_image = (img / np.max(img) * 255).astype(np.uint8)
    if input_image.shape != (1024, 1024):
        input_image = cv2.resize(input_image, (1024, 1024))  # Resize image to 1024x1024 if not already
    img_array = np.array(input_image, dtype=np.float32) / 255.0  # Normalize
    img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    return img_tensor

def colorize_predictions(predictions, indices_to_colors):
    colored_image = np.zeros((predictions.shape[0], predictions.shape[1], 3), dtype=np.uint8)
    for class_index, color in indices_to_colors.items():
        colored_image[predictions == class_index] = color
    return colored_image

def save_colored_image(img_array, save_path):
    cv2.imwrite(save_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV

def majority_voting_postprocess(predictions):
    # Find connected components
    labeled, num_features = ndimage.label(predictions)
    output_predictions = np.zeros_like(predictions)

    # Apply majority voting for each connected component
    for component in range(1, num_features + 1):
        component_mask = (labeled == component)
        if np.any(component_mask):
            # Find the most common class in this component
            component_classes = predictions[component_mask]
            most_common_class = np.bincount(component_classes).argmax()
            output_predictions[component_mask] = most_common_class

    return output_predictions

# Modify the predict_image function to include post-processing
def predict_image_with_postprocessing(model, device, img_tensor):
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        output = model(img_tensor)
        predicted_classes = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        # Apply majority voting post-processing
        predicted_classes = majority_voting_postprocess(predicted_classes)
    return predicted_classes

def create_timelapse(input_folder, output_folder):
    files_original = sorted([f for f in os.listdir(input_folder) if f.endswith('.tif')])
    files_predicted = sorted([f for f in os.listdir(output_folder) if f.endswith('.tif')])

    frames = []
    for file_original, file_predicted in zip(files_original, files_predicted):
        img1_path = os.path.join(input_folder, file_original)
        img2_path = os.path.join(output_folder, file_predicted)
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)

        img1 = (img1 / np.max(img1) * 255).astype(np.uint8)
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)  # Convert grayscale to color

        if img1.shape[:2] != (1024, 1024):
            img1 = cv2.resize(img1, (1024, 1024))
        if img2.shape[:2] != (1024, 1024):
            img2 = cv2.resize(img2, (1024, 1024))

        combined_img = np.vstack((img1, img2))
        frames.append(combined_img)

    return frames

def run_inference(input_folder, model_path):
    model, device = load_model(model_path)
    indices_to_colors = {
        0: (0, 0, 0),       # Background/Other
        1: (255, 0, 0),     # Red
        2: (0, 255, 0),     # Green
        3: (255, 255, 0),   # Yellow
        4: (0, 0, 255),     # Blue
        5: (255, 0, 255),   # Magenta
    }

    output_folder = input_folder + "_predicted"
    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.tif')]
    total_files = len(image_files)

    for i, image_name in enumerate(image_files):
        img_path = os.path.join(input_folder, image_name)
        img_tensor = preprocess_image(img_path)
        predictions = predict_image_with_postprocessing(model, device, img_tensor)
        colored_image = colorize_predictions(predictions, indices_to_colors)
        save_path = os.path.join(output_folder, image_name)
        save_colored_image(colored_image, save_path)

def run_full_workflow(input_folder, model_path):
    output_folder = input_folder + "_predicted"
    os.makedirs(output_folder, exist_ok=True)
    run_inference(input_folder, model_path)
    return create_timelapse(input_folder, output_folder)

def count_objects_by_color(img, color_to_class):
    counts = {class_idx: 0 for class_idx in range(1, 5)}
    for color, class_idx in color_to_class.items():
        if class_idx in counts:
            # Convert the color to numpy array in BGR format
            mask = cv2.inRange(img, np.array(color, dtype=np.uint8), np.array(color, dtype=np.uint8))
            num_objects, _ = cv2.connectedComponents(mask)
            counts[class_idx] = num_objects - 1  # subtract 1 to ignore the background component
    return counts

def group_and_compare(df):
    grouped = df.groupby('treatment')
    results = {}
    for treatment, group in grouped:
        fucci = group[group['cell_line'].str.contains('FUCCI')]['location']
        wt = group[group['cell_line'].str.contains('WT')]['location']
        results[treatment] = {'FUCCI': fucci.tolist(), 'WT': wt.tolist()}
    return results

def plot_and_save(cell_counts, output_filename):
    # Prepare data for stacked bar chart
    time_points = range(len(cell_counts))
    class_counts = {i: [] for i in range(1, 5)}  # Only classes 1, 2, 3, and 4

    for count_dict in cell_counts:
        for i in range(1, 5):
            class_counts[i].append(count_dict.get(i, 0))
    
    fig, ax = plt.subplots()

    # Plot stacked bar chart
    bottom = np.zeros(len(time_points))
    colors = ['red', 'green', 'yellow', 'blue']
    class_names = ['G1 Phase', 'S/G2/M Phase', 'Early S', 'Hoechst']

    for class_index, color, name in zip(range(1, 5), colors, class_names):
        counts = class_counts[class_index]
        ax.bar(time_points, counts, bottom=bottom, color=color, label=name)
        bottom += np.array(counts)

    ax.set_xlabel('Time')
    ax.set_ylabel('Cell Count')
    ax.legend()
    plt.savefig(output_filename)
    plt.close()

def process_group_02_03(input_folder, output_filename, indices_to_colors, color_to_class):
    cell_counts = []

    for root, _, files in os.walk(input_folder):
        for file_name in sorted(files):
            if file_name.lower().endswith('.tif'):
                img_path = os.path.join(root, file_name)
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    colored_img = colorize_predictions(img, indices_to_colors)
                    counts = count_objects_by_color(colored_img, color_to_class)
                    cell_counts.append(counts)

    plot_and_save(cell_counts, output_filename)

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <treatment> <input_folder_base>")
        sys.exit(1)

    treatment = sys.argv[1]
    input_folder_base = sys.argv[2]
    model_path = "models/good_models/UNET_efficientnet-b5_CrossEntropyLoss_Adam_0.0001_1024/best_model.pth"

    # Sample DataFrame creation for demonstration
    data = {
        'location': ['B02', 'B03', 'B04', 'B05', 'C02', 'C03', 'C04', 'C05', 'D02', 'D03', 'D04', 'D05', 'E02', 'E03', 'E04', 'E05', 'F02', 'F03', 'F04', 'F05', 'G02', 'G03', 'G04', 'G05'],
        'cell_line': ['Hs578t_FUCCI', 'Hs578t_FUCCI', 'Hs578t_WT', 'Hs578t_WT', 'Hs578t_FUCCI', 'Hs578t_FUCCI', 'Hs578t_WT', 'Hs578t_WT', 'Hs578t_FUCCI', 'Hs578t_FUCCI', 'Hs578t_WT', 'Hs578t_WT', 'Hs578t_FUCCI', 'Hs578t_FUCCI', 'Hs578t_WT', 'Hs578t_WT', 'Hs578t_FUCCI', 'Hs578t_FUCCI', 'Hs578t_WT', 'Hs578t_WT', 'Hs578t_FUCCI', 'Hs578t_FUCCI', 'Hs578t_WT', 'Hs578t_WT'],
        'treatment': ['DMSO', 'DMSO', 'DMSO', 'DMSO', 'nocudazole', 'nocudazole', 'nocudazole', 'nocudazole', 'palbociclib', 'palbociclib', 'palbociclib', 'palbociclib', 'flavopiridol', 'flavopiridol', 'flavopiridol', 'flavopiridol', 'bosutinib', 'bosutinib', 'bosutinib', 'bosutinib', 'cisplatin', 'cisplatin', 'cisplatin', 'cisplatin']
    }
    df = pd.DataFrame(data)
    grouped_data = group_and_compare(df)

    if treatment not in grouped_data:
        print(f"Treatment '{treatment}' not found in the dataset.")
        sys.exit(1)

    indices_to_colors = {
        0: (0, 0, 0),       # Background/Other
        1: (255, 0, 0),     # Red
        2: (0, 255, 0),     # Green
        3: (255, 255, 0),   # Yellow
        4: (0, 0, 255),     # Blue
        5: (255, 0, 255),   # Magenta
    }

    color_to_class = {
        (0, 0, 255): 1,     # Red in BGR
        (0, 255, 0): 2,     # Green
        (0, 255, 255): 3,   # Yellow in BGR
        (255, 0, 0): 4,     # Blue in BGR
    }

    for cell_line, locations in grouped_data[treatment].items():
        random_location = random.choice(locations)
        random_zone = f"{random_location}_s{random.randint(1, 4)}"
        group_02_03_folder = os.path.join(input_folder_base, "group_02_03", random_zone)
        group_04_05_folder = os.path.join(input_folder_base, "group_04_05", random_zone)

        if os.path.exists(group_02_03_folder):
            output_filename = f"{input_folder_base}/{treatment}-{cell_line}_{random_zone}.png"
            process_group_02_03(group_02_03_folder, output_filename, indices_to_colors, color_to_class)
            print(f"Processed and saved plot for {random_zone}: {output_filename}")

        if os.path.exists(group_04_05_folder):
            frames = run_full_workflow(group_04_05_folder, model_path)
            print(f"Workflow completed for {treatment} - {cell_line} in group_04_05!")

            cell_counts = []
            if os.path.exists(group_04_05_folder + "_predicted"):
                for image_name in sorted(os.listdir(group_04_05_folder + "_predicted")):
                    if image_name.lower().endswith('.tif'):
                        img_path = os.path.join(group_04_05_folder + "_predicted", image_name)
                        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                        if img is not None:
                            counts = count_objects_by_color(img, color_to_class)
                            cell_counts.append(counts)

                output_filename = f"{input_folder_base}/{treatment}-{cell_line}__{random_zone}.png"
                plot_and_save(cell_counts, output_filename)
                print(f"Plot saved for _{random_zone}: {output_filename}")

if __name__ == "__main__":
    main()
