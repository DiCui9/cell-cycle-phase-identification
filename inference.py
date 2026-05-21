import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from scipy import ndimage

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(encoder_name="efficientnet-b5", encoder_weights=None, in_channels=1, classes=6)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
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
    labeled, num_features = ndimage.label(predictions)
    output_predictions = np.zeros_like(predictions)
    for component in range(1, num_features + 1):
        component_mask = (labeled == component)
        if np.any(component_mask):
            component_classes = predictions[component_mask]
            most_common_class = np.bincount(component_classes).argmax()
            output_predictions[component_mask] = most_common_class
    return output_predictions

def predict_image_with_postprocessing(model, device, img_tensor):
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        output = model(img_tensor)
    predicted_classes = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    predicted_classes = majority_voting_postprocess(predicted_classes)
    return predicted_classes

def create_timelapse(input_folder, output_folder, frame_rate=2):
    video_name = os.path.basename(output_folder) + "_timelapse.mp4"
    video_path = os.path.join(output_folder, video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_size = (1024, 2048)
    out_video = cv2.VideoWriter(video_path, fourcc, frame_rate, video_size)
    files_original = sorted([f for f in os.listdir(input_folder) if f.endswith('.tif')])
    files_predicted = sorted([f for f in os.listdir(output_folder) if f.endswith('.tif')])
    for file_original, file_predicted in zip(files_original, files_predicted):
        img1_path = os.path.join(input_folder, file_original)
        img2_path = os.path.join(output_folder, file_predicted)
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)
        img1 = (img1 / np.max(img1) * 255).astype(np.uint8)
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if img1.shape[:2] != (1024, 1024):
            img1 = cv2.resize(img1, (1024, 1024))
        if img2.shape[:2] != (1024, 1024):
            img2 = cv2.resize(img2, (1024, 1024))
        combined_img = np.vstack((img1, img2))
        out_video.write(combined_img)
    out_video.release()

def run_inference(input_folder, model_path):
    model, device = load_model(model_path)
    indices_to_colors = {
        0: (0, 0, 0),
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (255, 255, 0),
        4: (0, 0, 255),
        5: (255, 0, 255),
    }
    output_folder = input_folder + "_predicted"
    os.makedirs(output_folder, exist_ok=True)
    for image_name in os.listdir(input_folder):
        if image_name.lower().endswith('.tif'):
            img_path = os.path.join(input_folder, image_name)
            img_tensor = preprocess_image(img_path)
            predictions = predict_image_with_postprocessing(model, device, img_tensor)
            colored_image = colorize_predictions(predictions, indices_to_colors)
            save_path = os.path.join(output_folder, image_name)
            save_colored_image(colored_image, save_path)

def count_objects_by_color(img, color_to_class):
    counts = {class_idx: 0 for class_idx in range(1, 5)}
    for color, class_idx in color_to_class.items():
        if class_idx in counts:
            mask = cv2.inRange(img, np.array(color, dtype=np.uint8), np.array(color, dtype=np.uint8))
            num_objects, _ = cv2.connectedComponents(mask)
            counts[class_idx] = num_objects - 1
    return counts

def create_plot(input_folder, output_folder):
    plot_name = os.path.basename(output_folder) + "_plot.png"
    plot_path = os.path.join(output_folder, plot_name)
    color_to_class = {
        (0, 0, 255): 1,
        (0, 255, 0): 2,
        (0, 255, 255): 3,
        (255, 0, 0): 4,
    }
    if os.path.exists(input_folder + "_predicted"):
        cell_counts = []
        for image_name in sorted(os.listdir(input_folder + "_predicted")):
            if image_name.lower().endswith('.tif'):
                img_path = os.path.join(input_folder + "_predicted", image_name)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is not None:
                    counts = count_objects_by_color(img, color_to_class)
                    cell_counts.append(counts)
        time_points = range(len(cell_counts))
        class_counts = {i: [] for i in range(1, 5)}
        for count_dict in cell_counts:
            for i in range(1, 5):
                class_counts[i].append(count_dict.get(i, 0))
        fig, ax = plt.subplots()
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
        fig.savefig(plot_path)

def run_full_workflow(input_folder, model_path):
    output_folder = input_folder + "_predicted"
    os.makedirs(output_folder, exist_ok=True)
    run_inference(input_folder, model_path)
    create_timelapse(input_folder, output_folder)
    create_plot(input_folder, output_folder)

def main():
    if len(sys.argv) != 3:
        print("Usage: script.py <input_folder> <model_path>")
        sys.exit(1)
    input_folder = sys.argv[1]
    model_path = sys.argv[2]
    run_full_workflow(input_folder, model_path)

if __name__ == "__main__":
    main()
