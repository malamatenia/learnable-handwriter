import numpy as np
import pandas as pd
import os
import json
import shutil
from pathlib import Path
import rootutils
import sys 

import cv2
import PIL
from PIL import Image, ImageOps
from scipy.ndimage import binary_dilation, label, gaussian_filter
from IPython.display import display
import matplotlib.gridspec as gridspec


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm


def horzpil(images, width, height, margin=2):
    """Concatenate images horizontally with a small margin between them."""
    images = [PIL.Image.open(image_path) for image_path in images]
    
    total_width = width * len(images) + margin * (len(images) - 1)
    max_height = height
    new_im = PIL.Image.new('RGB', (total_width, max_height), color=(255, 255, 255))
    
    x_offset = 0
    for img in images:
        new_im.paste(img, (x_offset, 0))
        x_offset += width + margin
    return new_im

def folder_to_grid(folder_paths, folder_order, character_index, image_width, image_height, margin=2):
    """Convert images in multiple folders to a horizontal grid based on the specified order."""
    grid_images = []
    
    for folder_name in folder_order:
        folder_path = os.path.join(folder_paths, folder_name)
        print(f"Processing folder: {folder_path}")  
        folder_images = []
        
        # Iterate through image names in specified order for each folder
        for i in character_index:
            image_path = os.path.join(folder_path, f"{i}.png")
            if os.path.exists(image_path):
                folder_images.append(image_path)
                #print(f"Taking image from {folder_name}: {i}.png")
                
        # Check if folder_images is not empty
        if folder_images:
            # Concatenate images horizontally for the current folder with a margin
            folder_grid = horzpil(folder_images, image_width, image_height, margin=margin)
            grid_images.append(folder_grid)
    
    # Concatenate grid images vertically to form the final grid if there are images
    if grid_images:
        final_grid = PIL.Image.new('RGB', (grid_images[0].width, sum(grid_img.height for grid_img in grid_images)), color=(255, 255, 255))
        y_offset = 0
        for grid_img in grid_images:
            final_grid.paste(grid_img, (0, y_offset))
            y_offset += grid_img.height
    else:
        final_grid = PIL.Image.new('RGB', (1, 1), color=(255, 255, 255))  # Create a blank white image
    
    display(final_grid)
    return final_grid


    # Custom colormap
cmap = LinearSegmentedColormap.from_list('custom_cmap', ['blue', 'white', 'red'], N=256)

def subtract_images(image1, image2, axes, opacity=0.5):
    result = image1.astype(float) - image2.astype(float)

    # Display the result on the third subplot
    axes[2].imshow(result, cmap=cmap, vmin=-255, vmax=255, alpha=opacity)
    axes[2].axis('off') 
    
    return result

def visualize_comparison(folder1, folder2, character_order, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for character in character_order:
        image1_path = os.path.join(folder1, f"{character}.png")
        image2_path = os.path.join(folder2, f"{character}.png")

        if not os.path.exists(image1_path):
            print(f"Image 1 not found: {image1_path}")
            continue
        if not os.path.exists(image2_path):
            print(f"Image 2 not found: {image2_path}")
            continue

        image1 = np.array(Image.open(image1_path).convert('L'))
        image2 = np.array(Image.open(image2_path).convert('L'))

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        
        axes[0].imshow(image1, cmap='gray', vmin=0, vmax=255)
        axes[0].axis('off')

        axes[1].imshow(image2, cmap='gray', vmin=0, vmax=255)
        axes[1].axis('off')

        result = subtract_images(image1, image2, axes, opacity=0.5)
        result_path = os.path.join(output_folder, f"{character}.png")
        plt.imsave(result_path, result, cmap=cmap, vmin=-255, vmax=255)

        visual_comparison_path = os.path.join(output_folder, f"visual_comparison_character_{character}.png")
        plt.savefig(visual_comparison_path)
        
        plt.tight_layout()
        plt.show()

def calculate_l2_norm(image1, image2):
    array1 = np.array(image1).astype(float) / 255.0 
    array2 = np.array(image2).astype(float) / 255.0 
    difference = array1 - array2
    l2_norm = np.sqrt(np.sum(difference**2))
    return l2_norm


# Flatten image into a vector
def flatten_image(image):
    return image.flatten() / 255.0  # Normalize pixel values to [0, 1]

# Calculate standard deviation of pixel values for a directory that contains all the document prototypes of a class
def calculate_metric_for_directory(directory_path, class_prefix, codename_to_id):
    if not os.path.isdir(directory_path):
        print("Error: Directory does not exist.")
        return None

    image_names = []
    stds = []

    character_range = list(range(28, 37)) + list(range(39, 49))

    for image_number in character_range:
        images = []

        for codename, doc_id in codename_to_id.items():
            if codename.startswith(class_prefix):
                subfolder_path = os.path.join(directory_path, doc_id)
                image_path = os.path.join(subfolder_path, f"{image_number}.png")

                if os.path.isfile(image_path):
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                    if image is not None:
                        flattened_image = flatten_image(image)
                        images.append(flattened_image)

        if images:  # Only compute if we found valid images
            images = np.array(images)
            variance = np.var(images, axis=0, ddof=1)
            std = np.sqrt(np.sum(variance))
        else:
            std = np.nan  # Or handle differently if needed

        image_names.append(f"Image {image_number}.png")
        stds.append(std)

    df = pd.DataFrame({'Image': image_names, 'Std': stds})
    return df

def merge_dataframes(st_df, nt_df):
    # Merge the DataFrames on the 'Image' column
    merged_df = pd.merge(st_df, nt_df, on='Image', suffixes=('_ST', '_NT'))
    return merged_df


# Files to ignore
ignore_files = {'grid-1l.png', 'grid.png', 'model.pth', 'config.yaml', 'transcribe.json'}
character_range = list(range(28, 37)) + list(range(39, 49)) #this range corresponds to the characters from a-u excluding j/k

def select_images(in_path: Path, out_path: Path):
    out_path.mkdir(parents=True, exist_ok=True)

    for file in in_path.iterdir():
        if file.suffix == '.png' and file.name not in ignore_files:
            try:
                file_index = int(file.stem)
            except ValueError:
                continue
            if file_index in character_range:
                shutil.copy2(file, out_path / file.name)

def selected_protos_directory(source_directory: Path, destination_root: Path):
    for folder in source_directory.iterdir():
        if folder.is_dir():
            new_folder_path = destination_root / folder.name
            new_folder_path.mkdir(parents=True, exist_ok=True)

            sprites_final_path = folder / 'sprites' / 'final'
            baseline_path = folder / 'baseline'

            if sprites_final_path.exists():
                select_images(sprites_final_path, new_folder_path)
            elif baseline_path.exists():
                select_images(baseline_path, new_folder_path)
            else:
                select_images(folder, new_folder_path)


def filter_prototypes(
    input_dir: Path, 
    output_dir: Path, 
    baseline_dir: Path, 
    threshold: float
):
    """
    Filters sprite images in all subfolders of `input_dir` using baseline masks from `baseline_dir`,
    and saves the result to corresponding subfolders in `output_dir`.

    Args:
        input_dir (Path): Directory containing input .png images.
        output_dir (Path): Directory to save filtered images.
        baseline_dir (Path): Directory with baseline masks.
        threshold (float): Value between 0 and 1 to threshold mask intensity.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for folder in input_dir.iterdir():
        if not folder.is_dir():
            continue  # Skip non-folder entries

        input_folder = folder
        output_folder = output_dir / folder.name

        if output_folder.exists():
            print(f"Skipping '{folder.name}' — already processed.")
            continue

        output_folder.mkdir(parents=True)

        images = []
        masks_ref = []
        image_files = sorted(folder.glob("*.png"))

        for img_path in image_files:
            image = Image.open(img_path).convert("L")
            images.append(image)

            mask_path = baseline_dir / img_path.name
            mask = Image.open(mask_path).convert("L")

            mask_array = np.array(mask)
            binary_mask = np.where(mask_array < threshold * 255, 255, 0).astype(np.uint8)

            dilated_mask = binary_dilation(binary_mask)
            dilated_mask = binary_dilation(dilated_mask)

            labeled_mask, num_features = label(dilated_mask)

            if num_features > 0:
                sizes = [np.sum(labeled_mask == i) for i in range(1, num_features + 1)]
                largest_label = np.argmax(sizes) + 1
                largest_mask = np.where(labeled_mask == largest_label, 255, 0).astype(np.uint8)
                convolved = gaussian_filter(largest_mask, sigma=1)
            else:
                convolved = np.zeros_like(mask_array)

            masks_ref.append(convolved)

        for image, mask, img_path in zip(images, masks_ref, image_files):
            img_array = np.array(image)
            filtered_image = 255 - mask + img_array * (mask / 255)
            filtered_image = Image.fromarray(filtered_image.astype(np.uint8))
            filtered_image.save(output_folder / img_path.name)

    print("Prototype filtering complete.")

def flag_prototypes(
    input_parent: Path,
    output_parent: Path,
    baseline_folder: Path,
    default_threshold: float = 0.8 * 255,
    sup_threshold: float = 0.65 * 255,
    show_plots: bool = False
):
    output_parent.mkdir(parents=True, exist_ok=True)

    for folder in input_parent.iterdir():
        if not folder.is_dir() or folder.name == ".DS_Store":
            continue

        input_folder = folder
        output_folder = output_parent / folder.name
        output_folder.mkdir(parents=True, exist_ok=True)

        deviations_per_image = []

        for file in sorted(os.listdir(input_folder)):
            if not file.endswith('.png'):
                continue

            file_path = input_folder / file
            image = Image.open(file_path).convert("L")
            img_array = np.array(image)

            # Load and process reference mask
            mask_path = baseline_folder / file
            mask = Image.open(mask_path).convert("L")
            mask_array = np.array(mask)

            binary_mask_ref = np.where(mask_array < default_threshold, 255, 0).astype(np.uint8)
            dilated_mask_ref = binary_mask_ref.copy()
            for _ in range(2):
                dilated_mask_ref = binary_dilation(dilated_mask_ref)
            labeled_mask_ref, num_features_ref = label(dilated_mask_ref)
            component_sizes_ref = [np.sum(labeled_mask_ref == i) for i in range(1, num_features_ref + 1)]
            if component_sizes_ref:
                largest_label_ref = np.argmax(component_sizes_ref) + 1
                largest_component_mask_ref = np.where(labeled_mask_ref == largest_label_ref, 255, 0).astype(np.uint8)
            else:
                largest_component_mask_ref = np.zeros_like(mask_array)
            convolved_mask_ref = gaussian_filter(largest_component_mask_ref, sigma=1)

            # Process document (sup) mask
            binary_mask_sup = np.where(img_array < sup_threshold, 255, 0).astype(np.uint8)
            labeled_mask_sup, num_features_sup = label(binary_mask_sup)
            component_sizes_sup = [np.sum(labeled_mask_sup == i) for i in range(1, num_features_sup + 1)]
            if component_sizes_sup:
                largest_label_sup = np.argmax(component_sizes_sup) + 1
                largest_component_mask_sup = np.where(labeled_mask_sup == largest_label_sup, 255, 0).astype(np.uint8)
            else:
                largest_component_mask_sup = np.zeros_like(mask_array)

            # Filtered image
            filtered_array = 255 - convolved_mask_ref + img_array * (convolved_mask_ref / 255.0)
            filtered_array = np.clip(filtered_array, 0, 255).astype(np.uint8)
            filtered_image_ref = Image.fromarray(filtered_array).convert("RGB")

            # Difference score (correct float computation)
            difference = (np.sum(largest_component_mask_sup * (255.0 - convolved_mask_ref)) / 255.0) / 255.0
            deviations_per_image.append(difference)

            # Color-coding
            if 15.5 <= difference <= 30:
                border_color = "orange"
            elif difference > 30:
                border_color = "red"
            else:
                border_color = "white"

            filtered_image_ref_with_border = ImageOps.expand(filtered_image_ref, border=2, fill=border_color)
            filtered_image_ref_with_border.save(output_folder / file)

            if show_plots:
                difference_array = (largest_component_mask_sup * (255.0 - convolved_mask_ref)) / 255.0
                difference_image = Image.fromarray(difference_array.astype(np.uint8))

                fig, axes = plt.subplots(1, 5, figsize=(12, 4))
                axes[0].imshow(img_array, cmap='gray', clim=(0, 255))
                axes[0].set_title(f'Doc prototype ({folder.name})')
                axes[1].imshow(convolved_mask_ref, cmap='gray', clim=(0, 255))
                axes[1].set_title('Ref Mask')
                axes[2].imshow(largest_component_mask_sup, cmap='gray', clim=(0, 255))
                axes[2].set_title('Doc Mask')
                axes[3].imshow(filtered_array, cmap='gray', clim=(0, 255))
                axes[3].set_title('Filtered Doc')
                axes[4].imshow(difference_image, cmap='gray', clim=(0, 255))
                axes[4].set_title(f'Diff Map: {difference:.2f}')

                for ax in axes:
                    ax.axis("off")
                plt.tight_layout()
                plt.show()

    print("Filtering and flagging complete.")



def plot_document_graphs(folder_1_paths, folder_2_paths, output_folder, char_occurrences_df, character_mapping, character_range, codename_to_id):
    num_rows = (len(folder_2_paths) + 2) // 3
    num_cols = 3

    fig = plt.figure(figsize=(22, 6 * num_rows))
    spec = gridspec.GridSpec(num_rows, num_cols + 1, width_ratios=[1, 1, 1, 0.05], wspace=0.3, hspace=0.4)

    axs = [fig.add_subplot(spec[i // 3, i % 3]) for i in range(len(folder_2_paths))]
    norm = LogNorm()

    for i, folder_2_path in enumerate(folder_2_paths):
        folder_comp_id = os.path.basename(os.path.normpath(folder_2_path))
        codename = [k for k, v in codename_to_id.items() if v == folder_comp_id][0]

        all_l2_diff_southern = []
        all_l2_diff_northern = []
        char_occurrences = char_occurrences_df[folder_comp_id]

        for character in character_range:
            doc_path = os.path.join(folder_2_path, f'{character}.png')

            northern_path = os.path.join(folder_1_paths[1], f'{character}.png')
            l2_northern = calculate_l2_norm(Image.open(northern_path), Image.open(doc_path))
            all_l2_diff_northern.append(l2_northern)

            southern_path = os.path.join(folder_1_paths[0], f'{character}.png')
            l2_southern = calculate_l2_norm(Image.open(southern_path), Image.open(doc_path))
            all_l2_diff_southern.append(l2_southern)

        ax = axs[i]
        scatter = ax.scatter(all_l2_diff_southern, all_l2_diff_northern, c=char_occurrences, cmap='viridis', norm=norm, alpha=0.6, s=60)
        ax.set_ylabel('Distance to ST prototype', fontsize=20)
        ax.set_xlabel('Distance to NT prototype', fontsize=20)
        ax.grid(True)
        ax.margins(0.3)

        for char, x, y in zip(character_range, all_l2_diff_southern, all_l2_diff_northern):
            ax.text(x, y, character_mapping.get(str(char), str(char)), fontsize=20, ha='right', va='bottom')

        ax.text(0.5, 0.98, f'Document: {codename}', color='black', fontsize=20, ha='center', va='top', transform=ax.transAxes)

    # Uniform formatting for all axes
    for ax in axs:
        ax.plot([0, 17], [0, 17], linestyle='--', color='gray')
        ax.set_xlim(0, 17)
        ax.set_ylim(0, 17)
        ax.set_xticks([0, 5, 10, 15])
        ax.set_yticks([0, 5, 10, 15])
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True)

    # Add the colorbar on the last column of the GridSpec
    cax = fig.add_subplot(spec[:, -1])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=norm), cax=cax)
    cbar.set_label('Occurrences')

    plt.savefig(os.path.join(output_folder, 'paper_document_graphs.jpeg'))
    plt.close()

def plot_letter_graphs(
    folder_1_paths,
    folder_2_paths,
    output_folder,
    manuscript_info_df,
    character_mapping,
    character_range,
    split_to_id,
    codename_to_id
):
    num_letters = len(character_range)
    num_rows = (num_letters + 2) // 3

    fig = plt.figure(figsize=(18, 5.5 * num_rows))
    spec = gridspec.GridSpec(num_rows, 3, wspace=0.3, hspace=0.4)
    axs = [fig.add_subplot(spec[i]) for i in range(num_letters)]

    id_to_codename = {v: k for k, v in codename_to_id.items()}

    for i, character in enumerate(character_range):
        ax = axs[i]
        char_label = character_mapping.get(str(character), str(character))
        ax.text(0.5, 0.98, f'Character: {char_label}', fontsize=22, ha='center', va='top', transform=ax.transAxes)

        southern_proto = Image.open(folder_1_paths[0] / f'{character}.png')
        northern_proto = Image.open(folder_1_paths[1] / f'{character}.png')

        all_l2_diff_south, all_l2_diff_north = [], []
        colors, markers, codenames = [], [], []

        for folder_2_path in folder_2_paths:
            comp_id = folder_2_path.name
            img_path = folder_2_path / f'{character}.png'
            if not img_path.exists():
                continue

            img = Image.open(img_path)
            l2_south = calculate_l2_norm(southern_proto, img)
            l2_north = calculate_l2_norm(northern_proto, img)

            all_l2_diff_south.append(l2_south)
            all_l2_diff_north.append(l2_north)

            codename = id_to_codename.get(comp_id, comp_id)
            codenames.append(codename)

            script = manuscript_info_df.loc[manuscript_info_df['ID'] == comp_id, 'script']
            script = script.iloc[0] if not script.empty else 'Unknown'
            split = 'train' if comp_id in split_to_id['train'] else 'test'

            colors.append('red' if script == 'Southern_Textualis' else 'blue')
            markers.append('o' if split == 'train' else 'x')

        for x, y, color, marker, codename in zip(all_l2_diff_south, all_l2_diff_north, colors, markers, codenames):
            ax.scatter(x, y, c=color, marker=marker, s=70)
            ax.annotate(codename, (x, y), textcoords="offset points", xytext=(2, 2), ha='left', va='bottom', fontsize=12)

        # Axis formatting
        ax.plot([0, 10], [0, 10], linestyle='--', color='gray')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_xticks([0, 5, 10])
        ax.set_yticks([0, 5, 10])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
        ax.grid(True)

    plt.tight_layout()
    output_folder.mkdir(parents=True, exist_ok=True)
    output_plot_path = output_folder / 'paper_letter_graphs.jpg'
    plt.savefig(output_plot_path, bbox_inches='tight')
    plt.close()