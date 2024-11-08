#Import the necessary libraries

import os
import gc
import time
import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from tensorflow.keras import layers, models

##############################################################

# Set the OMP and ONEDNN environment variable
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Parameters
IMG_WIDTH, IMG_HEIGHT = 128, 128  # Resize images to this size

# Assign folders with the images
TRAIN_FOLDER = 'train'
TEST_FOLDER = 'test'

#Counter Directory
COUNTER_DIR='counter'
RUN_COUNT_FILE = f'{COUNTER_DIR}/run_count.txt'  # File to track the run count

# Directory paths for each model
MODELS_DIR='models'
PCA_MODEL_DIR = f"{MODELS_DIR}/PCA_model"
ORB_MODEL_DIR = f"{MODELS_DIR}/ORB_model"
ENCODER_MODEL_DIR = f"{MODELS_DIR}/ENCODER_model"

#Save image files describing the architecture of the models
PCA_MODEL_ARCHITECTURE= f"{PCA_MODEL_DIR}/architecture"
ORB_MODEL_ARCHITECTURE= f"{ORB_MODEL_DIR}/architecture"
ENCODER_MODEL_ARCHITECTURE= f"{ENCODER_MODEL_DIR}/architecture"

#Save the model files in .pkl format
PCA_MODELS = f"{PCA_MODEL_DIR}/models"
ORB_MODELS = f"{ORB_MODEL_DIR}/models"
ENCODER_MODELS = f"{ENCODER_MODEL_DIR}/models"

#Save the images predicted by the models 
PCA_OUTPUT_IMAGES = f"{PCA_MODEL_DIR}/output_images"
ORB_OUTPUT_IMAGES = f"{ORB_MODEL_DIR}/output_images"
ENCODER_OUTPUT_IMAGES = f"{ENCODER_MODEL_DIR}/output_images"

#Save the model prediction for each of the model in .csv format
PCA_MODEL_PREDICTION = f"{PCA_MODEL_DIR}/predictions"
ORB_MODEL_PREDICTION = f"{ORB_MODEL_DIR}/predictions"
ENCODER_MODEL_PREDICTION = f"{ENCODER_MODEL_DIR}/predictions"

#Save the backup of the model files in .pkl format from each of the iteration for re-use
PCA_MODELS_BACKUPS= f"{PCA_MODEL_DIR}/models_backup"
ORB_MODELS_BACKUPS= f"{ORB_MODEL_DIR}/models_backup"
ENCODER_MODELS_BACKUPS= f"{ENCODER_MODEL_DIR}/models_backup"

#Log the information processed while processing the images
PCA_MODEL_LOGS = f"{PCA_MODEL_DIR}/logs"
ORB_MODEL_LOGS = f"{ORB_MODEL_DIR}/logs"
ENCODER_MODEL_LOGS = f"{ENCODER_MODEL_DIR}/logs"

#Outputs to the program
FINAL_OUTPUT_IMAGES = 'images'
PERTURBED_IMAGES= f"{FINAL_OUTPUT_IMAGES}/perturbed_images"
PREDICTIONS_DIR='predictions'
PERTURBED_PREDICTIONS= f"{PREDICTIONS_DIR}/perturbed_csv"

# Define batch size for processing images and memmap to optimize memory usage
BATCH_SIZE = 50 
MEMORY_MAP= 'memmap'

##############################################################

#Run count keeps track of the number of times the script has been trained
def update_run_count():
    try:
        # Read and update the run count from a file
        if os.path.exists(RUN_COUNT_FILE):
            with open(RUN_COUNT_FILE, 'r') as f:
                run_count = int(f.read().strip()) + 1
        else:
            run_count = 1

        # Save the updated run count back to the file
        with open(RUN_COUNT_FILE, 'w') as f:
            f.write(str(run_count))

    except Exception as e:
        print(f"Warning: Could not update run count due to: {e}")
    
    return run_count

##############################################################

#PLOTTING THE THREE ARCHITECTURES for PCA, ORB, and ENCODER models

#PCA MODEL ARCHITECTURE
def plot_pca_model_structure(run_count):
    layers_info = [
        ("Input Layer (Images)", (None, 28, 28, 3)),
        ("Flatten Layer", (None, 784)),
        ("PCA Layer (64 Components)", (None, 64)),
        ("KMeans Clustering Layer (2 Clusters)", (None, 2))
    ]
    plot_model_structure(layers_info, run_count, 'PCA', PCA_MODEL_ARCHITECTURE)

#ORB MODEL ARCHITECTURE
def plot_orb_model_structure(run_count):
    layers_info = [
        ("Input Layer (Images)", (None, 28, 28, 3)),
        ("ORB Feature Extraction Layer", (None, 500 * 32)),
        ("KMeans Clustering Layer (2 Clusters)", (None, 2))
    ]
    plot_model_structure(layers_info, run_count, 'ORB', ORB_MODEL_ARCHITECTURE)

#ENCODER MODEL ARCHITECTURE
def plot_encoder_model_structure(layers_info, run_count):
    plot_model_structure(layers_info, run_count, 'ENCODER', ENCODER_MODEL_ARCHITECTURE)

#A common plotting function for the three models.

def plot_model_structure(layers_info, run_count, model_name, output_directory):
    # Check if run_count is 1; if not, exit the function
    if run_count != 1:
        return

    filename = os.path.join(output_directory, f'{model_name}_architecture.jpg')
    plt.figure(figsize=(10, 8))  # Adjust figure size

    # Define colors for different layers
    colors = {
        "Input Layer (Images)": "blue",
        "Flatten Layer": "orange",
        "PCA Layer (64 Components)": "purple",
        "KMeans Clustering Layer (2 Clusters)": "green",
        "ORB Feature Extraction Layer": "orange",
        "Dense Layer 1 (128 Neurons)": "green",
        "Encoding Layer (64 Neurons)": "purple",
        "Dense Layer 2 (128 Neurons)": "red",
        "Dense Layer 3 (Output)": "cyan",
        "Output Reshape Layer": "magenta"
    }

    # Set vertical positions for the layers
    y_positions = np.linspace(len(layers_info) * 1.5, 0, len(layers_info))  # Adjusted spacing

    for y, (layer_name, output_shape) in zip(y_positions, layers_info):
        plt.text(0, y, layer_name, fontsize=12, ha='right', color=colors.get(layer_name, 'black'))
        plt.plot([0, 1], [y, y], color=colors.get(layer_name, 'black'), lw=4)  # Straight line
        plt.text(1.05, y, f'Output Shape: {output_shape}', fontsize=12, ha='left', color='black')

    # Adding arrows to indicate flow
    for i in range(len(y_positions) - 1):
        plt.annotate('', xy=(1, y_positions[i + 1]), xytext=(1, y_positions[i]),
                     arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Set limits ensuring all layers are visible
    plt.xlim(-0.5, 1.5)
    plt.ylim(-1, len(layers_info) * 1.5)  # Increased the y-limit for more space
    plt.axis('off')
    plt.title(f'{model_name} Model Architecture', fontsize=16)
    plt.grid(False)
    plt.tight_layout()  # Adjust layout to fit elements better

    # Save the plot as an image file
    plt.savefig(filename, bbox_inches='tight', dpi=300)  # Save with tight bounding box and high resolution
    plt.close()  # Close the plot to free up memory
    print(" ")
    print(f"{model_name} model architecture saved to {filename}")
    print(" ")

##############################################################

# Load and preprocess images
def load_images(image_folder, log_file):
    images, image_names, log_entries = [], [], []
    if not os.path.exists(image_folder):
        log_entries.append(f"Image folder does not exist: {image_folder}\n")
        with open(log_file, 'a') as f:
            f.write("\n".join(log_entries))
        return np.array([]), image_names  # Return empty arrays for consistency

    for filename in os.listdir(image_folder):
        img_path = os.path.join(image_folder, filename)
        try:
            img = cv2.imread(img_path)
            if img is not None:
                log_entries.append(f"Loaded image: {filename} | Shape: {img.shape} | Dtype: {img.dtype} | "
                                   f"Sample Pixels: {img[:3, :3].tolist()}\n")
                
                # Resize to expected shape only after ensuring the image is loaded
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))  # Resize to expected shape
                log_entries.append(f"Resized image: {filename} | Shape: {img.shape} | Sample Pixels: {img[:3, :3].tolist()}\n")
                
                images.append(img)
                image_names.append(filename)
            else:
                log_entries.append(f"Failed to load image: {filename}\n")
                
        except Exception as e:
            log_entries.append(f"Error loading image {filename}: {e}\n")

    # Normalize images
    normalized_images = np.array(images, dtype=np.float32) / 255.0

    # Log normalized images details
    log_entries.append("\nImages list after converting to float32 and normalization (divided by 255):\n")
    for img in normalized_images:
        # Consider logging only part of each array to avoid excessive data
        log_entries.append(f"Shape: {img.shape}, Sample Values: {img.flatten()[:10].tolist()}\n")
    
    log_entries.append("\nDetails of the complete image list with names and flattened pixel values:\n")
    for name, img in zip(image_names, images):
        pixel_summary = img.flatten()[:10]  # First 10 pixel values
        log_entries.append(f"Image Name: {name} | Shape: {img.shape} | Pixel Values Sample: {pixel_summary.tolist()}\n")
 
    # Write all log entries at once
    with open(log_file, 'w') as f:
        f.write("\n".join(log_entries))

    return normalized_images, image_names

##############################################################

# Function to save output trained images with red/green marking, it is set with counter to save only 30 images for each model
def save_output_images_models(images, cluster_labels, image_names, output_folder, run_count):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize a counter to limit the number of saved images
    count = 0
    for img, label, name in zip(images, cluster_labels, image_names):
        if count >= 30:
            break

        # Keep the original image for visibility and scale back to [0, 255]
        combined_img = (img * 255).astype(np.uint8)  # Convert back to 8-bit format

        # Determine the border color based on the label
        border_color = (0, 0, 255) if label == 1 else (0, 255, 0)
        border_thickness = 10

        # Draw a rectangle around the entire image
        cv2.rectangle(combined_img, (0, 0), (combined_img.shape[1], combined_img.shape[0]), border_color, border_thickness)

        output_path = os.path.join(output_folder, f"{os.path.splitext(name)[0]}_run_{run_count}_output.jpg")
        success = cv2.imwrite(output_path, combined_img)
        
        if success:
            count += 1  # Increment counter after saving the image
        else:
            print(f"Warning: Failed to save image {name} at {output_path}")

# Function to save final output images with colored borders based on predictions
def save_output_images_final(final_df, run_count):
    # Create a directory for the output images for the current run
    output_run_folder = os.path.join(FINAL_OUTPUT_IMAGES, f'test_run_{run_count}')
    os.makedirs(output_run_folder, exist_ok=True)

    count = 0
    for index, row in final_df.iterrows():
        if count >= 100:
            break

        image_name = row['Image Name']
        prediction = row['final_prediction']
        
        # Load the image (assuming images are stored in a specific directory)
        image_path = os.path.join(TEST_FOLDER, image_name)
        # Ensure the output image has a .jpg extension
        output_image_name = f"{os.path.splitext(image_name)[0]}.jpg"
        output_path = os.path.join(output_run_folder, output_image_name)
        
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: {image_path} does not exist or cannot be read.")
            continue
        
        # Determine border color based on prediction
        border_color = (0, 255, 0) if prediction == 0 else (0, 0, 255)  # Green for 0, Red for 1

        # Add a border to the image
        bordered_image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=border_color)
              
        # Save the bordered image with .jpg extension
        success = cv2.imwrite(output_path, bordered_image)
        if success:
            count += 1  # Increment counter after saving the image
        else:
            print(f"Warning: Failed to save image {image_name} at {output_path}")


    # Step 1: Identify "hate" images (where prediction == 1)
    perturbed_images_df = final_df[final_df['final_prediction'] == 1][['Image Name']].copy()
    perturbed_images_df['Cluster Label'] = 'hate'  # Set label to 'hate'

    # Step 2: Save to `perturbed_images.csv`
    perturbed_csv_path = os.path.join(PERTURBED_PREDICTIONS, 'perturbed_images.csv')
    perturbed_images_df.to_csv(perturbed_csv_path, index=False)


    # Separate loop to save all "perturbed" images (without count limit)
    for index, row in perturbed_images_df.iterrows():
        image_name = row['Image Name']
        # Load the image from the test folder
        image_path = os.path.join(TEST_FOLDER, image_name)
        output_image_name = f"{os.path.splitext(image_name)[0]}.jpg"
        perturbed_image_path = os.path.join(PERTURBED_IMAGES, output_image_name)

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: {image_path} does not exist or cannot be read.")
            continue

        # Add red border to indicate "hate"
        bordered_image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 255))
        success = cv2.imwrite(perturbed_image_path, bordered_image)
        if not success:
           print(f"Warning: Failed to save perturbed image {image_name} at {perturbed_image_path}")


##############################################################

def save_predictions_to_csv(image_names, cluster_labels, run_count, file_path):
    # Create a DataFrame for the predictions
    predictions_df = pd.DataFrame({
        'Run Count': [run_count] * len(image_names),
        'Image Name': image_names,
        'Prediction': cluster_labels
    })
    
    # Append to the existing CSV file (no headers if it already exists)
    predictions_df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)

##############################################################

# Function to load predictions from each model's CSV file
def load_predictions(file_path):
    """Load predictions from a CSV file with error handling."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise IOError(f"An error occurred while loading the file {file_path}: {e}")

# Function to generate final predictions with duplicate checks and unique formatting
def generate_final_predictions(run_count):
    print(" ")
    print("Generating Final Predictions...")
    print(" ")

    # Directory paths for prediction files
    pca_predictions_path = f'{PCA_MODEL_PREDICTION}/test_predictions_pca_{run_count}.csv'
    orb_predictions_path = f'{ORB_MODEL_PREDICTION}/test_predictions_orb_{run_count}.csv'
    encoder_predictions_path = f'{ENCODER_MODEL_PREDICTION}/test_predictions_encoder_{run_count}.csv'

    try:
        # Load the predictions from each model
        encoder_df = load_predictions(encoder_predictions_path)
        orb_df = load_predictions(orb_predictions_path)
        pca_df = load_predictions(pca_predictions_path)
        
        # Calculate the final labels using majority voting
        final_df = majority_voting(encoder_df, orb_df, pca_df)

        # Calculate % hate and % nonhate
        total_images = len(final_df)
        hate_count = np.sum(final_df['final_prediction'] == 1)
        nonhate_count = total_images - hate_count
        hate_percentage = (hate_count / total_images) * 100
        nonhate_percentage = (nonhate_count / total_images) * 100

        # Add summary row with percentages to the final DataFrame for this run
        summary_df = pd.DataFrame({
            'Image Name': ['Summary'],
            'Run Count': [""],
            'Prediction_encoder': [f'% Hate: {hate_percentage:.2f}, % NonHate: {nonhate_percentage:.2f}'],
            'Prediction_orb': [''],
            'Prediction_pca': [''],
            'final_prediction': ['']
        })
        
       # Append the summary row to the final DataFrame
        final_combined_df = pd.concat([final_df, summary_df], ignore_index=True)

        # Save the final predictions to a new CSV file for this run
        final_predictions_path = os.path.join(PREDICTIONS_DIR, f'final_test_predictions_{run_count}.csv')
        final_combined_df.to_csv(final_predictions_path, index=False)
        print(" ")
        print(f"Final predictions saved to {final_predictions_path}")
        print(" ")

        # Call the function to save final output images with markings
        save_output_images_final(final_combined_df, run_count)

        # Set pandas options to display all rows and columns without truncation
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        # Print the entire contents of the final predictions CSV file
        print("final test predictions:")
        print(" ")
        print(final_combined_df)
        
        # Reset pandas options back to default after printing
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.max_colwidth')
    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except IOError as io_error:
        print(io_error)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Function to determine final labels based on majority voting
def majority_voting(encoder_df, orb_df, pca_df):
    # Merging the dataframes on the 'Image Name' column
    merged_df = encoder_df.merge(orb_df, on="Image Name", suffixes=('_encoder', '_orb'))
    merged_df = merged_df.merge(pca_df, on="Image Name", suffixes=('', '_pca'))

    # Check if the expected columns exist
    if 'Prediction' not in merged_df.columns:
        raise KeyError("The 'Prediction' column was not found in the merged DataFrame.")
    
    # Rename the PCA prediction column for clarity
    merged_df.rename(columns={'Prediction': 'Prediction_pca'}, inplace=True)

    # Ensure 'Run Count' is named consistently across all dataframes
    merged_df['Run Count'] = merged_df['Run Count_encoder']  # Use the encoder's run count as the main reference

    # Apply majority voting to each image
    def assign_final_label(row):
        # Get the predictions from each model
        labels = [row['Prediction_encoder'], row['Prediction_orb'], row['Prediction_pca']]
        # Assign final label based on the majority vote
        return 1 if sum(labels) > 1 else 0

    # Calculate the final prediction using majority voting
    merged_df['final_prediction'] = merged_df.apply(assign_final_label, axis=1)

    # Now, keep only the relevant columns for the final output
    final_df = merged_df[['Image Name', 
                           'Run Count',  # Use the renamed 'Run Count'
                           'Prediction_encoder',
                           'Prediction_orb',
                           'Prediction_pca',  # This will be the PCA prediction
                           'final_prediction']]  # This will be the majority vote result

    return final_df

##############################################################

#PCA MODEL

#Extract PCA features
def extract_pca_features(train_images,run_count):
    features = []
    n_samples = len(train_images)
    n_features = train_images.shape[1] * train_images.shape[2] * train_images.shape[3]  # Assuming images are 4D: (batch, height, width, channels)
    n_components = min(64, n_samples, n_features)
    pca = PCA(n_components=n_components, svd_solver='full')

    # Memory-mapped file to store flattened features
    memmap_path = f'{MEMORY_MAP}/pca_features_{run_count}.dat'
    memmap_array = np.memmap(memmap_path, dtype=np.float32, mode='w+', shape=(n_samples, n_features))


    # Process in batches of BATCH_SIZE
    for start_idx in range(0, n_samples, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, n_samples)
        batch_images = train_images[start_idx:end_idx]
        
        # Flatten and transform batch images
        batch_images_flattened = batch_images.reshape(len(batch_images), -1)
        memmap_array[start_idx:end_idx] = batch_images_flattened

        print(f" Processed pca batch {start_idx + 1} to {end_idx} of {n_samples} images")
        
        # Clear memory after each batch
        del batch_images, batch_images_flattened
        gc.collect()

        time.sleep(2)  # Add a 2-second delay after processing each batch

    # Fit PCA on memory-mapped array
    pca.fit(memmap_array)

    # Apply PCA transformation and normalize features
    features = pca.transform(memmap_array)
    features = normalize(features)
    
    # Save the PCA model
    joblib.dump(pca, f'{PCA_MODELS}/pca_model.pkl') # Save the fitted PCA model
    joblib.dump(pca, f'{PCA_MODELS_BACKUPS}/pca_model_{run_count}.pkl')

    return features

#Trains the model using PCA dimension reduction and cluster the training images using KMeans
def training_pca_model(run_count):
    print(" ")
    print("Running training for PCA Model...")
    print(" ")
    
    train_images, train_image_names = load_images(TRAIN_FOLDER, f'{PCA_MODEL_LOGS}/training_logs_pca_{run_count}.txt')
    if len(train_images) == 0:
        print("No images loaded. Check the 'train' folder for images.")
        return
    
    features = extract_pca_features(train_images,run_count)
    features = normalize(features)
 
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(features)
    joblib.dump(kmeans, f'{PCA_MODELS}/kmeans_pca_model.pkl')
    joblib.dump(kmeans, f'{PCA_MODELS_BACKUPS}/kmeans_pca_model_run_count_{run_count}.pkl')

    cluster_labels = kmeans.predict(features)
    print(" ") 
    print(" PCA Model trained and saved. Images saved with red/green border for hate/non-hate.")
    print(" ")

#Prediction function to make the predictions
def predictions_pca_model(run_count):
    print(" ")
    print("Running Predictions for PCA Model...")
    print(" ")

    test_images, test_image_names = load_images(TEST_FOLDER, f'{PCA_MODEL_LOGS}/testing_logs_pca_{run_count}.txt')
    if len(test_images) == 0:
        print("No images loaded. Check the 'test' folder for images.")
        return

    # Load the PCA model fitted during training
    try:
        pca = joblib.load(f'{PCA_MODELS}/pca_model.pkl')

        features = pca.transform(test_images.reshape(len(test_images), -1))
        features = normalize(features)

        kmeans = joblib.load(f'{PCA_MODELS}/kmeans_pca_model.pkl')
        cluster_labels = kmeans.predict(features)
        output_test_folder = f'{PCA_OUTPUT_IMAGES}/test_run_{run_count}'

        save_output_images_models(test_images, cluster_labels, test_image_names, output_test_folder, run_count)
        save_predictions_to_csv(test_image_names, cluster_labels, run_count, f'{PCA_MODEL_PREDICTION}/test_predictions_pca_{run_count}.csv')
    except FileNotFoundError:
        print("PCA model file not found. Please ensure the model is trained before making predictions.")
        return

##############################################################

#ORB MODEL

#Extract feature from image
def extract_orb_features(image, max_features=500):
        #Extracts ORB features from a single image and returns a feature vector.
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(image, None)
        
        if descriptors is not None:
            # Only keep the first max_features if more exist
            if len(descriptors) > max_features:
                descriptors = descriptors[:max_features]

            # Flatten and pad the descriptors to ensure uniform length
            feature_vector = np.zeros(max_features * 32)  # ORB descriptors are 32 bytes
            feature_vector[:descriptors.shape[0] * 32] = descriptors.flatten()  # Fill the feature vector
        else:
            feature_vector = np.zeros(max_features * 32)  # Placeholder for no descriptors
        return feature_vector
 
#Trains the model using ORB and cluster the training images using KMeans
def training_orb_model(run_count):
    print(" ")
    print("Running training for ORB Model...")
    print(" ")

    train_images, train_image_names = load_images(TRAIN_FOLDER, f'{ORB_MODEL_LOGS}/training_logs_orb_{run_count}.txt')
    if len(train_images) == 0:
        print("No images loaded. Check the 'train' folder for images.")
        return
    
    max_features = 500  # Set a maximum number of features to keep from ORB

    features = []

    # Create a memory-mapped file to store ORB features
    memmap_path = f'{MEMORY_MAP}/orb_features_{run_count}.dat'
    memmap_array = np.memmap(memmap_path, dtype=np.float32, mode='w+', shape=(len(train_images), max_features * 32))


    # Process images in batches
    for start_idx in range(0, len(train_images), BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, len(train_images))
        batch_images = train_images[start_idx:end_idx]

        # Extract features for each image in the batch
        batch_features = [extract_orb_features(image, max_features) for image in batch_images]
        features.extend(batch_features)

        # Save batch features to memory-mapped file
        memmap_array[start_idx:end_idx] = np.array(batch_features)
        print(f" Processed orb training batch {start_idx + 1} to {end_idx} of {len(train_images)} images")

        # Free memory after each batch
        del batch_images, batch_features
        gc.collect()  # Collect garbage to free up memory
        time.sleep(2)  # Optional: add delay to reduce load

        time.sleep(2)  # Add a 2-second delay after processing each batch

    # Normalize and convert features to an array for KMeans
    features = normalize(np.array(features))
  
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(features)

    joblib.dump(kmeans, f'{ORB_MODELS}/kmeans_orb_model.pkl')
    joblib.dump(kmeans, f'{ORB_MODELS_BACKUPS}/kmeans_orb_model_run_count_{run_count}.pkl')
    cluster_labels = kmeans.predict(features)
    print(" ") 
    print(" ORB Model trained and saved. Images saved with red/green border for hate/non-hate.")
    print(" ")

#Prediction function to make the predictions
def predictions_orb_model(run_count):
    print(" ") 
    print("Running Predictions for ORB Model...")
    print(" ")

    test_images, test_image_names = load_images(TEST_FOLDER, f'{ORB_MODEL_LOGS}/testing_logs_orb_{run_count}.txt')
    print(" ") 
    print(f"Loaded {len(test_images)} testing images.")
    print(" ")
   
    if len(test_images) == 0:
        print("No images loaded. Check the 'test' folder for images.")
        return

    max_features = 500  # Should match the training phase maximum features

    features = []

    # Process test images in batches
    for start_idx in range(0, len(test_images), BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, len(test_images))
        batch_test_images = test_images[start_idx:end_idx]

        # Extract features for each image in the batch
        batch_features = [extract_orb_features(image, max_features) for image in batch_test_images]
        features.extend(batch_features)

        print(f" Processed orb test batch {start_idx + 1} to {end_idx} of {len(test_images)} images")

        time.sleep(2)  # Add a 2-second delay after processing each batch

    # Normalize and convert features to an array for KMeans
    features = normalize(np.array(features))

    # Load the KMeans model
    kmeans = joblib.load(f'{ORB_MODELS}/kmeans_orb_model.pkl')
    cluster_labels = kmeans.predict(features)

    output_test_folder = f'{ORB_OUTPUT_IMAGES}/test_run_{run_count}'
    save_output_images_models(test_images, cluster_labels, test_image_names, output_test_folder, run_count)
    save_predictions_to_csv(test_image_names, cluster_labels, run_count, f'{ORB_MODEL_PREDICTION}/test_predictions_orb_{run_count}.csv')


##############################################################

#ENCODER MODEL

def build_autoencoder():
    input_img = layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    x = layers.Flatten()(input_img)
    x = layers.Dense(128, activation='relu')(x)
    encoded = layers.Dense(64, activation='relu')(x)

    x = layers.Dense(128, activation='relu')(encoded)
    x = layers.Dense(IMG_WIDTH * IMG_HEIGHT * 3, activation='sigmoid')(x)
    decoded = layers.Reshape((IMG_HEIGHT, IMG_WIDTH, 3))(x)

    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

#Trains the model using autoencoder and cluster the training images using KMeans
def training_encoder_model(run_count):

    print(" ")
    print("Running training for ENCODER Model...")
    print(" ")

    features = []
    train_images, train_image_names = load_images(TRAIN_FOLDER, f'{ENCODER_MODEL_LOGS}/training_logs_encoder_{run_count}.txt')
    print(f"Loaded {len(train_images)} training images.")
    print(" ")
    
    # Check if any images were loaded
    if train_images.shape[0] == 0:
        print("No images loaded. Check the 'train' folder for images.")
        return
    else:
        #Build the autoencoder
        autoencoder = build_autoencoder()

        # Train the autoencoder
        autoencoder.fit(train_images, train_images, epochs=50, batch_size=32, shuffle=True)
        autoencoder.save(f'{ENCODER_MODELS}/autoencoder_model.keras')
        autoencoder.save(f'{ENCODER_MODELS_BACKUPS}/autoencoder_model_run_count_{run_count}.keras')     

        # Get encoded features for clustering
        encoder = models.Model(autoencoder.input, autoencoder.layers[2].output)  # Get encoding layer

        # Create a memory-mapped file for encoded features
        memmap_path = f'{MEMORY_MAP}/autoencoder_features_{run_count}.dat'
        memmap_array = np.memmap(memmap_path, dtype=np.float32, mode='w+', shape=(len(train_images), 128))  # Assuming 64 encoding dimensions

        # Process images in batches
        for start_idx in range(0, len(train_images), BATCH_SIZE):
             end_idx = min(start_idx + BATCH_SIZE, len(train_images))
             batch_images = train_images[start_idx:end_idx]

             batch_features = encoder.predict(batch_images)
             features.extend(batch_features)

             # Save batch features to memory-mapped file
             memmap_array[start_idx:end_idx] = batch_features
             print(" ") 
             print(f" Processed encoder training batch {start_idx + 1} to {end_idx} of {len(train_images)} images")
             print(" ")

             # Free memory after each batch
             del batch_images, batch_features
             gc.collect()  # Collect garbage to free up memory
             time.sleep(2)  # Optional: add delay to reduce load

        # Normalize features and cluster with KMeans
        features = normalize(np.array(features))

        # Clustering using K-Means
        kmeans = KMeans(n_clusters=2, random_state=0)
        kmeans.fit(features)

        # Save the clustering model
        joblib.dump(kmeans, f'{ENCODER_MODELS}/kmeans_autoencoder_model.pkl')
        joblib.dump(kmeans, f'{ENCODER_MODELS_BACKUPS}/kmeans_autoencoder_model_run_count_{run_count}.pkl')

        # Get cluster labels for the images
        cluster_labels = kmeans.predict(features)
        print(" ") 
        print(" ENCODER Model trained and saved. Also, images saved with red border for hate and green border for non hate")
        print(" ")

#Prediction function to make the predictions
def predictions_encoder_model(run_count):
    
    print(" ") 
    print("Running Predictions for ENCODER Model...")
    print(" ") 

    features = []

    test_images, test_image_names = load_images(TEST_FOLDER, f'{ENCODER_MODEL_LOGS}/testing_logs_encoder_{run_count}.txt')
    print(" ") 
    print(f"Loaded {len(test_images)} testing images.")
    print(" ")
  
    # Load the autoencoder model
    autoencoder = models.load_model(f'{ENCODER_MODELS}/autoencoder_model.keras')
    
    # Extract encoded features (latent space)
    encoder = models.Model(autoencoder.input, autoencoder.layers[2].output)

      # Process test images in batches
    for start_idx in range(0, len(test_images), BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, len(test_images))
        batch_test_images = test_images[start_idx:end_idx]

        batch_features = encoder.predict(batch_test_images)
        features.extend(batch_features)
        print(" ") 
        print(f" Processed encoder test batch {start_idx + 1} to {end_idx} of {len(test_images)} images")
        print(" ")

    # Normalize features and predict cluster labels
    features = normalize(np.array(features))
    
    # Load KMeans model
    kmeans = joblib.load(f'{ENCODER_MODELS}/kmeans_autoencoder_model.pkl')

    # Predict hate/non-hate labels
    cluster_labels = kmeans.predict(features)  # Predict cluster labels

    output_test_folder = f'{ENCODER_OUTPUT_IMAGES}/test_run_{run_count}'
 
    save_output_images_models(test_images, cluster_labels, test_image_names, output_test_folder, run_count)
    
    # Append test predictions to CSV
    save_predictions_to_csv(test_image_names, cluster_labels, run_count, f'{ENCODER_MODEL_PREDICTION}/test_predictions_encoder_{run_count}.csv')

##############################################################

# functions to run the models parallely
def run_pca_model(run_count):
    plot_pca_model_structure(run_count)

    training_pca_model(run_count)
    time.sleep(3)
    predictions_pca_model(run_count)
    time.sleep(1)

def run_orb_model(run_count):
    plot_orb_model_structure(run_count)
    training_orb_model(run_count)
    time.sleep(3)
    predictions_orb_model(run_count)
    time.sleep(1)

def run_encoder_model(run_count):
    # Define the layers and their shapes
    layers_info = [
        ("Input Layer", (IMG_HEIGHT, IMG_WIDTH, 3)),
        ("Flatten Layer", (IMG_HEIGHT * IMG_WIDTH * 3,)),
        ("Dense Layer 1 (128 Neurons)", (128,)),
        ("Encoding Layer (64 Neurons)", (64,)),
        ("Dense Layer 2 (128 Neurons)", (128,)),
        ("Dense Layer 3 (Output)", (IMG_HEIGHT * IMG_WIDTH * 3,)),
        ("Output Reshape Layer", (IMG_HEIGHT, IMG_WIDTH, 3))
    ]
    plot_encoder_model_structure(layers_info, run_count)

    training_encoder_model(run_count)
    time.sleep(3)
    predictions_encoder_model(run_count)
    time.sleep(1)


##############################################################

if __name__ == "__main__":

    directories = [COUNTER_DIR, MODELS_DIR, PCA_MODEL_DIR, ORB_MODEL_DIR, ENCODER_MODEL_DIR, PCA_MODEL_PREDICTION, ORB_MODEL_PREDICTION, ENCODER_MODEL_PREDICTION, PCA_OUTPUT_IMAGES, ORB_OUTPUT_IMAGES, ENCODER_OUTPUT_IMAGES, FINAL_OUTPUT_IMAGES, PERTURBED_IMAGES, PREDICTIONS_DIR, PERTURBED_PREDICTIONS, PCA_MODELS, ORB_MODELS, ENCODER_MODELS, PCA_MODELS_BACKUPS, ORB_MODELS_BACKUPS, ENCODER_MODELS_BACKUPS, PCA_MODEL_LOGS, ORB_MODEL_LOGS, ENCODER_MODEL_LOGS, PCA_MODEL_ARCHITECTURE, ORB_MODEL_ARCHITECTURE, ENCODER_MODEL_ARCHITECTURE, MEMORY_MAP ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    # Update run count
    run_count = update_run_count()
    
    print("Note: ")
    print("1. The model may take 20 minutes to 120 minutes to process the data depending on size of the training dataset...") 
    print("2. please do not close the window while the program is executing or run high-end tasks...")
    print("3. The cursor may appear to be stuck sometimes... please do not shut down the system.")
    
    # Run the models in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            'PCA': executor.submit(run_pca_model, run_count),
            'ORB': executor.submit(run_orb_model, run_count),
        }

        # Wait for each future to complete and print results
        for model_name, future in futures.items():
            try:
                future.result()  # This will raise any exceptions encountered in the function
                print(" ")
                print(f"{model_name} Model completed successfully.")
                print(" ")
            except Exception as e:
                print(f"Error in {model_name} Model: {e}") 
    # After PCA and ORB are finished, run the ENCODER model
    try:
        print("Running ENCODER Model...")
        run_encoder_model(run_count)  # Run the third model after PCA and ORB
        print(" ")
        print("ENCODER Model completed successfully.")
        print(" ")
    except Exception as e:
        print(f"Error in ENCODER Model: {e}")

    # Run the final prediction generation function
    generate_final_predictions(run_count)
