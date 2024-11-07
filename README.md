*****************************************************************************
Overview:

This project applies unsupervised learning techniques to classify images into two categories: hate (1) and non-hate (0). The project is an image classification pipeline using ORB feature extraction, PCA for dimensionality reduction, and a trained autoencoder for feature extraction with KMeans clustering to identify potential hate indicators in images. It assigns each image to one of two clusters (e.g., hate or nonhate) based on visual features, saving processed images with color-coded borders and generating prediction summaries. The models run in parallel for better performance, images are processed in batches with time intervals, and memory mapping technique is used to optimize performance. The final prediction of an image as hate or nonhate is based on an average value tracked from the three models.

Requirements: Python3.x, packages os, time, numpy, pandas, scikit-learn, joblib, opencv-python, matplotlib, tensorflow, concurrent.futures

Main Files:
trainference.py: Main script that orchestrates training and prediction steps.
'train' and 'test' folders. Place training images in the 'train' folder and test images in the 'test' folder in the same path as the script.

Run the script: python trainference.py

Features: The program loads and preprocesses images for training and testing. It uses PCA or Principal Component Analysis technique for dimensionality reduction, ORB or Oriented FAST and Rotated BRIEF technique to extract up to 500 features from the image, and an autoencoder model to extract features from images. Finally, K-Means clustering mechanism is used to classify images based on extracted features for each of the above three models. We save the model files in .pkl format and output predictions in CSV format for each model, then create a final predictions csv file based on the average. We also generate marked images with border color coding and a summary percentages for hate and non-hate classifications. 

*****************************************************************************

Code Overview:

Imports: import os, time, numpy, pandas, cv2, joblib, KMeans, PCA, Normalize, layers, models

Environment Variables:
OMP_NUM_THREADS: Sets the number of threads for OpenMP (default: 1).
TF_ENABLE_ONEDNN_OPTS: Disables OneDNN optimizations in TensorFlow (default: 0).

#Initialize parameters
IMG_WIDTH, IMG_HEIGHT = 128, 128  # Resize images to this size
BATCH_SIZE = 500 # Define batch size for processing images

Functions:
update_run_count(): This function updates the run_count file to keep track of the current run.
plot_model_structure(parameters): Architecture of 3 models are plotted and saved to an image file.
training__model(parameters): Trains the model using PCA, ORB, and Autoencoder.
predictions_model(parameters): Loads the trained models and performs predictions.
save_output_images_models(): Saves images with colored borders based on clustering results.
save_output_images_final(parameters): Saves final images for the combined model.
save_predictions_to_csv(parameters): Saves the predictions to a CSV file
generate_final_predictions(parameters): Function to generate final predictions
def majority_voting(paramaeter): Function to determine final labels based on majority voting

Log files: Logs are saved in for tracking image loading and processing details.

*****************************************************************************

Project Structure:
.

├── counter                                 # Tracks the run count

├── images/                                 # Folder to store final images

├── memmap/                                 # Save memmap files for better performance.

├── models/         

├────────├ENCODER_model/                    # Saves the outputs for the ENCODER MODEL

├────────────────├architecture/

├────────────────├logs/  

├────────────────├models/

├────────────────├models_backup/

├────────────────├output_images/ 

├────────────────├predictions/

├────────├ORB_model/                         # Saves the outputs for the ORB MODEL

├────────────────├architecture/

├────────────────├logs/  

├────────────────├models/

├────────────────├models_backup/

├────────────────├output_images/ 

├────────────────├predictions/ 

├─────────├PCA_model/                        # Saves the outputs for the PCA MODEL

├────────────────├architecture/      

├────────────────├logs/  

├────────────────├models/

├────────────────├models_backup/

├────────────────├output_images/ 

├────────────────├predictions/

├── predictions/                              # Folder to save final predictions

├── test/                                     # Folder to store test images

├── train/                                    # Folder to store training images

├── trainference.py                           # Main script for training and predictions    

├── README.md                                 # Project documentation
    
*******************************************************************************

How it works:

1. Import the necessary modules to run the script.
2. Set environment variables for optimal performance.
3. Define image dimensions and sets up tracking for model runs.
4. Update and record the current run count for the script in a txt file.
5. Read images from specified folders, resize them, and log key details.
5. Normalizes pixel values to range between 0 and 1 for input into the model.
6. Construct PCA model, ORB model and ENCODER model and plot the model architecture.
7. Normalizes the encoded features and apply KMeans clustering with 2 clusters.
8. Saves the trained clustering model as a .pkl file.
9. Add a border to each image to denote hate/non-hate classification.
10. Save output images in separate folders based on the current run count.
11. Save predictions and run-specific summary data (hate/non-hate percentages) to CSV files.

*****************************************************************************
