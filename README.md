# Advanced Brain Tumor Detection and Segmentation

This project implements an advanced brain tumor detection and segmentation system using a Support Vector Machine (SVM) classifier and image processing techniques (Watershed algorithm, OpenCV). It can classify MRI images as containing a tumor or not, segment the tumor region, calculate its percentage of the total brain area, and provide a size analysis.

## Features

*   **Tumor Classification**: Classifies MRI images into 'Tumor Detected' or 'No Tumor' categories using a trained SVM model.
*   **Tumor Segmentation**: Utilizes image processing (blurring, thresholding, morphological operations, Watershed algorithm) to segment the tumor region from the MRI image.
*   **Tumor Area Percentage**: Calculates the percentage of the image area occupied by the detected tumor.
*   **Tumor Size Analysis**: Provides detailed analysis including total tumor count, largest tumor area, and a size category (Very Small, Small, Medium, Large, Very Large).
*   **Visualizations**: Displays original, processed, tumor mask, and segmented images, along with a pie chart of tumor vs. normal tissue and a comprehensive text summary.
*   **Detailed Report**: Generates a detailed text report including severity assessment and recommendations.

## Requirements

The project uses the following Python libraries:

*   `numpy`
*   `opencv-python` (cv2)
*   `matplotlib`
*   `scikit-learn` (sklearn)
*   `scikit-image` (skimage)

You can install these dependencies using pip:

```bash
pip install numpy opencv-python matplotlib scikit-learn scikit-image
```

## Data Structure

The project expects a specific directory structure for training data:

```
brain_tumor/
├── Training/
│   ├── no_tumor/           # Contains MRI images without tumors
│   └── pituitary_tumor/    # Contains MRI images with pituitary tumors
├── Testing/
│   ├── no_tumor/           # Contains test MRI images without tumors
│   └── pituitary_tumor/    # Contains test MRI images with pituitary tumors
```

The `load_data` method currently loads images from `brain_tumor/Training/no_tumor/` and `brain_tumor/Training/pituitary_tumor/`. Ensure your training images are placed in these directories.

## Usage

To run the brain tumor detector, execute the `test1.py` script:

```bash
python test1.py
```

The script will:
1.  Load training data (up to 500 images each for 'no_tumor' and 'pituitary_tumor').
2.  Train an SVM model.
3.  Present a menu:
    *   **1. Test an MRI image**: Prompts you to enter the path to an MRI image file. The system will then predict, segment, and analyze the tumor, displaying various visualizations and a detailed report.
    *   **2. Exit**: Exits the application.

### Example Walkthrough

1.  Run `python test1.py`.
2.  The model will train.
3.  When prompted, enter `1` to test an image.
4.  Provide the path to an MRI image, for example: `brain_tumor/Testing/pituitary_tumor/image(1).jpg`
5.  The system will output the prediction, confidence, tumor percentage, and display a series of plots and a detailed report.

**Important Note:** The model is a "Simple Brain Tumor Detector" and is intended for demonstration and educational purposes only. **It is not a substitute for professional medical advice, diagnosis, or treatment.** Always consult with a qualified medical professional for any health concerns.

## How it Works

The `SimpleBrainTumorDetector` class encapsulates the entire detection pipeline:

1.  **`load_data()`**: Reads grayscale MRI images from specified training directories, resizes them to 200x200 pixels, flattens them, normalizes pixel values, and assigns labels (0 for no tumor, 1 for tumor).
2.  **`train()`**:
    *   Splits the loaded data into training and testing sets.
    *   Trains a Support Vector Classifier (SVC) with an RBF kernel.
    *   Evaluates the model's accuracy and provides a classification report.
3.  **`segment_tumor(image)`**:
    *   Applies median blurring for noise reduction.
    *   Uses Otsu's thresholding to convert the image to binary.
    *   Performs morphological opening and dilation to refine the background.
    *   Applies distance transform and watershed algorithm for robust tumor segmentation.
    *   Removes small contours (noise) from the segmented mask.
    *   Calculates the tumor percentage based on pixel count.
4.  **`analyze_tumor_size(image, tumor_mask)`**:
    *   Finds contours in the tumor mask.
    *   Sorts contours by area to identify the largest tumor.
    *   Calculates total tumor area percentage, tumor count, largest tumor area percentage, center location, and bounding box of the largest tumor.
5.  **`predict(image_path)`**:
    *   Loads and preprocesses a single MRI image.
    *   Uses the trained SVM to predict whether a tumor is present and its probability.
    *   Calls `segment_tumor` and `analyze_tumor_size` to get detailed information about the tumor.
    *   Returns a dictionary containing all prediction and analysis results, including the segmented image.
6.  **`overlay_segmentation(image, mask)`**: Creates a colored overlay of the segmented tumor area (red) on the original grayscale image for better visualization.
7.  **`main()`**: The entry point of the script, which orchestrates the training, prediction, and user interaction loop. It also handles error management and displays results graphically.
