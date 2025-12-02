import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from skimage import filters, measure, morphology

# Simple Brain Tumor Detector
class SimpleBrainTumorDetector:
    def __init__(self):
        self.model = None
        self.img_size = (200, 200)
    
    def load_data(self):
        """Load all training images"""
        print("Loading data...")
        X, y = [], []
        
        # Load no tumor images
        no_tumor_path = 'brain_tumor/Training/no_tumor/'
        if os.path.exists(no_tumor_path):
            for img_file in os.listdir(no_tumor_path)[:500]:  # Limit to 500 images
                img = cv2.imread(os.path.join(no_tumor_path, img_file), 0)
                if img is not None:
                    img = cv2.resize(img, self.img_size)
                    X.append(img.flatten() / 255.0)
                    y.append(0)
        
        # Load tumor images
        tumor_path = 'brain_tumor/Training/pituitary_tumor/'
        if os.path.exists(tumor_path):
            for img_file in os.listdir(tumor_path)[:500]:  # Limit to 500 images
                img = cv2.imread(os.path.join(tumor_path, img_file), 0)
                if img is not None:
                    img = cv2.resize(img, self.img_size)
                    X.append(img.flatten() / 255.0)
                    y.append(1)
        
        return np.array(X), np.array(y)
    
    def train(self):
        """Train the model"""
        X, y = self.load_data()
        print(f"Data shape: {X.shape}")
        print(f"Class 0 (No Tumor): {sum(y==0)}")
        print(f"Class 1 (Tumor): {sum(y==1)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train SVM
        print("\nTraining model...")
        self.model = SVC(kernel='rbf', probability=True, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"\nTraining Accuracy: {train_score:.2%}")
        print(f"Testing Accuracy: {test_score:.2%}")
        
        y_pred = self.model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['No Tumor', 'Tumor']))
        
        print("\n Model trained successfully!")
        return self
    
    def segment_tumor(self, image):
        """Segment tumor from MRI image and calculate percentage"""
        # Preprocess image
        img = cv2.medianBlur(image, 5)
        
        # Apply thresholding
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Remove small noise
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        # Find unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        
        # Mark the region of unknown with zero
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), markers)
        
        # Create mask for tumor region
        tumor_mask = np.zeros_like(image, dtype=np.uint8)
        tumor_mask[markers == -1] = 255
        
        # Remove small regions (noise)
        contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 50:  # Remove small areas
                cv2.drawContours(tumor_mask, [contour], -1, 0, -1)
        
        # Calculate tumor percentage
        total_pixels = image.size
        tumor_pixels = np.sum(tumor_mask > 0)
        tumor_percentage = (tumor_pixels / total_pixels) * 100
        
        return tumor_mask, tumor_percentage
    
    def analyze_tumor_size(self, image, tumor_mask):
        """Analyze tumor size and location"""
        # Find contours of tumor regions
        contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return {
                'area_percentage': 0.0,
                'tumor_count': 0,
                'largest_area': 0.0,
                'center_location': None,
                'bounding_box': None
            }
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Get the largest tumor
        largest_contour = contours[0]
        
        # Calculate area percentage
        total_area = image.shape[0] * image.shape[1]
        tumor_area = sum(cv2.contourArea(cnt) for cnt in contours)
        area_percentage = (tumor_area / total_area) * 100
        
        # Calculate center of largest tumor
        M = cv2.moments(largest_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            center_location = (cx, cy)
        else:
            center_location = (image.shape[1]//2, image.shape[0]//2)
        
        # Get bounding box of largest tumor
        x, y, w, h = cv2.boundingRect(largest_contour)
        bounding_box = (x, y, w, h)
        
        return {
            'area_percentage': area_percentage,
            'tumor_count': len(contours),
            'largest_area': (cv2.contourArea(largest_contour) / total_area) * 100,
            'center_location': center_location,
            'bounding_box': bounding_box
        }
    
    def predict(self, image_path):
        """Predict tumor for a single image"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Read and preprocess image
        img = cv2.imread(image_path, 0)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        original_img = img.copy()
        img_resized = cv2.resize(img, self.img_size)
        img_flat = img_resized.flatten().reshape(1, -1) / 255.0
        
        # Predict using classifier
        prediction = self.model.predict(img_flat)[0]
        proba = self.model.predict_proba(img_flat)[0]
        
        # Segment tumor and calculate percentage
        tumor_mask, tumor_percentage = self.segment_tumor(img_resized)
        
        # Analyze tumor size and characteristics
        tumor_analysis = self.analyze_tumor_size(img_resized, tumor_mask)
        
        result = {
            'has_tumor': bool(prediction),
            'prediction': 'Tumor Detected' if prediction == 1 else 'No Tumor',
            'confidence': max(proba),
            'no_tumor_prob': proba[0],
            'tumor_prob': proba[1],
            'tumor_percentage': tumor_percentage,
            'tumor_analysis': tumor_analysis,
            'original_image': original_img,
            'processed_image': img_resized,
            'tumor_mask': tumor_mask,
            'segmented_image': self.overlay_segmentation(img_resized, tumor_mask)
        }
        
        return result
    
    def overlay_segmentation(self, image, mask):
        """Overlay tumor segmentation on original image"""
        # Create colored overlay
        colored_overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Create red overlay for tumor regions
        red_overlay = np.zeros_like(colored_overlay)
        red_overlay[mask > 0] = [0, 0, 255]  # Red color
        
        # Blend overlay with original
        alpha = 0.3
        result = cv2.addWeighted(colored_overlay, 1, red_overlay, alpha, 0)
        
        return result

def main():
    """Main function with simple menu"""
    print("="*60)
    print("ADVANCED BRAIN TUMOR DETECTOR")
    print("="*60)
    print("Features:")
    print("- Tumor Classification (Yes/No)")
    print("- Tumor Segmentation")
    print("- Tumor Area Percentage Calculation")
    print("- Tumor Size Analysis")
    print("="*60)
    
    # Create and train detector
    detector = SimpleBrainTumorDetector()
    detector.train()
    
    while True:
        print("\n" + "="*60)
        print("1. Test an MRI image")
        print("2. Exit")
        print("="*60)
        
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == '1':
            img_path = input("Enter image path: ").strip()
            
            if not os.path.exists(img_path):
                print(f"Error: File '{img_path}' not found!")
                continue
            
            try:
                # Predict
                result = detector.predict(img_path)
                
                print("\n" + "="*60)
                print("PREDICTION RESULT:")
                print("="*60)
                print(f"Result: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.2%}")
                print(f"No Tumor Probability: {result['no_tumor_prob']:.4f}")
                print(f"Tumor Probability: {result['tumor_prob']:.4f}")
                print("-"*40)
                
                if result['has_tumor']:
                    print("TUMOR ANALYSIS:")
                    print("-"*40)
                    print(f"Tumor Area in Image: {result['tumor_percentage']:.2f}%")
                    print(f"Total Tumor Count: {result['tumor_analysis']['tumor_count']}")
                    print(f"Largest Tumor Area: {result['tumor_analysis']['largest_area']:.2f}%")
                    
                    # Size classification
                    tumor_percent = result['tumor_percentage']
                    if tumor_percent < 5:
                        size_category = "Very Small"
                    elif tumor_percent < 15:
                        size_category = "Small"
                    elif tumor_percent < 30:
                        size_category = "Medium"
                    elif tumor_percent < 50:
                        size_category = "Large"
                    else:
                        size_category = "Very Large"
                    
                    print(f"Size Category: {size_category}")
                    print("-"*40)
                    print("WARNING: Possible tumor detected!")
                    print("   Please consult a medical professional immediately.")
                else:
                    print("No tumor detected.")
                    print("   The MRI scan appears normal.")
                
                # Display image with multiple views
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                
                # Original Image
                axes[0, 0].imshow(result['original_image'], cmap='gray')
                axes[0, 0].set_title('Original MRI Image')
                axes[0, 0].axis('off')
                
                # Processed Image
                axes[0, 1].imshow(result['processed_image'], cmap='gray')
                axes[0, 1].set_title('Processed Image (200x200)')
                axes[0, 1].axis('off')
                
                # Tumor Mask
                axes[0, 2].imshow(result['tumor_mask'], cmap='gray')
                axes[0, 2].set_title('Tumor Segmentation Mask')
                axes[0, 2].axis('off')
                
                # Segmented Overlay
                axes[1, 0].imshow(cv2.cvtColor(result['segmented_image'], cv2.COLOR_BGR2RGB))
                axes[1, 0].set_title('Tumor Overlay (Red Areas)')
                axes[1, 0].axis('off')
                
                # Tumor Percentage Visualization
                if result['has_tumor']:
                    # Create pie chart for tumor vs normal area
                    tumor_area = result['tumor_percentage']
                    normal_area = 100 - tumor_area
                    
                    labels = ['Tumor Area', 'Normal Brain Tissue']
                    sizes = [tumor_area, normal_area]
                    colors = ['red', 'green']
                    
                    axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                    axes[1, 1].set_title(f'Tumor Area: {tumor_area:.1f}%')
                
                # Text Summary
                axes[1, 2].axis('off')
                summary_text = f"""
                DIAGNOSIS SUMMARY:
                
                Prediction: {result['prediction']}
                Confidence: {result['confidence']:.1%}
                
                Tumor Probability: {result['tumor_prob']:.2%}
                No Tumor Probability: {result['no_tumor_prob']:.2%}
                
                """
                
                if result['has_tumor']:
                    summary_text += f"""
                    TUMOR DETAILS:
                    
                    Tumor Area: {result['tumor_percentage']:.2f}%
                    Tumor Count: {result['tumor_analysis']['tumor_count']}
                    Largest Tumor: {result['tumor_analysis']['largest_area']:.2f}%
                    
                    RECOMMENDATION:
                    Consult a neurosurgeon immediately
                    Further MRI with contrast recommended
                    """
                else:
                    summary_text += """
                    RECOMMENDATION:
                    No immediate concern
                    Regular follow-up recommended
                    """
                
                axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes,
                               fontsize=10, verticalalignment='center',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                # Main title with color coding
                title_color = 'red' if result['has_tumor'] else 'green'
                plt.suptitle(f"Brain Tumor Analysis: {result['prediction']}", 
                           fontsize=18, fontweight='bold', color=title_color)
                
                plt.tight_layout()
                plt.show()
                
                # Print detailed report
                print("\n" + "="*60)
                print("DETAILED REPORT:")
                print("="*60)
                if result['has_tumor']:
                    print(f"Tumor occupies {result['tumor_percentage']:.2f}% of visible brain area")
                    print(f"Number of tumor regions: {result['tumor_analysis']['tumor_count']}")
                    print(f"Largest tumor size: {result['tumor_analysis']['largest_area']:.2f}% of image")
                    
                    # Severity assessment
                    if result['tumor_percentage'] < 10:
                        severity = "Mild"
                        urgency = "Schedule appointment within 1-2 weeks"
                    elif result['tumor_percentage'] < 25:
                        severity = "Moderate"
                        urgency = "Schedule appointment within 3-5 days"
                    else:
                        severity = "Severe"
                        urgency = "Seek immediate medical attention"
                    
                    print(f"Severity Assessment: {severity}")
                    print(f"Recommendation: {urgency}")
                
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
        
        elif choice == '2':
            print("\nThank you for using Brain Tumor Detector!")
            break
        
        else:
            print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":    
    main()