import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

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
    
    def predict(self, image_path):
        """Predict tumor for a single image"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Read and preprocess image
        img = cv2.imread(image_path, 0)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        original_img = img.copy()
        img = cv2.resize(img, self.img_size)
        img_flat = img.flatten().reshape(1, -1) / 255.0
        
        # Predict
        prediction = self.model.predict(img_flat)[0]
        proba = self.model.predict_proba(img_flat)[0]
        
        result = {
            'has_tumor': bool(prediction),
            'prediction': 'Tumor Detected' if prediction == 1 else 'No Tumor',
            'confidence': max(proba),
            'no_tumor_prob': proba[0],
            'tumor_prob': proba[1],
            'original_image': original_img,
            'processed_image': img
        }
        
        return result

def main():
    """Main function with simple menu"""
    print("="*60)
    print("SIMPLE BRAIN TUMOR DETECTOR")
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
                
                print("\n" + "-"*40)
                print("PREDICTION RESULT:")
                print("-"*40)
                print(f"Result: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.2%}")
                print(f"No Tumor Probability: {result['no_tumor_prob']:.4f}")
                print(f"Tumor Probability: {result['tumor_prob']:.4f}")
                print("-"*40)
                
                if result['has_tumor']:
                    print("WARNING: Possible tumor detected!")
                    print("   Please consult a medical professional.")
                else:
                    print("No tumor detected.")
                    print("   The MRI scan appears normal.")
                
                # Display image
                plt.figure(figsize=(10, 4))
                
                plt.subplot(1, 2, 1)
                plt.imshow(result['original_image'], cmap='gray')
                plt.title(f'Original Image\n{os.path.basename(img_path)}')
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(result['processed_image'], cmap='gray')
                plt.title('Processed Image (200x200)')
                plt.axis('off')
                
                # Color code based on prediction
                title_color = 'red' if result['has_tumor'] else 'green'
                plt.suptitle(f"Diagnosis: {result['prediction']}", 
                           fontsize=16, fontweight='bold', color=title_color)
                
                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '2':
            print("\nThank you for using Brain Tumor Detector!")
            break
        
        else:
            print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()