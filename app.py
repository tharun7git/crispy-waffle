#app.py:

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import streamlit as st
import matplotlib.pyplot as plt
import nibabel as nib  # For medical image formats
from sklearn.metrics import confusion_matrix
import seaborn as sns
import io
import base64
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Function to preprocess the uploaded image
def preprocess_image(image_file):
    """Preprocess the image for model input"""
    # Create a temporary file to save the uploaded file
    temp_file = f"temp_upload.{image_file.name.split('.')[-1]}"
    with open(temp_file, "wb") as f:
        f.write(image_file.getbuffer())
    
    # Handle different image formats
    if temp_file.endswith(('.nii', '.nii.gz')):
        # For NIfTI format
        img_data = nib.load(temp_file)
        img_array = img_data.get_fdata()
        
        # For simplicity, we'll take a middle slice from the 3D volume
        if len(img_array.shape) == 3:
            middle_slice = img_array.shape[2] // 2
            img = img_array[:, :, middle_slice]
        else:
            img = img_array
    else:
        # For standard image formats
        img = cv2.imread(temp_file, cv2.IMREAD_GRAYSCALE)
    
    # Store original image for display
    original_img = img.copy()
    
    # Resize to model input size
    img = cv2.resize(img, (224, 224))
    
    # Normalize
    img = img / 255.0
    
    # Expand dimensions for model input
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    
    # Remove temp file
    try:
        os.remove(temp_file)
    except:
        pass
    
    return img, original_img

def segment_tumor(img, model):
    """Advanced segmentation using GradCAM with improved implementation"""
    # Check input shape and convert if necessary
    if img.shape[-1] == 3:
        # Convert 3-channel RGB to 1-channel grayscale for model input
        img_for_model = img[:, :, :, 0:1]  # Just take first channel 
    else:
        img_for_model = img
    
    # Get the last convolutional layer in the model
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break
    
    if last_conv_layer is None:
        # If no convolutional layer is found, use a different approach or raise an error
        return None
    
    # Create a model that maps from input to both the output and the last convolutional layer
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )
    
    # Get the score for the target class
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_for_model)
        target_class = tf.argmax(predictions[0])
        target_class_score = predictions[0, target_class]
    
    # Calculate gradients
    grads = tape.gradient(target_class_score, conv_outputs)
    
    # Get mean gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight feature maps with gradients
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    # Resize heatmap to original image size
    heatmap = cv2.resize(heatmap, (224, 224))
    
    # Apply smoothing to the heatmap
    heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
    
    # Apply threshold to highlight only the most important parts
    _, binary_heatmap = cv2.threshold(heatmap, 0.5, 1, cv2.THRESH_BINARY)
    
    # Convert to RGB and apply colormap
    heatmap_colored = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    
    # Original image processing for overlay
    if img.shape[-1] == 3:
        orig_img = img[0]  # Already RGB
        # Ensure it's uint8 type
        if orig_img.dtype != np.uint8:
            orig_img = np.uint8(255 * orig_img)
    else:
        orig_img = img[0, :, :, 0]
        if orig_img.dtype != np.uint8:
            orig_img = np.uint8(255 * orig_img)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2RGB)
    
    # Ensure both images have the same data type and dimensions before overlay
    assert orig_img.shape == heatmap_colored.shape, "Image shapes don't match"
    assert orig_img.dtype == heatmap_colored.dtype, "Image types don't match"
    
    # Overlay heatmap on original image with transparency
    superimposed_img = cv2.addWeighted(orig_img, 0.6, heatmap_colored, 0.4, 0)
    
    # Draw contour around potential tumor region
    binary_heatmap = np.uint8(255 * binary_heatmap)
    contours, _ = cv2.findContours(binary_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the superimposed image
    cv2.drawContours(superimposed_img, contours, -1, (0, 255, 0), 2)
    
    return superimposed_img
def get_tumor_info(class_id):
    """Return information about the tumor type and potential causes"""
    tumor_info = {
        0: {
            "type": "Glioma",
            "description": "Gliomas are tumors that start in the glial cells of the brain or spine.",
            "causes": [
                "Genetic mutations",
                "Ionizing radiation exposure",
                "Family history of gliomas"
            ],
            "confidence_factors": [
                "Irregular shape and infiltrative appearance",
                "Location typically within brain tissue",
                "Often shows heterogeneous enhancement on MRI"
            ]
        },
        1: {
            "type": "Meningioma",
            "description": "Meningiomas arise from the meninges, the membranes that surround the brain and spinal cord.",
            "causes": [
                "Hormonal factors",
                "Genetic disorders like neurofibromatosis type 2",
                "Previous radiation therapy to the head"
            ],
            "confidence_factors": [
                "Well-defined, round or oval shape",
                "Typically attached to dura mater",
                "Homogeneous enhancement on MRI"
            ]
        },
        2: {
            "type": "No Tumor",
            "description": "No tumor detected in the provided scan.",
            "causes": [],
            "confidence_factors": [
                "Absence of abnormal mass or enhancement",
                "Normal brain anatomy observed"
            ]
        },
        3: {
            "type": "Pituitary",
            "description": "Pituitary adenomas are tumors that form in the pituitary gland at the base of the brain.",
            "causes": [
                "Multiple endocrine neoplasia type 1 (MEN1)",
                "Familial isolated pituitary adenoma (FIPA)",
                "Carney complex"
            ],
            "confidence_factors": [
                "Location in the sella turcica region",
                "Potential extension into suprasellar area",
                "Homogeneous enhancement with contrast"
            ]
        }
    }
    
    return tumor_info.get(class_id, {"type": "Unknown", "description": "Unknown tumor type", "causes": [], "confidence_factors": []})

# Convert image to base64 string for HTML display
def get_image_base64(img):
    # If img is a numpy array
    if isinstance(img, np.ndarray):
        # Convert to PIL Image
        img_pil = Image.fromarray(img)
        buffered = io.BytesIO()
        img_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    return None

# Function to handle analysis for web API
def analyze_image(file):
    try:
        # Check if model exists
        model_path = "best_model.h5"
        if not os.path.exists(model_path):
            return {"error": "Model not found. Please train the model first."}
            
        # Load the model
        model = load_model(model_path)
        
        # Preprocess the image
        processed_img, original_img = preprocess_image(file)
        
        # Get model prediction
        prediction = model.predict(processed_img)
        class_id = np.argmax(prediction[0])
        confidence = float(prediction[0, class_id] * 100)

        # === DEBUG PRINTS ===
        print("DEBUG - Prediction vector:", prediction[0])
        print("DEBUG - Predicted class index:", class_id, "Type:", type(class_id))
        # ====================

        # Get confidence scores for all classes
        all_confidences = {
            "Glioma": float(prediction[0, 0] * 100),
            "Meningioma": float(prediction[0, 1] * 100),
            "No Tumor": float(prediction[0, 2] * 100),
            "Pituitary Adenoma": float(prediction[0, 3] * 100)
        }
        
        # Get tumor information
        tumor_info = get_tumor_info(class_id)
        print("DEBUG - Tumor info:", tumor_info)

        # ... rest of your function ...
        
        # Convert original image for web display
        original_img_resized = cv2.resize(original_img, (224, 224))
        original_base64 = get_image_base64(original_img_resized)
        
        # Create segmentation
        # Use repeat instead of tf.image.grayscale_to_rgb
        img_3channel = np.repeat(processed_img, 3, axis=-1)
        segmented_img = segment_tumor(img_3channel, model)
        segmented_base64 = None
        if segmented_img is not None:
            segmented_base64 = get_image_base64(segmented_img)
        
        # Prepare response
        response = {
            "result": tumor_info["type"],
            "confidence": confidence,
            "description": tumor_info["description"],
            "causes": tumor_info["causes"],
            "confidence_factors": tumor_info["confidence_factors"],
            "original_image": original_base64,
            "segmented_image": segmented_base64
        }
        
        return response
        
    except Exception as e:
        import traceback
        return {"error": f"An error occurred: {str(e)}\n{traceback.format_exc()}"}

# Main Streamlit app
def main():
    st.set_page_config(page_title="Brain Tumor Detection System", layout="wide")
    
    st.title("Brain Tumor Detection System")
    
    # Add tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Analysis", "Batch Processing", "Model Performance"])
    
    with tab1:
        # File uploader
        st.header("Upload MRI Scan")
        uploaded_file = st.file_uploader("Choose an MRI scan image", 
                                       type=["jpg", "jpeg", "png", "dcm", "nii", "nii.gz"],
                                       key="single_upload")
        
        # Check if model exists
        model_path = "best_model.h5"
        model_exists = os.path.exists(model_path)
        
        if not model_exists:
            st.warning("Model file not found. Please train the model first or place the model file in the current directory.")
            
            # Placeholder for model training button or instructions
            if st.button("Create Sample Model (Demo Only)"):
                st.info("Creating a sample model for demonstration purposes...")
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
                
                model = Sequential([
                    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
                    MaxPooling2D((2, 2)),
                    Conv2D(64, (3, 3), activation='relu'),
                    MaxPooling2D((2, 2)),
                    Conv2D(128, (3, 3), activation='relu'),
                    MaxPooling2D((2, 2)),
                    Flatten(),
                    Dense(128, activation='relu'),
                    Dropout(0.5),
                    Dense(3, activation='softmax')  # 3 classes
                ])
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                model.save(model_path)
                st.success("Sample model created! You can now upload an image for analysis.")
                model_exists = True
        
        # Process uploaded file
        if uploaded_file is not None and model_exists:
            # Display spinner during processing
            with st.spinner("Processing MRI scan..."):
                # Load the model
                model = load_model(model_path)
                
                # Preprocess the image
                processed_img, original_img = preprocess_image(uploaded_file)
                
                # Get model prediction
                prediction = model.predict(processed_img)
                class_id = np.argmax(prediction[0])
                confidence = float(prediction[0, class_id] * 100)
                
                # Get confidence scores for all classes
                all_confidences = {
                    "Glioma": float(prediction[0, 0] * 100),
                    "Meningioma": float(prediction[0, 1] * 100),
                    "No Tumor": float(prediction[0, 2] * 100),
                    "Pituitary Adenoma": float(prediction[0, 3] * 100)
                }
                
                # Get tumor information
                tumor_info = get_tumor_info(class_id)
            
            # Display results
            st.header("Analysis Results")
            
            # Create two columns for images
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original MRI Scan")
                st.image(original_img, width=350, caption="Original MRI Scan")
            
            # Create segmentation
            with col2:
                st.subheader("Tumor Analysis")
                # Use repeat instead of tf.image.grayscale_to_rgb
                img_3channel = np.repeat(processed_img, 3, axis=-1)
                segmented_img = segment_tumor(img_3channel, model)
                if segmented_img is not None:
                    st.image(segmented_img, width=350, caption="Tumor Segmentation")
                else:
                    st.error("Segmentation failed. Could not find convolutional layers in model.")
            
            # Display confidence scores for all classes in a bar chart
            st.subheader("Confidence Scores")
            
            # Sort classes by confidence
            sorted_confidences = {k: v for k, v in sorted(all_confidences.items(), 
                                                         key=lambda item: item[1], 
                                                         reverse=True)}
            
            # Create a bar chart
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(sorted_confidences.keys(), sorted_confidences.values())
            
            # Color the highest confidence bar differently
            highest_class = max(all_confidences, key=all_confidences.get)
            for i, bar in enumerate(bars):
                if list(sorted_confidences.keys())[i] == highest_class:
                    bar.set_color('green')
                else:
                    bar.set_color('lightgrey')
            
            ax.set_ylabel('Confidence (%)')
            ax.set_title('Prediction Confidence by Class')
            ax.set_ylim(0, 100)
            
            # Add value labels on top of bars
            for i, v in enumerate(sorted_confidences.values()):
                ax.text(i, v + 2, f"{v:.1f}%", ha='center')
                
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display tumor information
            st.header(f"Result: {tumor_info['type']}")
            st.subheader(f"Confidence: {confidence:.2f}%")
            
            st.write(tumor_info['description'])
            
            # Potential causes
            if tumor_info['causes']:
                st.subheader("Potential Causes:")
                for cause in tumor_info['causes']:
                    st.markdown(f"- {cause}")
            
            # Confidence factors
            if tumor_info['confidence_factors']:
                st.subheader("Determining Factors:")
                for factor in tumor_info['confidence_factors']:
                    st.markdown(f"- {factor}")
            
            # Additional information
            st.info("Note: This analysis is based on machine learning predictions and should be confirmed by a medical professional.")
    
    with tab2:
        st.header("Batch Processing")
        st.write("Upload multiple MRI scans for batch analysis.")
        
        # Folder path input
        folder_path = st.text_input("Enter the path to a folder containing MRI scans:")
        
        if st.button("Process Folder") and folder_path:
            if os.path.exists(folder_path):
                # Check if model exists
                if not os.path.exists("best_model.h5"):
                    st.error("Model file not found. Please train the model first.")
                else:
                    # Load the model
                    model = load_model("best_model.h5")
                    
                    # Get list of image files
                    valid_extensions = ['.jpg', '.jpeg', '.png', '.dcm', '.nii', '.nii.gz']
                    image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f.lower())[1] in valid_extensions]
                    
                    if not image_files:
                        st.error("No valid image files found in the directory.")
                    else:
                        # Process each image
                        results = []
                        for i, img_file in enumerate(image_files):
                            # Update progress bar
                            progress_text = st.empty()
                            progress_text.text(f"Processing {i+1}/{len(image_files)}: {img_file}")
                            
                            # Load and preprocess the image
                            img_path = os.path.join(folder_path, img_file)
                            
                            # Create temporary file-like object
                            class TempFile:
                                def __init__(self, path, name):
                                    self.path = path
                                    self.name = name
                                    
                                def getbuffer(self):
                                    with open(self.path, 'rb') as f:
                                        return f.read()
                            
                            temp_file = TempFile(img_path, img_file)
                            
                            # Preprocess
                            try:
                                processed_img, _ = preprocess_image(temp_file)
                                
                                # Get prediction
                                prediction = model.predict(processed_img)
                                class_id = np.argmax(prediction[0])
                                confidence = float(prediction[0, class_id] * 100)
                                
                                # Get tumor type
                                tumor_info = get_tumor_info(class_id)
                                tumor_type = tumor_info['type']
                                
                                # Store results
                                results.append({
                                    'filename': img_file,
                                    'tumor_type': tumor_type,
                                    'confidence': confidence,
                                    'class_id': int(class_id)
                                })
                            except Exception as e:
                                results.append({
                                    'filename': img_file,
                                    'tumor_type': 'Error',
                                    'confidence': 0.0,
                                    'class_id': -1,
                                    'error': str(e)
                                })
                        
                        # Clear progress text
                        progress_text.empty()
                        
                        # Display results as table
                        st.subheader("Batch Processing Results")
                        
                        # Convert results to dataframe
                        import pandas as pd
                        results_df = pd.DataFrame(results)
                        
                        # Display table
                        st.dataframe(results_df[['filename', 'tumor_type', 'confidence']])
                        
                        # Show distribution of predictions
                        st.subheader("Prediction Distribution")
                        
                        # Filter out errors
                        valid_results = [r for r in results if r['class_id'] >= 0]
                        if valid_results:
                            # Count by tumor type
                            tumor_counts = {}
                            for r in valid_results:
                                if r['tumor_type'] not in tumor_counts:
                                    tumor_counts[r['tumor_type']] = 0
                                tumor_counts[r['tumor_type']] += 1
                            
                            # Create pie chart
                            fig, ax = plt.subplots()
                            ax.pie(tumor_counts.values(), labels=tumor_counts.keys(), autopct='%1.1f%%')
                            ax.axis('equal')
                            st.pyplot(fig)
                        
                        # Option to save results to CSV
                        if st.button("Save Results to CSV"):
                            results_df.to_csv('batch_results.csv', index=False)
                            st.success("Results saved to batch_results.csv")
            else:
                st.error(f"The folder path does not exist: {folder_path}")
    
    with tab3:
        st.header("Model Performance")
        st.write("View confusion matrix and performance metrics for the model.")
        
        if os.path.exists("best_model.h5"):
            if st.button("Evaluate Model"):
                with st.spinner("Evaluating model on validation data..."):
                    # Load the model
                    model = load_model("best_model.h5")
                    
                    # Check if validation data exists
                    if not os.path.exists("datasets/validation"):
                        st.error("Validation data not found. Please set up your dataset directory.")
                    else:
                        # Prepare validation data
                        validation_datagen = ImageDataGenerator(rescale=1./255)
                        validation_generator = validation_datagen.flow_from_directory(
                            'datasets/validation',
                            target_size=(224, 224),
                            batch_size=16,
                            class_mode='categorical',
                            color_mode='grayscale',
                            shuffle=False
                        )
                        
                        # Get class indices
                        class_indices = validation_generator.class_indices
                        class_names = list(class_indices.keys())
                        
                        # Predict on validation data
                        y_true = validation_generator.classes
                        y_pred_probas = model.predict(validation_generator)
                        y_pred = np.argmax(y_pred_probas, axis=1)
                        
                        # Calculate and display confusion matrix
                        st.subheader("Confusion Matrix")
                        cm = confusion_matrix(y_true, y_pred)
                        
                        # Plot confusion matrix
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                                   xticklabels=class_names, yticklabels=class_names)
                        plt.ylabel('True Label')
                        plt.xlabel('Predicted Label')
                        st.pyplot(fig)
                        
                        # Calculate metrics
                        from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
                        
                        accuracy = accuracy_score(y_true, y_pred) * 100
                        precision = precision_score(y_true, y_pred, average='weighted') * 100
                        recall = recall_score(y_true, y_pred, average='weighted') * 100
                        f1 = f1_score(y_true, y_pred, average='weighted') * 100
                        
                        # Display metrics
                        st.subheader("Performance Metrics")
                        
                        metrics_col1, metrics_col2 = st.columns(2)
                        
                        with metrics_col1:
                            st.metric("Accuracy", f"{accuracy:.2f}%")
                            st.metric("Precision", f"{precision:.2f}%")
                            
                        with metrics_col2:
                            st.metric("Recall", f"{recall:.2f}%")
                            st.metric("F1 Score", f"{f1:.2f}%")
                        
                        # Detailed classification report
                        st.subheader("Detailed Classification Report")
                        report = classification_report(y_true, y_pred, target_names=class_names)
                        st.text(report)

# Flask Web API endpoint
def create_flask_app():
    from flask import Flask, request, jsonify, send_from_directory
    
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        # Read the index.html file
        try:
            with open('templates/index.html', 'r') as f:
                html_content = f.read()
                return html_content
        except:
            return "Index file not found. Please make sure templates/index.html exists."
    
    @app.route('/upload', methods=['POST'])
    def upload_file():
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
            
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
            
        if file:
            response = analyze_image(file)
            return jsonify(response)
    
    return app

# Run the app
if __name__ == "__main__":
    # Check if running as web service or Streamlit
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        app = create_flask_app()
        app.run(debug=True, port=5000)
    else:
        main()
