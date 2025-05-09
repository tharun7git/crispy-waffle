import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import streamlit as st

def create_tumor_model():
    """Create an improved CNN model for brain tumor detection"""
    # Define input shape for grayscale
    inputs = tf.keras.Input(shape=(224, 224, 1))
    
    # Manual conversion from grayscale to RGB by using a 1x1 convolution
    x = Conv2D(3, (1, 1), padding='same')(inputs)
    
    # Load pre-trained VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Make some of the base model layers trainable (fine-tuning)
    # First, set all layers to non-trainable
    for layer in base_model.layers:
        layer.trainable = False
        
    # Then, make the last 4 layers trainable
    for layer in base_model.layers[-4:]:
        layer.trainable = True
    
    # Apply the base model to our input
    x = base_model(x)
    
    # Add classification layers
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(4, activation='softmax')(x)  # 4 classes to match dataset
    
    # Create the full model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model with a lower learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def train_model(train_data_path, validation_data_path, epochs=20):
    """Train the model with data augmentation and improved monitoring"""
    # Create the model
    model = create_tumor_model()
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,  # MRI can be viewed from different angles
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]
    )
    
    # Just rescaling for validation
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    # Data generators
    train_generator = train_datagen.flow_from_directory(
        train_data_path,
        target_size=(224, 224),
        batch_size=16,  # Smaller batch size for better generalization
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=True
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_data_path,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=False  # Don't shuffle validation data
    )
    
    # Calculate class weights to handle imbalanced data
    class_indices = train_generator.class_indices
    classes = list(class_indices.keys())
    
    # Count samples in each class
    class_counts = [len(os.listdir(os.path.join(train_data_path, class_name))) 
                   for class_name in classes]
    
    # Compute class weights inversely proportional to class frequencies
    class_weights = {}
    total = sum(class_counts)
    
    for i, count in enumerate(class_counts):
        # Avoid division by zero
        if count > 0:
            class_weights[i] = total / (len(class_counts) * count)
        else:
            class_weights[i] = 1.0
    
    # Print class weights for debugging
    print("Class weights:", class_weights)
    
    # Enhanced callbacks
    checkpoint = ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=8,  # More patience
        restore_best_weights=True,
        verbose=1
    )
    
    # Add reduce learning rate on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
    
    # Train the model - REMOVED 'workers' and 'use_multiprocessing' params
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop, reduce_lr],
        class_weight=class_weights  # Using class weights
    )
    
    return history, model

def main():
    st.set_page_config(page_title="Brain Tumor Model Training", layout="wide")
    
    st.title("Brain Tumor Detection Model Training")
    
    st.write("""
    This application will train a deep learning model to detect brain tumors from MRI scans.
    Make sure you have your dataset organized in the following structure:
    
    ```
    datasets/
    ├── training/
    │   ├── glioma/
    │   ├── meningioma/ 
    │   ├── pituitary/
    │   ├── no_tumor/
    └── validation/
        ├── glioma/
        ├── meningioma/
        ├── pituitary/
        ├── no_tumor/
    ```
    """)
    
    # Check if dataset exists
    dataset_exists = os.path.exists("datasets/training") and os.path.exists("datasets/validation")
    
    if not dataset_exists:
        st.warning("Dataset folders not found. Please organize your data as described above.")
        
        # Create folders button
        if st.button("Create Directory Structure"):
            os.makedirs("datasets/training/glioma", exist_ok=True)
            os.makedirs("datasets/training/meningioma", exist_ok=True)
            os.makedirs("datasets/training/pituitary", exist_ok=True)
            os.makedirs("datasets/training/no_tumor", exist_ok=True)
            os.makedirs("datasets/validation/glioma", exist_ok=True)
            os.makedirs("datasets/validation/meningioma", exist_ok=True)
            os.makedirs("datasets/validation/pituitary", exist_ok=True)
            os.makedirs("datasets/validation/no_tumor", exist_ok=True)
            st.success("Directory structure created! Please add your images to these folders.")
    
    # Training parameters
    st.header("Training Parameters")
    
    epochs = st.slider("Number of Epochs", min_value=15, max_value=50, value=30)
    
    # Option to make base model trainable
    fine_tune = st.checkbox("Fine-tune VGG16 model (unfreeze last layers)", value=True)
    
    # Start training button
    if st.button("Start Training"):
        if not dataset_exists:
            st.error("Cannot start training. Dataset folders not found.")
        else:
            # Display dataset statistics first
            st.subheader("Dataset Statistics")
            
            # Count files in each class
            classes = ["glioma", "meningioma", "pituitary", "no_tumor"]
            train_counts = {}
            val_counts = {}
            
            for cls in classes:
                train_path = f"datasets/training/{cls}"
                val_path = f"datasets/validation/{cls}"
                
                if os.path.exists(train_path):
                    train_counts[cls] = len(os.listdir(train_path))
                else:
                    train_counts[cls] = 0
                
                if os.path.exists(val_path):
                    val_counts[cls] = len(os.listdir(val_path))
                else:
                    val_counts[cls] = 0
            
            # Display counts
            col1, col2 = st.columns(2)
            with col1:
                st.write("Training set:")
                for cls, count in train_counts.items():
                    st.write(f"- {cls}: {count} images")
                st.write(f"Total: {sum(train_counts.values())} images")
                
            with col2:
                st.write("Validation set:")
                for cls, count in val_counts.items():
                    st.write(f"- {cls}: {count} images")
                st.write(f"Total: {sum(val_counts.values())} images")
            
            # Training progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_display = st.empty()
            
            # Custom callback to update Streamlit components
            class StreamlitCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    
                    # Format metrics for display
                    metrics_str = f"Epoch {epoch+1}/{epochs} - "
                    metrics_str += f"loss: {logs.get('loss', 0):.4f} - "
                    metrics_str += f"accuracy: {logs.get('accuracy', 0):.4f} - "
                    metrics_str += f"val_loss: {logs.get('val_loss', 0):.4f} - "
                    metrics_str += f"val_accuracy: {logs.get('val_accuracy', 0):.4f}"
                    
                    status_text.text(metrics_str)
                    
                    # Create a dictionary to store metrics for visualization
                    metrics_dict = {
                        'Training Loss': logs.get('loss', 0),
                        'Validation Loss': logs.get('val_loss', 0),
                        'Training Accuracy': logs.get('accuracy', 0),
                        'Validation Accuracy': logs.get('val_accuracy', 0)
                    }
                    
                    # Display metrics as a chart
                    metrics_display.line_chart(
                        metrics_dict
                    )
            
            # Train the model
            try:
                with st.spinner("Training in progress..."):
                    # Custom callback
                    streamlit_callback = StreamlitCallback()
                    
                    # Add callbacks list including our custom Streamlit callback
                    history, model = train_model(
                        train_data_path='datasets/training',
                        validation_data_path='datasets/validation',
                        epochs=epochs
                    )
                    
                    # Evaluate on validation data to get per-class metrics
                    validation_datagen = ImageDataGenerator(rescale=1./255)
                    validation_generator = validation_datagen.flow_from_directory(
                        'datasets/validation',
                        target_size=(224, 224),
                        batch_size=16,
                        class_mode='categorical',
                        color_mode='grayscale',
                        shuffle=False
                    )
                    
                    results = model.evaluate(validation_generator)
                    
                # Display training results
                st.success("Training completed successfully!")
                
                # Save the model
                model.save('best_model.h5')
                
                # Plot training history
                st.subheader("Training History")
                
                # Convert history.history to lists if needed
                history_dict = {}
                for key in history.history:
                    history_dict[key] = history.history[key]
                
                # Create tabs for different metrics
                tab1, tab2, tab3 = st.tabs(["Accuracy", "Loss", "Precision/Recall"])
                
                with tab1:
                    # Plot training & validation accuracy
                    st.line_chart(
                        data={
                            'Training Accuracy': history_dict['accuracy'],
                            'Validation Accuracy': history_dict['val_accuracy']
                        }
                    )
                
                with tab2:
                    # Plot training & validation loss
                    st.line_chart(
                        data={
                            'Training Loss': history_dict['loss'],
                            'Validation Loss': history_dict['val_loss']
                        }
                    )
                
                with tab3:
                    # Plot precision and recall if available
                    if 'precision' in history_dict and 'recall' in history_dict:
                        st.line_chart(
                            data={
                                'Precision': history_dict['precision'],
                                'Recall': history_dict['recall'],
                                'Val Precision': history_dict['val_precision'],
                                'Val Recall': history_dict['val_recall']
                            }
                        )
                
                st.info("The model has been saved as 'best_model.h5'. You can now use it for predictions.")
                
            except Exception as e:
                st.error(f"An error occurred during training: {str(e)}")
                st.error("Traceback:")
                import traceback
                st.code(traceback.format_exc())

# Run the app
if __name__ == "__main__":
    main()
