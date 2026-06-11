import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress unnecessary TensorFlow logs
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def main():
    print("=== PROJECT 2: HANDWRITTEN DIGIT RECOGNIZER ===")
    
    # 1. Load and prepare the MNIST dataset
    print("📥 Loading MNIST Dataset...")
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # 2. Normalize pixel values to be between 0 and 1
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape images to specify the single grayscale channel (28x28x1)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    # 3. Design the Convolutional Neural Network (CNN) Architecture
    print("🏗️ Building Convolutional Neural Network (CNN)...")
    model = models.Sequential([
        # First convolutional layer
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and Dense Output Layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax') # 10 classes for digits 0-9
    ])
    
    # 4. Compile the Model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # 5. Train the Model
    epochs = 3
    print(f"🚀 Training model for {epochs} epochs...")
    model.fit(X_train, y_train, epochs=epochs, batch_size=64, validation_split=0.1)
    
    # 6. Evaluate Model Performance
    print("\n📈 Evaluating model on test data...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    
    # 7. Live Demonstration / Testing on 3 Sample Images
    print("\n🔮 Live Testing Demonstration (Sample Predictions):")
    sample_indices = [15, 42, 100] # Random pick items from test set
    
    for idx in sample_indices:
        sample_img = X_test[idx]
        actual_label = y_test[idx]
        
        # Expand dimensions to mimic a batch dimension (1, 28, 28, 1)
        input_data = np.expand_dims(sample_img, axis=0)
        prediction_probabilities = model.predict(input_data, verbose=0)
        predicted_label = np.argmax(prediction_probabilities)
        
        print(f"Sample Index {idx} -> Actual Label: {actual_label} | Predicted Label: {predicted_label}")

if __name__ == "__main__":
    main()
