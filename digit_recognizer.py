import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def main():
    print("=== PROJECT 2: HANDWRITTEN DIGIT RECOGNIZER ===")
    
    
    print("📥 Loading MNIST Dataset...")
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    
    print("🏗️ Building Convolutional Neural Network (CNN)...")
    model = models.Sequential([
        
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax') 
    ])
    
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    
    epochs = 3
    print(f"🚀 Training model for {epochs} epochs...")
    model.fit(X_train, y_train, epochs=epochs, batch_size=64, validation_split=0.1)
    
    
    print("\n📈 Evaluating model on test data...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    
    
    print("\n🔮 Live Testing Demonstration (Sample Predictions):")
    sample_indices = [15, 42, 100] 
    
    for idx in sample_indices:
        sample_img = X_test[idx]
        actual_label = y_test[idx]
        
        
        input_data = np.expand_dims(sample_img, axis=0)
        prediction_probabilities = model.predict(input_data, verbose=0)
        predicted_label = np.argmax(prediction_probabilities)
        
        print(f"Sample Index {idx} -> Actual Label: {actual_label} | Predicted Label: {predicted_label}")

if __name__ == "__main__":
    main()
