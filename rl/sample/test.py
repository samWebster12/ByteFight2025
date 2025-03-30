import numpy as np
import matplotlib.pyplot as plt
from sample_cnn import NumPyCNN, predict

def visualize_sample_and_prediction(sample_input, true_label, predicted_label, index=0):
    """Visualize a sample image and its prediction"""
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(sample_input[index, 0], cmap='gray')
    plt.title(f"True: {true_label[index]}, Predicted: {predicted_label[index]}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # File paths - note the updated weight file name
    weights_file = 'weights/enhanced_cnn_weights.npz'
    sample_file = 'weights/sample_data.npz'
    pytorch_outputs_file = 'weights/layer_outputs.npz'
    
    # Create and load NumPy model
    numpy_model = NumPyCNN()
    numpy_model.load_weights(weights_file)
    
    # Verify the model
    #verify_all_layers()#(numpy_model, weights_file, sample_file, pytorch_outputs_file)
    
    # Load sample data
    sample_data = np.load(sample_file)
    sample_input = sample_data['inputs']
    sample_labels = sample_data['labels']
    
    # Make predictions
    predictions, probabilities = predict(numpy_model, sample_input)
    
    # Print predictions
    print("\nPredictions on sample data:")
    for i in range(len(predictions)):
        print(f"Sample {i+1}: True label = {sample_labels[i]}, " 
              f"Predicted = {predictions[i]}, "
              f"Confidence = {probabilities[i][predictions[i]]:.4f}")
    
    # Visualize a few samples
    for i in range(min(5, len(sample_input))):
        visualize_sample_and_prediction(sample_input, sample_labels, predictions, i)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == sample_labels)
    print(f"Accuracy on sample data: {accuracy:.4f}")
    
if __name__ == "__main__":
    main()