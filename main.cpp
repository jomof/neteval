#include <cstdio>
#include <cstdlib>
#include <random>
#include <cmath>

// Neural network parameters
typedef struct {
    int input_size;    // Number of input features
    int hidden_size;   // Number of neurons in the hidden layer
    int output_size;   // Number of output classes

    float* input_to_hidden_weights;   // [input_size][hidden_size]
    float* hidden_biases;             // [hidden_size]
    float* hidden_to_output_weights;  // [hidden_size][output_size]
    float* output_biases;             // [output_size]
} NeuralNetwork;

// Batch data structure for inputs and outputs
typedef struct {
    int batch_size;
    float* inputs;   // [batch_size][input_size]
    float* outputs;  // [batch_size][output_size]
} Batch;

void initialize_weights(float* array, int size) {
    if (array == nullptr) return;

    static std::default_random_engine engine(42);
    std::uniform_real_distribution distribution(-0.5f, 0.5f);

    for (int i = 0; i < size; i++) {
        array[i] = distribution(engine);
    }
}

NeuralNetwork* initialize_network(int input_size, int hidden_size, int output_size) {
    auto* nn = (NeuralNetwork*) malloc(sizeof(NeuralNetwork));
    nn->input_size = input_size;
    nn->hidden_size = hidden_size;
    nn->output_size = output_size;

    // Allocate memory for weights and biases
    nn->input_to_hidden_weights = (float*) malloc(input_size * hidden_size * sizeof(float));
    nn->hidden_biases = (float*) malloc(hidden_size * sizeof(float));
    nn->hidden_to_output_weights = (float*) malloc(hidden_size * output_size * sizeof(float));
    nn->output_biases = (float*) malloc(output_size * sizeof(float));

    // Initialize weights and biases
    initialize_weights(nn->input_to_hidden_weights, input_size * hidden_size);
    initialize_weights(nn->hidden_biases, hidden_size);
    initialize_weights(nn->hidden_to_output_weights, hidden_size * output_size);
    initialize_weights(nn->output_biases, output_size);

    return nn;
}

Batch* initialize_batch(int batch_size, int input_size, int output_size) {
    auto* batch = static_cast<Batch *>(malloc(sizeof(Batch)));
    batch->batch_size = batch_size;
    batch->inputs = (float*) malloc(batch_size * input_size * sizeof(float));
    batch->outputs = (float*) malloc(batch_size * output_size * sizeof(float));
    return batch;
}

void matvec_mult(const float* input, const float* weights, const float* bias, float* output,
                 int batch_size, int input_size, int output_size) {
    if (bias == nullptr || input == nullptr || weights == nullptr || output == nullptr) return;
    for (int b = 0; b < batch_size; b++) {
        for (int j = 0; j < output_size; j++) {
            float sum = bias[j];
            for (int i = 0; i < input_size; i++) {
                sum += input[b * input_size + i] * weights[i * output_size + j];
            }
            output[b * output_size + j] = sum;
        }
    }
}

void relu(float* data, int size) {
    if (data == nullptr) return;
    for (int i = 0; i < size; i++) {
        data[i] = fmaxf(0.0f, data[i]);
    }
}

void softmax(float* data, int batch_size, int output_size) {
    if (data == nullptr) return;
    for (int b = 0; b < batch_size; b++) {
        float max_val = -INFINITY;
        float sum = 0.0f;

        // Find max for numerical stability
        for (int j = 0; j < output_size; j++) {
            if (data[b * output_size + j] > max_val) {
                max_val = data[b * output_size + j];
            }
        }

        // Compute softmax
        for (int j = 0; j < output_size; j++) {
            data[b * output_size + j] = expf(data[b * output_size + j] - max_val);
            sum += data[b * output_size + j];
        }
        for (int j = 0; j < output_size; j++) {
            data[b * output_size + j] /= sum;
        }
    }
}

void forward(NeuralNetwork* nn, Batch* batch) {
    int batch_size = batch->batch_size;

    // Allocate intermediate outputs
    auto* hidden_outputs = static_cast<float *>(malloc(batch_size * nn->hidden_size * sizeof(float)));

    // Input to Hidden Layer
    matvec_mult(batch->inputs, nn->input_to_hidden_weights, nn->hidden_biases,
                hidden_outputs, batch_size, nn->input_size, nn->hidden_size);
    relu(hidden_outputs, batch_size * nn->hidden_size);

    // Hidden to Output Layer
    matvec_mult(hidden_outputs, nn->hidden_to_output_weights, nn->output_biases,
                batch->outputs, batch_size, nn->hidden_size, nn->output_size);
    softmax(batch->outputs, batch_size, nn->output_size);

    free(hidden_outputs);  // Free intermediate buffer
}

int main() {
    int input_size = 7840;    // Example: MNIST input size
    int hidden_size = 1280;   // Hidden layer neurons
    int output_size = 10;    // Output classes
    int batch_size = 320;     // Batch size for evaluation
    srand(42);

    NeuralNetwork* nn = initialize_network(input_size, hidden_size, output_size);
    Batch* batch = initialize_batch(batch_size, input_size, output_size);

    // Populate batch inputs with random data
    initialize_weights(batch->inputs, batch_size * input_size);

    // Forward pass
    forward(nn, batch);

    // Print first output vector
    for (int i = 0; i < output_size; i++) {
        printf("%f ", batch->outputs[i]);
    }
    printf("\n");

    // Cleanup
    free(nn->input_to_hidden_weights);
    free(nn->hidden_biases);
    free(nn->hidden_to_output_weights);
    free(nn->output_biases);
    free(nn);
    free(batch->inputs);
    free(batch->outputs);
    free(batch);

    return 0;
}
