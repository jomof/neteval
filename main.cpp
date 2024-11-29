#include <cstdio>
#include <cstdlib>
#include <random>
#include <cmath>
#include <chrono>
#include <iostream>
#include <cassert>
#include <omp.h>

// Unaligned no Transpose: 5.98828e+08 elements/second
// ALIGN 64 with no Transpose: 5.8782e+08 elements/second
// Unaligned with Transpose: 1.46894e+09 elements/second
// ALIGN 32 with Transpose: 1.49653e+09 elements/second
// ALIGN 64 with Transpose: 1.49756e+09 elements/second
// ALIGN 128 with Transpose: 1.49422e+09 elements/second
// ALIGN 64 with Transpose and -ffast-math: 1.57071e+10 elements/second
// ALIGN 64 with Transpose and -ffast-math and OMP: 3.17545e+10 elements/second

#define ALIGN 64
#if ALIGN != 0
static_assert(ALIGN >= sizeof(void*) && (ALIGN & (ALIGN - 1)) == 0, "ALIGN must be a power of 2 and >= sizeof(void*)");
#define XALLOC(x) aligned_alloc_wrapper(ALIGN, x)
#else
#define XALLOC(x) malloc(x)
#endif

#define BATCH_SIZE 32
#define INPUT_SIZE 7840
#define HIDDEN_SIZE  (12800 * 10)
#define OUTPUT_SIZE 10

void* aligned_alloc_wrapper(size_t alignment, size_t size) {
    // Ensure the size is a multiple of alignment
    if (size % alignment != 0) {
        size = (size + alignment - 1) & ~(alignment - 1);
    }

    void* ptr = std::aligned_alloc(alignment, size);
    if (!ptr) {
        std::cerr << "std::aligned_alloc failed for alignment=" << alignment << ", size=" << size << std::endl;
        std::abort(); // Fail fast for debugging purposes
    }

    return ptr;
}


// Neural network parameters
typedef struct {
    float* input_to_hidden_weights;   // [INPUT_SIZE][HIDDEN_SIZE]
    float* hidden_biases;             // [HIDDEN_SIZE]
    float* hidden_to_output_weights;  // [HIDDEN_SIZE][OUTPUT_SIZE]
    float* output_biases;             // [OUTPUT_SIZE]
} NeuralNetwork;

// Batch data structure for inputs and outputs
typedef struct {
    float* inputs;   // [BATCH_SIZE][INPUT_SIZE]
    float* outputs;  // [BATCH_SIZE][OUTPUT_SIZE]
} Batch;

void initialize_weights(float* array, int size) {
    if (array == nullptr) return;

    static std::default_random_engine engine(42);
    std::uniform_real_distribution distribution(-0.5f, 0.5f);

    for (int i = 0; i < size; i++) {
        array[i] = distribution(engine);
    }
}



NeuralNetwork* initialize_network() {
    auto* nn = (NeuralNetwork*) XALLOC(sizeof(NeuralNetwork));

    // Allocate memory for weights and biases
    nn->input_to_hidden_weights = (float*) XALLOC(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    nn->hidden_biases = (float*) XALLOC(HIDDEN_SIZE * sizeof(float));
    nn->hidden_to_output_weights = (float*) XALLOC(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    nn->output_biases = (float*) XALLOC(OUTPUT_SIZE * sizeof(float));

    assert(nn->input_to_hidden_weights != nullptr);
    assert(nn->hidden_biases != nullptr);
    assert(nn->hidden_to_output_weights != nullptr);
    assert(nn->output_biases != nullptr);

    // Initialize weights and biases
    initialize_weights(nn->input_to_hidden_weights, INPUT_SIZE * HIDDEN_SIZE);
    initialize_weights(nn->hidden_biases, HIDDEN_SIZE);
    initialize_weights(nn->hidden_to_output_weights, HIDDEN_SIZE * OUTPUT_SIZE);
    initialize_weights(nn->output_biases, OUTPUT_SIZE);

    return nn;
}

Batch* initialize_batch() {
    auto* batch = static_cast<Batch *>(XALLOC(sizeof(Batch)));
    batch->inputs = (float*) XALLOC(BATCH_SIZE * INPUT_SIZE * sizeof(float));
    batch->outputs = (float*) XALLOC(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    return batch;
}

template<long T_INPUT_SIZE, long T_OUTPUT_SIZE>
long matvec_mult(const float* input, const float* transposed_weights, const float* bias, float* output) {
    long ops = 0;

    #pragma omp parallel
    {
        #pragma omp single
        std::cout << "Number of threads: " << omp_get_num_threads() << std::endl;
    }

    #pragma omp parallel for collapse(1) reduction(+:ops) schedule(static)
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int j = 0; j < T_OUTPUT_SIZE; j++) {
            float sum = bias[j];
            for (int i = 0; i < T_INPUT_SIZE; i++) {
                // Access transposed_weights sequentially
                sum += input[b * T_INPUT_SIZE + i] * transposed_weights[j * T_INPUT_SIZE + i];
            }
            output[b * T_OUTPUT_SIZE + j] = sum;
            ops += T_INPUT_SIZE + 1; // Count operations: input_size multiplications + 1 addition for bias
        }
    }

    return ops;
}

long relu(float* data, int size) {
    if (data == nullptr) return 0;
    for (int i = 0; i < size; i++) {
        data[i] = fmaxf(0.0f, data[i]);
    }
    return size;
}

long softmax(float* data) {
    if (data == nullptr) return 0;
    for (int b = 0; b < BATCH_SIZE; b++) {
        float max_val = -INFINITY;
        float sum = 0.0f;

        // Find max for numerical stability
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (data[b * OUTPUT_SIZE + j] > max_val) {
                max_val = data[b * OUTPUT_SIZE + j];
            }
        }

        // Compute softmax
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            data[b * OUTPUT_SIZE + j] = expf(data[b * OUTPUT_SIZE + j] - max_val);
            sum += data[b * OUTPUT_SIZE + j];
        }
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            data[b * OUTPUT_SIZE + j] /= sum;
        }
    }
    return BATCH_SIZE * OUTPUT_SIZE * 3;
}

long forward(NeuralNetwork* nn, Batch* batch) {
    long ops = 0;

    // Allocate intermediate outputs
    auto* hidden_outputs = static_cast<float *>(XALLOC(BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));

    // Input to Hidden Layer
    ops += matvec_mult<INPUT_SIZE, HIDDEN_SIZE>(batch->inputs, nn->input_to_hidden_weights, nn->hidden_biases, hidden_outputs);
    ops += relu(hidden_outputs, BATCH_SIZE * HIDDEN_SIZE);

    // Hidden to Output Layer
    ops += matvec_mult<HIDDEN_SIZE, OUTPUT_SIZE>(hidden_outputs, nn->hidden_to_output_weights, nn->output_biases, batch->outputs);
    ops += softmax(batch->outputs);

    free(hidden_outputs);  // Free intermediate buffer
    return ops;
}

int main() {
    srand(42);
    omp_set_num_threads(4);

    NeuralNetwork* nn = initialize_network();
    Batch* batch = initialize_batch();

    // Populate batch inputs with random data
    initialize_weights(batch->inputs, BATCH_SIZE * INPUT_SIZE);

    // Forward pass
    auto start = std::chrono::high_resolution_clock::now();
    auto ops = forward(nn, batch);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double throughput = ops / elapsed.count();

    // Print first output vector
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("%f ", batch->outputs[i]);
    }
    printf("\n");

    std::cout << "Throughput: " << throughput << " elements/second" << std::endl;
    std::cout << "Execution Time: " << elapsed.count() << " seconds" << std::endl;

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
