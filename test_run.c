/* Tests for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

// Define TESTING before including run.c to avoid main() conflict
#define TESTING
#include "run.c"

// Helper function to generate text and capture output to a string
void generate_to_string(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
                        char *prompt, int steps, char *output, int max_output_len) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);

    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    int next;
    int token = prompt_tokens[0];
    int pos = 0;
    int output_pos = 0;

    while (pos < steps && output_pos < max_output_len - 1) {
        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // decode the token
        char* piece = decode(tokenizer, token, next);

        // Copy to output buffer (respecting boundaries)
        if (piece != NULL && piece[0] != '\0') {
            if (piece[1] == '\0') {
                unsigned char byte_val = piece[0];
                if (!(isprint(byte_val) || isspace(byte_val))) {
                    token = next;
                    continue; // skip non-printable single bytes
                }
            }
            int piece_len = strlen(piece);
            if (output_pos + piece_len < max_output_len - 1) {
                strcpy(output + output_pos, piece);
                output_pos += piece_len;
            }
        }

        token = next;
    }

    output[output_pos] = '\0';
    free(prompt_tokens);
}

// Test generation with seed 42
void test_generation_seed_42() {
    printf("Testing generation with seed=42 (100 steps)...\n");

    Transformer transformer;
    build_transformer(&transformer, "stories15M.bin");

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, "tokenizer.bin", transformer.config.vocab_size);

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, 1.0f, 0.9f, 42);

    char output[4096] = {0};
    generate_to_string(&transformer, &tokenizer, &sampler, NULL, 100, output, sizeof(output));

    const char *expected = "Once upon a time, there was a little girl named Lily. She loved to play in the park with her friends. One day, while playing, she saw a boy crying. She went over to him and asked him what was wrong. The boy said he had lost his ball and couldn't find it anywhere.\nLily felt compassionate and wanted to help the boy. She asked him where he last had it and he pointed to a bench. She went to";

    assert(strncmp(output, expected, strlen(expected)) == 0);
    printf("  PASSED: Output matches expected text\n");

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
}

// Test generation with seed 123
void test_generation_seed_123() {
    printf("Testing generation with seed=123 (50 steps)...\n");

    Transformer transformer;
    build_transformer(&transformer, "stories15M.bin");

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, "tokenizer.bin", transformer.config.vocab_size);

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, 1.0f, 0.9f, 123);

    char output[4096] = {0};
    generate_to_string(&transformer, &tokenizer, &sampler, NULL, 50, output, sizeof(output));

    const char *expected = "Once upon a time, there was a little girl named Lily. She loved to play outside in the snow. One day, she saw a big pile of snow and wanted to play in it. But she didn't have any fun bo";

    assert(strncmp(output, expected, strlen(expected)) == 0);
    printf("  PASSED: Output matches expected text\n");

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
}

// Test greedy generation (temperature = 0.0)
void test_greedy_generation() {
    printf("Testing greedy generation (temp=0.0, 80 steps)...\n");

    Transformer transformer;
    build_transformer(&transformer, "stories15M.bin");

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, "tokenizer.bin", transformer.config.vocab_size);

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, 0.0f, 0.9f, 42);

    char output[4096] = {0};
    generate_to_string(&transformer, &tokenizer, &sampler, NULL, 80, output, sizeof(output));

    const char *expected = "Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty.\nLily wanted to play with the ball, but it was too high up in the sky. She tried to jump and reach it, but";

    assert(strncmp(output, expected, strlen(expected)) == 0);
    printf("  PASSED: Output matches expected text\n");

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
}

// Test RNG determinism
void test_rng_determinism() {
    printf("Testing RNG determinism...\n");

    unsigned long long state1 = 42;
    unsigned long long state2 = 42;

    for (int i = 0; i < 100; i++) {
        unsigned int r1 = random_u32(&state1);
        unsigned int r2 = random_u32(&state2);
        assert(r1 == r2);
    }

    printf("  PASSED: RNG with same seed produces same sequence\n");
}

// Test softmax function
void test_softmax() {
    printf("Testing softmax function...\n");

    float x[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    softmax(x, 5);

    // Check that probabilities sum to approximately 1.0
    float sum = 0.0f;
    for (int i = 0; i < 5; i++) {
        sum += x[i];
        assert(x[i] > 0.0f && x[i] < 1.0f);
    }
    assert(fabs(sum - 1.0f) < 1e-6f);

    // Check that values are in ascending order (since input was ascending)
    for (int i = 0; i < 4; i++) {
        assert(x[i] < x[i+1]);
    }

    printf("  PASSED: Softmax normalizes correctly\n");
}

// Test RMS norm function
void test_rmsnorm() {
    printf("Testing RMS norm function...\n");

    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float weight[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float output[4];

    rmsnorm(output, x, weight, 4);

    // Check that output is normalized
    float ss = 0.0f;
    for (int i = 0; i < 4; i++) {
        ss += output[i] * output[i];
    }
    ss /= 4;

    // RMS should be close to 1.0 when weights are all 1.0
    assert(fabs(sqrtf(ss) - 1.0f) < 0.1f);

    printf("  PASSED: RMS normalization works correctly\n");
}

// Test matmul function with identity matrix
void test_matmul() {
    printf("Testing matrix multiplication...\n");

    // Simple 2x2 identity matrix multiply
    float x[2] = {1.0f, 2.0f};
    float w[4] = {1.0f, 0.0f, 0.0f, 1.0f}; // Identity matrix (row-major)
    float out[2];

    matmul(out, x, w, 2, 2);

    // With identity matrix, output should equal input
    assert(fabs(out[0] - 1.0f) < 1e-6f);
    assert(fabs(out[1] - 2.0f) < 1e-6f);

    printf("  PASSED: Matrix multiplication works correctly\n");
}

// Test sample_argmax function
void test_sample_argmax() {
    printf("Testing argmax sampling...\n");

    float probs[5] = {0.1f, 0.2f, 0.5f, 0.15f, 0.05f};
    int result = sample_argmax(probs, 5);

    assert(result == 2); // Index 2 has highest probability (0.5)
    printf("  PASSED: Argmax returns correct index\n");
}

int main() {
    printf("========================================\n");
    printf("  Llama-2 Transformer Tests\n");
    printf("========================================\n\n");

    // Test basic mathematical functions
    test_rng_determinism();
    test_softmax();
    test_rmsnorm();
    test_matmul();
    test_sample_argmax();

    printf("\n");

    // Test full generation with different seeds and parameters
    test_generation_seed_42();
    test_generation_seed_123();
    test_greedy_generation();

    printf("\n");
    printf("========================================\n");
    printf("  All Tests Passed Successfully!\n");
    printf("========================================\n");

    return 0;
}
