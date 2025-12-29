// Generate reference logits from C implementation
// This outputs logits for comparison with Verilog implementation

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Config parameters and weight arrays (all defined in model_data.c)
extern int config_dim;
extern int config_hidden_dim;
extern int config_n_layers;
extern int config_n_heads;
extern int config_n_kv_heads;
extern int config_vocab_size;
extern int config_seq_len;

// Weight arrays
extern float token_embedding_table[];
extern float rms_att_weight[];
extern float wq[];
extern float wk[];
extern float wv[];
extern float wo[];
extern float rms_ffn_weight[];
extern float w1[];
extern float w2[];
extern float w3[];
extern float rms_final_weight[];
extern float wcls[];

// Runtime state
float *state_x;
float *state_xb;
float *state_xb2;
float *state_hb;
float *state_hb2;
float *state_q;
float *state_k;
float *state_v;
float *state_att;
float *state_logits;
float *state_key_cache;
float *state_value_cache;

void malloc_run_state() {
    int kv_dim = (config_dim * config_n_kv_heads) / config_n_heads;
    state_x = calloc(config_dim, sizeof(float));
    state_xb = calloc(config_dim, sizeof(float));
    state_xb2 = calloc(config_dim, sizeof(float));
    state_hb = calloc(config_hidden_dim, sizeof(float));
    state_hb2 = calloc(config_hidden_dim, sizeof(float));
    state_q = calloc(config_dim, sizeof(float));
    state_k = calloc(kv_dim, sizeof(float));
    state_v = calloc(kv_dim, sizeof(float));
    state_att = calloc(config_n_heads * config_seq_len, sizeof(float));
    state_logits = calloc(config_vocab_size, sizeof(float));
    state_key_cache = calloc(config_n_layers * config_seq_len * kv_dim, sizeof(float));
    state_value_cache = calloc(config_n_layers * config_seq_len * kv_dim, sizeof(float));
}

void free_run_state() {
    free(state_x);
    free(state_xb);
    free(state_xb2);
    free(state_hb);
    free(state_hb2);
    free(state_q);
    free(state_k);
    free(state_v);
    free(state_att);
    free(state_logits);
    free(state_key_cache);
    free(state_value_cache);
}

void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

float* forward(int token, int pos) {
    int dim = config_dim;
    int hidden_dim = config_hidden_dim;
    int head_size = dim / config_n_heads;
    int kv_dim = (config_dim * config_n_kv_heads) / config_n_heads;
    int kv_mul = config_n_heads / config_n_kv_heads;

    // Copy embedding into x
    float* content_row = token_embedding_table + token * dim;
    memcpy(state_x, content_row, dim * sizeof(float));

    // Forward through all layers
    for (int l = 0; l < config_n_layers; l++) {
        // Attention rmsnorm
        rmsnorm(state_xb, state_x, rms_att_weight + l * dim, dim);

        // QKV matmuls
        matmul(state_q, state_xb, wq + l * dim * dim, dim, dim);
        matmul(state_k, state_xb, wk + l * dim * kv_dim, dim, kv_dim);
        matmul(state_v, state_xb, wv + l * dim * kv_dim, dim, kv_dim);

        // RoPE rotation
        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? state_q : state_k;
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        // Save KV cache
        int loff = l * config_seq_len * kv_dim;
        float* key_cache_row = state_key_cache + loff + pos * kv_dim;
        float* value_cache_row = state_value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, state_k, kv_dim * sizeof(float));
        memcpy(value_cache_row, state_v, kv_dim * sizeof(float));

        // Multihead attention
        for (int h = 0; h < config_n_heads; h++) {
            float* q = state_q + h * head_size;
            float* att = state_att + h * config_seq_len;

            // Attention scores
            for (int t = 0; t <= pos; t++) {
                float* k = state_key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }

            // Softmax
            softmax(att, pos + 1);

            // Weighted sum of values
            float* xb = state_xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v = state_value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // Output projection
        matmul(state_xb2, state_xb, wo + l * dim * dim, dim, dim);

        // Residual connection
        for (int i = 0; i < dim; i++) {
            state_x[i] += state_xb2[i];
        }

        // FFN rmsnorm
        rmsnorm(state_xb, state_x, rms_ffn_weight + l * dim, dim);

        // FFN
        matmul(state_hb, state_xb, w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(state_hb2, state_xb, w3 + l * dim * hidden_dim, dim, hidden_dim);

        // SwiGLU
        for (int i = 0; i < hidden_dim; i++) {
            float val = state_hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= state_hb2[i];
            state_hb[i] = val;
        }

        // Final matmul
        matmul(state_xb, state_hb, w2 + l * hidden_dim * dim, hidden_dim, dim);

        // Residual
        for (int i = 0; i < dim; i++) {
            state_x[i] += state_xb[i];
        }
    }

    // Final rmsnorm
    rmsnorm(state_x, state_x, rms_final_weight, dim);

    // Classifier
    matmul(state_logits, state_x, token_embedding_table, dim, config_vocab_size);

    return state_logits;
}

int main(int argc, char *argv[]) {
    printf("Generating reference logits from C implementation\n");
    printf("==================================================\n\n");

    // Allocate state
    malloc_run_state();

    // Test token 1 (BOS) at position 0
    int token = 1;
    int pos = 0;

    printf("Running inference for token=%d, pos=%d\n", token, pos);

    float* logits = forward(token, pos);

    printf("Inference complete!\n\n");

    // Print top 20 logits
    printf("Top 20 logits:\n");
    for (int k = 0; k < 20; k++) {
        float max_logit = -1e30f;
        int max_idx = 0;
        for (int i = 0; i < config_vocab_size; i++) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                max_idx = i;
            }
        }
        printf("  [%5d] = %12.6f\n", max_idx, max_logit);
        logits[max_idx] = -1e30f;
    }

    // Write full logits to file for comparison
    FILE* f = fopen("reference_logits_token1_pos0.txt", "w");
    if (f) {
        // Restore logits
        logits = forward(token, pos);

        fprintf(f, "# Reference logits for token=%d, pos=%d\n", token, pos);
        fprintf(f, "# Format: index logit_value\n");
        for (int i = 0; i < config_vocab_size; i++) {
            fprintf(f, "%d %.8f\n", i, logits[i]);
        }
        fclose(f);
        printf("\n✓ Full logits written to: reference_logits_token1_pos0.txt\n");
    }

    free_run_state();

    printf("\n==================================================\n");
    printf("Reference generation complete!\n");

    return 0;
}
