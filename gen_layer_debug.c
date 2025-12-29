/* Generate layer-by-layer intermediate outputs for comparison with Verilog */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

// Model configuration
typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

typedef struct {
    float* token_embedding_table;
    float* rms_att_weight;
    float* rms_ffn_weight;
    float* wq;
    float* wk;
    float* wv;
    float* wo;
    float* w1;
    float* w2;
    float* w3;
    float* rms_final_weight;
    float* wcls;
} TransformerWeights;

typedef struct {
    float *x;
    float *xb;
    float *xb2;
    float *hb;
    float *hb2;
    float *q;
    float *k;
    float *v;
    float *att;
    float *logits;
    float* key_cache;
    float* value_cache;
} RunState;

Config config;
TransformerWeights weights;
RunState state;
int fd;
float* data;
size_t file_size;

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
        if (x[i] > max_val) max_val = x[i];
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

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip freq_cis_real
    ptr += p->seq_len * head_size / 2; // skip freq_cis_imag
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoint(char* checkpoint) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(1); }
    if (fread(&config, sizeof(Config), 1, file) != 1) { exit(1); }
    int shared_weights = config.vocab_size > 0 ? 1 : 0;
    config.vocab_size = abs(config.vocab_size);
    fseek(file, 0, SEEK_END);
    file_size = ftell(file);
    fclose(file);
    fd = open(checkpoint, O_RDONLY);
    if (fd == -1) { fprintf(stderr, "open failed!\n"); exit(1); }
    data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(1); }
    float* weights_ptr = data + sizeof(Config)/sizeof(float);
    memory_map_weights(&weights, &config, weights_ptr, shared_weights);
}

void malloc_run_state() {
    int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
    state.x = calloc(config.dim, sizeof(float));
    state.xb = calloc(config.dim, sizeof(float));
    state.xb2 = calloc(config.dim, sizeof(float));
    state.hb = calloc(config.hidden_dim, sizeof(float));
    state.hb2 = calloc(config.hidden_dim, sizeof(float));
    state.q = calloc(config.dim, sizeof(float));
    state.k = calloc(config.dim, sizeof(float));
    state.v = calloc(config.dim, sizeof(float));
    state.key_cache = calloc(config.n_layers * config.seq_len * kv_dim, sizeof(float));
    state.value_cache = calloc(config.n_layers * config.seq_len * kv_dim, sizeof(float));
    state.att = calloc(config.n_heads * config.seq_len, sizeof(float));
    state.logits = calloc(config.vocab_size, sizeof(float));
}

void print_array(const char* name, float* arr, int n) {
    printf("%s first 10 values:\n", name);
    for (int i = 0; i < (n < 10 ? n : 10); i++) {
        printf("  [%d] = %.8f\n", i, arr[i]);
    }
}

void forward_with_debug(int token, int pos) {
    int dim = config.dim;
    int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
    int kv_mul = config.n_heads / config.n_kv_heads;
    int hidden_dim = config.hidden_dim;
    int head_size = dim / config.n_heads;

    // Token embedding
    float* content_row = weights.token_embedding_table + token * dim;
    memcpy(state.x, content_row, dim * sizeof(float));

    printf("\n=== EMBEDDING (token=%d) ===\n", token);
    print_array("Embedding", state.x, dim);

    // Process each layer
    for (int l = 0; l < config.n_layers; l++) {
        printf("\n========================================\n");
        printf("=== LAYER %d START ===\n", l);
        printf("========================================\n");

        print_array("Layer input (x)", state.x, dim);

        // Attention RMSNorm
        rmsnorm(state.xb, state.x, weights.rms_att_weight + l*dim, dim);
        printf("\n--- After attention RMSNorm ---\n");
        print_array("xb (normalized)", state.xb, dim);

        // QKV matmuls
        int loff = l * config.seq_len * kv_dim;
        state.k = state.key_cache + loff + pos * kv_dim;
        state.v = state.value_cache + loff + pos * kv_dim;
        matmul(state.q, state.xb, weights.wq + l*dim*dim, dim, dim);
        matmul(state.k, state.xb, weights.wk + l*dim*kv_dim, dim, kv_dim);
        matmul(state.v, state.xb, weights.wv + l*dim*kv_dim, dim, kv_dim);

        printf("\n--- After Q/K/V matmuls ---\n");
        print_array("q", state.q, dim);
        print_array("k", state.k, kv_dim);
        print_array("v", state.v, kv_dim);

        // RoPE
        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? state.q : state.k;
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }
        printf("\n--- After RoPE ---\n");
        print_array("q_rope", state.q, dim);
        print_array("k_rope", state.k, kv_dim);

        // Attention
        memset(state.xb, 0, dim * sizeof(float));
        for (int h = 0; h < config.n_heads; h++) {
            float* q = state.q + h * head_size;
            float* att = state.att + h * config.seq_len;
            for (int t = 0; t <= pos; t++) {
                float* k = state.key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }
            softmax(att, pos + 1);
            float* xb = state.xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v = state.value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }
        printf("\n--- After attention (weighted V) ---\n");
        print_array("xb (attention output)", state.xb, dim);

        // Output projection
        matmul(state.xb2, state.xb, weights.wo + l*dim*dim, dim, dim);
        printf("\n--- After Wo projection ---\n");
        print_array("xb2 (projected)", state.xb2, dim);

        // Residual
        for (int i = 0; i < dim; i++) {
            state.x[i] += state.xb2[i];
        }
        printf("\n--- After first residual ---\n");
        print_array("x (residual)", state.x, dim);

        // FFN RMSNorm
        rmsnorm(state.xb, state.x, weights.rms_ffn_weight + l*dim, dim);
        printf("\n--- After FFN RMSNorm ---\n");
        print_array("xb (normalized)", state.xb, dim);

        // FFN
        matmul(state.hb, state.xb, weights.w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(state.hb2, state.xb, weights.w3 + l*dim*hidden_dim, dim, hidden_dim);

        printf("\n--- After W1 and W3 ---\n");
        print_array("hb (W1 output)", state.hb, hidden_dim);
        print_array("hb2 (W3 output)", state.hb2, hidden_dim);

        // SwiGLU
        for (int i = 0; i < hidden_dim; i++) {
            float val = state.hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= state.hb2[i];
            state.hb[i] = val;
        }
        printf("\n--- After SwiGLU ---\n");
        print_array("hb (SwiGLU output)", state.hb, hidden_dim);

        // W2 projection
        matmul(state.xb, state.hb, weights.w2 + l*dim*hidden_dim, hidden_dim, dim);
        printf("\n--- After W2 ---\n");
        print_array("xb (W2 output)", state.xb, dim);

        // Final residual
        for (int i = 0; i < dim; i++) {
            state.x[i] += state.xb[i];
        }

        printf("\n========================================\n");
        printf("=== LAYER %d OUTPUT ===\n", l);
        printf("========================================\n");
        print_array("x (layer output)", state.x, dim);
    }

    // Final RMSNorm
    rmsnorm(state.x, state.x, weights.rms_final_weight, dim);
    printf("\n========================================\n");
    printf("=== AFTER FINAL RMSNORM ===\n");
    printf("========================================\n");
    print_array("x (after final rmsnorm)", state.x, dim);

    // Classifier
    matmul(state.logits, state.x, weights.wcls, dim, config.vocab_size);
    printf("\n========================================\n");
    printf("=== LOGITS (first 20 and key indices) ===\n");
    printf("========================================\n");
    for (int i = 0; i < 20; i++) {
        printf("  logits[%d] = %.8f\n", i, state.logits[i]);
    }

    // Find top logits
    float max_val = state.logits[0];
    int max_idx = 0;
    for (int i = 1; i < config.vocab_size; i++) {
        if (state.logits[i] > max_val) {
            max_val = state.logits[i];
            max_idx = i;
        }
    }
    printf("\n  TOP LOGIT: logits[%d] = %.8f\n", max_idx, max_val);
    printf("  logits[9038] = %.8f  (expected top in C)\n", state.logits[9038]);
    printf("  logits[3118] = %.8f\n", state.logits[3118]);
    printf("  logits[17143] = %.8f  (Verilog top)\n", state.logits[17143]);
}

int main(int argc, char** argv) {
    char* checkpoint_path = "stories15M.bin";
    if (argc > 1) checkpoint_path = argv[1];

    read_checkpoint(checkpoint_path);
    malloc_run_state();

    printf("Model config:\n");
    printf("  dim=%d, hidden_dim=%d\n", config.dim, config.hidden_dim);
    printf("  n_layers=%d, n_heads=%d, n_kv_heads=%d\n",
           config.n_layers, config.n_heads, config.n_kv_heads);
    printf("  vocab_size=%d, seq_len=%d\n", config.vocab_size, config.seq_len);

    // Run forward pass for BOS token (1) at position 0
    printf("\n\n");
    printf("################################################################\n");
    printf("# FORWARD PASS: token=1 (BOS), pos=0\n");
    printf("################################################################\n");
    forward_with_debug(1, 0);

    // Cleanup
    munmap(data, file_size);
    close(fd);

    return 0;
}
