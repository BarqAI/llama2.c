/* Inference for Llama-2 Transformer model in pure C - FLATTENED VERSION */
/* All structs removed, binary data embedded as arrays */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>

// ----------------------------------------------------------------------------
// Model configuration - flattened (no Config struct)
// These are extern declarations - the actual arrays are in model_data.c
extern int config_dim;
extern int config_hidden_dim;
extern int config_n_layers;
extern int config_n_heads;
extern int config_n_kv_heads;
extern int config_vocab_size;
extern int config_seq_len;
extern int config_shared_weights;

// ----------------------------------------------------------------------------
// Model weights - flattened (no TransformerWeights struct)
// These are extern declarations - the actual arrays are in model_data.c
extern float token_embedding_table[];
extern float rms_att_weight[];
extern float rms_ffn_weight[];
extern float wq[];
extern float wk[];
extern float wv[];
extern float wo[];
extern float w1[];
extern float w2[];
extern float w3[];
extern float rms_final_weight[];
extern float *wcls;  // May point to token_embedding_table if shared

// ----------------------------------------------------------------------------
// Run state - flattened (no RunState struct)
// These are allocated dynamically
float *state_x;          // activation at current time stamp (dim,)
float *state_xb;         // same, but inside a residual branch (dim,)
float *state_xb2;        // an additional buffer just for convenience (dim,)
float *state_hb;         // buffer for hidden dimension in the ffn (hidden_dim,)
float *state_hb2;        // buffer for hidden dimension in the ffn (hidden_dim,)
float *state_q;          // query (dim,)
float *state_k;          // key (current pointer into key_cache)
float *state_v;          // value (current pointer into value_cache)
float *state_att;        // buffer for scores/attention values (n_heads, seq_len)
float *state_logits;     // output logits (vocab_size,)
float *state_key_cache;  // (layer, seq_len, kv_dim)
float *state_value_cache;// (layer, seq_len, kv_dim)

// ----------------------------------------------------------------------------
// Functions for state management

void init_run_state() {
    // Allocate buffers for the run state
    int kv_dim = (config_dim * config_n_kv_heads) / config_n_heads;

    state_x = calloc(config_dim, sizeof(float));
    state_xb = calloc(config_dim, sizeof(float));
    state_xb2 = calloc(config_dim, sizeof(float));
    state_hb = calloc(config_hidden_dim, sizeof(float));
    state_hb2 = calloc(config_hidden_dim, sizeof(float));
    state_q = calloc(config_dim, sizeof(float));
    state_key_cache = calloc(config_n_layers * config_seq_len * kv_dim, sizeof(float));
    state_value_cache = calloc(config_n_layers * config_seq_len * kv_dim, sizeof(float));
    state_att = calloc(config_n_heads * config_seq_len, sizeof(float));
    state_logits = calloc(config_vocab_size, sizeof(float));

    // Check all allocations succeeded
    if (!state_x || !state_xb || !state_xb2 || !state_hb || !state_hb2 || !state_q
     || !state_key_cache || !state_value_cache || !state_att || !state_logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state() {
    free(state_x);
    free(state_xb);
    free(state_xb2);
    free(state_hb);
    free(state_hb2);
    free(state_q);
    free(state_att);
    free(state_logits);
    free(state_key_cache);
    free(state_value_cache);
}

// ----------------------------------------------------------------------------
// Neural net blocks; the dynamics of the Transformer

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

float* forward(int token, int pos) {
    // a few convenience variables
    int dim = config_dim;
    int kv_dim = (config_dim * config_n_kv_heads) / config_n_heads;
    int kv_mul = config_n_heads / config_n_kv_heads;
    int hidden_dim = config_hidden_dim;
    int head_size = dim / config_n_heads;

    // copy the token embedding into x
    float* content_row = token_embedding_table + token * dim;
    memcpy(state_x, content_row, dim * sizeof(*state_x));

    // forward all the layers
    for(unsigned long long l = 0; l < config_n_layers; l++) {

        // attention rmsnorm
        rmsnorm(state_xb, state_x, rms_att_weight + l*dim, dim);

        // key and value point to the kv cache
        int loff = l * config_seq_len * kv_dim; // kv cache layer offset
        state_k = state_key_cache + loff + pos * kv_dim;
        state_v = state_value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
        matmul(state_q, state_xb, wq + l*dim*dim, dim, dim);
        matmul(state_k, state_xb, wk + l*dim*kv_dim, dim, kv_dim);
        matmul(state_v, state_xb, wv + l*dim*kv_dim, dim, kv_dim);

        // RoPE relative positional encoding
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? state_q : state_k;
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // multihead attention
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < config_n_heads; h++) {
            float* q = state_q + h * head_size;
            float* att = state_att + h * config_seq_len;

            for (int t = 0; t <= pos; t++) {
                float* k = state_key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }

            softmax(att, pos + 1);

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

        // final matmul to get the output of the attention
        matmul(state_xb2, state_xb, wo + l*dim*dim, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            state_x[i] += state_xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(state_xb, state_x, rms_ffn_weight + l*dim, dim);

        // FFN: self.w2(F.silu(self.w1(x)) * self.w3(x))
        matmul(state_hb, state_xb, w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(state_hb2, state_xb, w3 + l*dim*hidden_dim, dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = state_hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= state_hb2[i];
            state_hb[i] = val;
        }

        // final matmul to get the output of the ffn
        matmul(state_xb, state_hb, w2 + l*dim*hidden_dim, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            state_x[i] += state_xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(state_x, state_x, rms_final_weight, dim);

    // classifier into logits
    matmul(state_logits, state_x, wcls, config_dim, config_vocab_size);
    return state_logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

// Still need a minimal struct for TokenIndex (used internally by tokenizer)
typedef struct {
    char *str;
    int id;
} TokenIndex;

// Tokenizer state - flattened (no Tokenizer struct)
char** tokenizer_vocab;
float* tokenizer_vocab_scores;
TokenIndex *tokenizer_sorted_vocab;
int tokenizer_vocab_size;
unsigned int tokenizer_max_token_length;
unsigned char tokenizer_byte_pieces[512];

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(char* tokenizer_path, int vocab_size) {
    tokenizer_vocab_size = vocab_size;
    tokenizer_vocab = (char**)malloc(vocab_size * sizeof(char*));
    tokenizer_vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    tokenizer_sorted_vocab = NULL;

    for (int i = 0; i < 256; i++) {
        tokenizer_byte_pieces[i * 2] = (unsigned char)i;
        tokenizer_byte_pieces[i * 2 + 1] = '\0';
    }

    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&tokenizer_max_token_length, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "failed read\n");
        exit(EXIT_FAILURE);
    }

    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(tokenizer_vocab_scores + i, sizeof(float), 1, file) != 1) {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        if (fread(&len, sizeof(int), 1, file) != 1) {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        tokenizer_vocab[i] = (char *)malloc(len + 1);
        if (fread(tokenizer_vocab[i], len, 1, file) != 1) {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        tokenizer_vocab[i][len] = '\0';
    }
    fclose(file);
}

void free_tokenizer() {
    for (int i = 0; i < tokenizer_vocab_size; i++) {
        free(tokenizer_vocab[i]);
    }
    free(tokenizer_vocab);
    free(tokenizer_vocab_scores);
    free(tokenizer_sorted_vocab);
}

char* decode(int prev_token, int token) {
    char *piece = tokenizer_vocab[token];
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)tokenizer_byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return;
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok = { .str = str };
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (tokenizer_sorted_vocab == NULL) {
        tokenizer_sorted_vocab = malloc(tokenizer_vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < tokenizer_vocab_size; i++) {
            tokenizer_sorted_vocab[i].str = tokenizer_vocab[i];
            tokenizer_sorted_vocab[i].id = i;
        }
        qsort(tokenizer_sorted_vocab, tokenizer_vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    char* str_buffer = malloc((tokenizer_max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;
    *n_tokens = 0;

    if (bos) tokens[(*n_tokens)++] = 1;

    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", tokenizer_sorted_vocab, tokenizer_vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    for (char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) {
            str_len = 0;
        }

        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';

        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        int id = str_lookup(str_buffer, tokenizer_sorted_vocab, tokenizer_vocab_size);

        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }

    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            sprintf(str_buffer, "%s%s", tokenizer_vocab[tokens[i]], tokenizer_vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, tokenizer_sorted_vocab, tokenizer_vocab_size);
            if (id != -1 && tokenizer_vocab_scores[id] > best_score) {
                best_score = tokenizer_vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break;
        }

        tokens[best_idx] = best_id;
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--;
    }

    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler

typedef struct {
    float prob;
    int index;
} ProbIndex;

// Sampler state - flattened (no Sampler struct)
int sampler_vocab_size;
ProbIndex* sampler_probindex;
float sampler_temperature;
float sampler_topp;
unsigned long long sampler_rng_state;

int sample_argmax(float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1;
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;
        }
    }

    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index;
}

void build_sampler(int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler_vocab_size = vocab_size;
    sampler_temperature = temperature;
    sampler_topp = topp;
    sampler_rng_state = rng_seed;
    sampler_probindex = malloc(sampler_vocab_size * sizeof(ProbIndex));
}

void free_sampler() {
    free(sampler_probindex);
}

unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(float* logits) {
    int next;
    if (sampler_temperature == 0.0f) {
        next = sample_argmax(logits, sampler_vocab_size);
    } else {
        for (int q=0; q<sampler_vocab_size; q++) {
            logits[q] /= sampler_temperature;
        }
        softmax(logits, sampler_vocab_size);
        float coin = random_f32(&sampler_rng_state);
        if (sampler_topp <= 0 || sampler_topp >= 1) {
            next = sample_mult(logits, sampler_vocab_size, coin);
        } else {
            next = sample_topp(logits, sampler_vocab_size, sampler_topp, sampler_probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int));
    encode(prompt, 1, 0, prompt_tokens, &num_prompt_tokens);

    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;

    while (pos < steps) {
        float* logits = forward(token, pos);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(logits);
        }
        pos++;

        if (next == 1) { break; }

        char* piece = decode(token, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;

        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0';
        }
    }
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   run_flat [options]\n");
    fprintf(stderr, "Example: run_flat -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    // default parameters
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;
    float topp = 0.9f;
    int steps = 256;
    char *prompt = NULL;
    unsigned long long rng_seed = 0;

    // parse command line arguments
    for (int i = 1; i < argc; i+=2) {
        if (i + 1 >= argc) { error_usage(); }
        if (argv[i][0] != '-') { error_usage(); }
        if (strlen(argv[i]) != 2) { error_usage(); }

        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else { error_usage(); }
    }

    // parameter validation
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // Initialize model (weights are compiled in from model_data.c)
    init_run_state();
    if (steps == 0 || steps > config_seq_len) steps = config_seq_len;

    // Build tokenizer
    build_tokenizer(tokenizer_path, config_vocab_size);

    // Build sampler
    build_sampler(config_vocab_size, temperature, topp, rng_seed);

    // Generate!
    generate(prompt, steps);

    // Cleanup
    free_sampler();
    free_tokenizer();
    free_run_state();

    return 0;
}
#endif
