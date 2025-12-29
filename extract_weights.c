/* Tool to extract weights from binary file and generate C arrays */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Config structure from run.c
typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

void write_float_array(FILE *out, const char *name, float *data, long long size) {
    fprintf(out, "float %s[%lld] = {\n", name, size);

    for (long long i = 0; i < size; i++) {
        if (i % 8 == 0) fprintf(out, "    ");
        fprintf(out, "%.9ef", data[i]);
        if (i < size - 1) fprintf(out, ",");
        if (i % 8 == 7 || i == size - 1) fprintf(out, "\n");
        else fprintf(out, " ");
    }

    fprintf(out, "};\n\n");
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <checkpoint.bin> <output.c>\n", argv[0]);
        return 1;
    }

    char *checkpoint_path = argv[1];
    char *output_path = argv[2];

    // Open checkpoint file
    FILE *model_file = fopen(checkpoint_path, "rb");
    if (!model_file) {
        fprintf(stderr, "Error: Could not open %s\n", checkpoint_path);
        return 1;
    }

    // Read config
    Config config;
    if (fread(&config, sizeof(Config), 1, model_file) != 1) {
        fprintf(stderr, "Error: Could not read config\n");
        fclose(model_file);
        return 1;
    }

    // Handle shared weights
    int shared_weights = config.vocab_size > 0 ? 1 : 0;
    config.vocab_size = abs(config.vocab_size);

    printf("Model configuration:\n");
    printf("  dim: %d\n", config.dim);
    printf("  hidden_dim: %d\n", config.hidden_dim);
    printf("  n_layers: %d\n", config.n_layers);
    printf("  n_heads: %d\n", config.n_heads);
    printf("  n_kv_heads: %d\n", config.n_kv_heads);
    printf("  vocab_size: %d\n", config.vocab_size);
    printf("  seq_len: %d\n", config.seq_len);
    printf("  shared_weights: %d\n\n", shared_weights);

    // Calculate sizes
    int head_size = config.dim / config.n_heads;
    long long n_layers = config.n_layers;

    long long token_embedding_size = (long long)config.vocab_size * config.dim;
    long long rms_att_weight_size = n_layers * config.dim;
    long long wq_size = n_layers * config.dim * (config.n_heads * head_size);
    long long wk_size = n_layers * config.dim * (config.n_kv_heads * head_size);
    long long wv_size = n_layers * config.dim * (config.n_kv_heads * head_size);
    long long wo_size = n_layers * (config.n_heads * head_size) * config.dim;
    long long rms_ffn_weight_size = n_layers * config.dim;
    long long w1_size = n_layers * config.dim * config.hidden_dim;
    long long w2_size = n_layers * config.hidden_dim * config.dim;
    long long w3_size = n_layers * config.dim * config.hidden_dim;
    long long rms_final_weight_size = config.dim;
    long long freq_size = config.seq_len * head_size / 2;
    long long wcls_size = shared_weights ? 0 : (long long)config.vocab_size * config.dim;

    // Read all weights into memory
    fseek(model_file, 0, SEEK_END);
    long file_size = ftell(model_file);
    fseek(model_file, sizeof(Config), SEEK_SET);

    long weights_size = (file_size - sizeof(Config)) / sizeof(float);
    float *weights = (float*)malloc(weights_size * sizeof(float));

    if (fread(weights, sizeof(float), weights_size, model_file) != weights_size) {
        fprintf(stderr, "Error: Could not read all weights\n");
        free(weights);
        fclose(model_file);
        return 1;
    }
    fclose(model_file);

    printf("Total weights read: %ld floats (%.2f MB)\n\n", weights_size,
           (float)(weights_size * sizeof(float)) / (1024*1024));

    // Open output file
    FILE *out = fopen(output_path, "w");
    if (!out) {
        fprintf(stderr, "Error: Could not open output file %s\n", output_path);
        free(weights);
        return 1;
    }

    // Write header
    fprintf(out, "/* Auto-generated model weights from %s */\n\n", checkpoint_path);
    fprintf(out, "#include <stddef.h>\n\n");

    // Write config
    fprintf(out, "/* Model Configuration */\n");
    fprintf(out, "int config_dim = %d;\n", config.dim);
    fprintf(out, "int config_hidden_dim = %d;\n", config.hidden_dim);
    fprintf(out, "int config_n_layers = %d;\n", config.n_layers);
    fprintf(out, "int config_n_heads = %d;\n", config.n_heads);
    fprintf(out, "int config_n_kv_heads = %d;\n", config.n_kv_heads);
    fprintf(out, "int config_vocab_size = %d;\n", config.vocab_size);
    fprintf(out, "int config_seq_len = %d;\n", config.seq_len);
    fprintf(out, "int config_shared_weights = %d;\n\n", shared_weights);

    // Write weights
    printf("Writing token embeddings...\n");
    float *ptr = weights;
    write_float_array(out, "token_embedding_table", ptr, token_embedding_size);
    ptr += token_embedding_size;

    printf("Writing attention RMS weights...\n");
    write_float_array(out, "rms_att_weight", ptr, rms_att_weight_size);
    ptr += rms_att_weight_size;

    printf("Writing query weights...\n");
    write_float_array(out, "wq", ptr, wq_size);
    ptr += wq_size;

    printf("Writing key weights...\n");
    write_float_array(out, "wk", ptr, wk_size);
    ptr += wk_size;

    printf("Writing value weights...\n");
    write_float_array(out, "wv", ptr, wv_size);
    ptr += wv_size;

    printf("Writing output weights...\n");
    write_float_array(out, "wo", ptr, wo_size);
    ptr += wo_size;

    printf("Writing FFN RMS weights...\n");
    write_float_array(out, "rms_ffn_weight", ptr, rms_ffn_weight_size);
    ptr += rms_ffn_weight_size;

    printf("Writing w1 weights...\n");
    write_float_array(out, "w1", ptr, w1_size);
    ptr += w1_size;

    printf("Writing w2 weights...\n");
    write_float_array(out, "w2", ptr, w2_size);
    ptr += w2_size;

    printf("Writing w3 weights...\n");
    write_float_array(out, "w3", ptr, w3_size);
    ptr += w3_size;

    printf("Writing final RMS weights...\n");
    write_float_array(out, "rms_final_weight", ptr, rms_final_weight_size);
    ptr += rms_final_weight_size;

    // Skip freq_cis_real and freq_cis_imag (RoPE frequencies)
    ptr += freq_size * 2;

    if (!shared_weights) {
        printf("Writing classifier weights...\n");
        write_float_array(out, "wcls", ptr, wcls_size);
        ptr += wcls_size;
    } else {
        fprintf(out, "/* Classifier weights are shared with token embeddings */\n");
        fprintf(out, "float *wcls = token_embedding_table;\n\n");
    }

    fclose(out);
    free(weights);

    printf("\nDone! Generated %s\n", output_path);
    return 0;
}
