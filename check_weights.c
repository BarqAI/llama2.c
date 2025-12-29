/* Check Layer 1 weights */
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

int main() {
    Config config;
    int fd = open("stories15M.bin", O_RDONLY);
    if (fd == -1) { perror("open"); return 1; }

    // Get file size
    off_t file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);

    // Memory map
    float* data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) { perror("mmap"); return 1; }

    // Read config
    config = *(Config*)data;
    int shared_weights = config.vocab_size > 0 ? 1 : 0;
    config.vocab_size = abs(config.vocab_size);

    printf("Config: dim=%d, n_layers=%d, n_heads=%d\n",
           config.dim, config.n_layers, config.n_heads);

    // Calculate weight offsets
    float* ptr = data + sizeof(Config)/sizeof(float);
    float* token_embedding_table = ptr;
    ptr += config.vocab_size * config.dim;
    float* rms_att_weight = ptr;  // (layer, dim)

    printf("\n=== Layer 0 rms_att_weight (first 5) ===\n");
    for (int i = 0; i < 5; i++) {
        printf("  rms_att_weight[%d] = %.8f\n", i, rms_att_weight[i]);
    }

    printf("\n=== Layer 1 rms_att_weight (first 5) ===\n");
    int layer1_offset = 1 * config.dim;  // layer 1 starts at offset dim
    for (int i = 0; i < 5; i++) {
        printf("  rms_att_weight[%d] = %.8f\n", layer1_offset + i, rms_att_weight[layer1_offset + i]);
    }

    munmap(data, file_size);
    close(fd);
    return 0;
}
