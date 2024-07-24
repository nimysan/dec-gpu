#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// CUDA 核函数声明
__global__ void des_encrypt_kernel(unsigned char *plaintext, unsigned char *ciphertext, unsigned char *keys, int num_keys);

// 密钥排列表
static const char key_permutation[] = {
    57, 49, 41, 33, 25, 17, 9, 1,
    59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13, 5,
    63, 55, 47, 39, 31, 23, 15, 7,
    56, 48, 40, 32, 24, 16, 8, 0
};

// 初始置换表
static const char initial_permutation[] = {
    58, 50, 42, 34, 26, 18, 10, 2,
    60, 52, 44, 36, 28, 20, 12, 4,
    62, 54, 46, 38, 30, 22, 14, 6,
    64, 56, 48, 40, 32, 24, 16, 8,
    57, 49, 41, 33, 25, 17, 9, 1,
    59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13, 5,
    63, 55, 47, 39, 31, 23, 15, 7
};

__constant__ char d_key_permutation[64];
__constant__ char d_initial_permutation[64];

// 置换函数
__device__ void permute(char *data, const char *table, int length) {
    char result[64];
    for (int i = 0; i < length; i++) {
        result[i] = data[table[i] - 1];
    }
    memcpy(data, result, length);
}

__device__ unsigned int str2int(const char *str, int base) {
    unsigned int result = 0;
    int i = 0;
    while (str[i] != '\0') {
        unsigned int digit;
        if (str[i] >= '0' && str[i] <= '9') {
            digit = str[i] - '0';
        } else if (str[i] >= 'A' && str[i] <= 'F') {
            digit = str[i] - 'A' + 10;
        } else if (str[i] >= 'a' && str[i] <= 'f') {
            digit = str[i] - 'a' + 10;
        } else {
            break;
        }
        result = result * base + digit;
        i++;
    }
    return result;
}

// DES 加密核函数
__global__ void des_encrypt_kernel(unsigned char *plaintext, unsigned char *ciphertext, unsigned char *keys, int num_keys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_keys) {
        unsigned char key[8];
        for (int i = 0; i < 8; i++) {
            key[i] = keys[idx * 8 + i];
        }

        char key_binary[64 + 1];
        char plaintext_binary[64 + 1];
        for (int i = 0; i < 8; i++) {
            // sprintf(key_binary + i * 8, "%08b", key[i]);
            // sprintf(plaintext_binary + i * 8, "%08b", plaintext[i]);
        }
        key_binary[64] = '\0';
        plaintext_binary[64] = '\0';

        // permute(key_binary, key_permutation, 64);
        // permute(plaintext_binary, initial_permutation, 64);
        permute(key_binary, d_key_permutation, 64);
        permute(plaintext_binary, d_initial_permutation, 64);

        // 执行 DES 加密算法的其余部分...

        for (int i = 0; i < 8; i++) {
            ciphertext[idx * 8 + i] = (unsigned char)str2int(plaintext_binary + i * 8, 2);
        }
    }
}

int main() {
    unsigned char plaintext[8] = {0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22};
    unsigned char ciphertext[8] = {0x94, 0xA3, 0x3C, 0x50, 0x39, 0x2D, 0x45, 0xE3};
    unsigned char *keys, *test_ciphertexts;
    unsigned char key[8];

    int num_keys = 0x100000000; // 最多测试 2^32 个密钥
    int found = 0;

    cudaMemcpyToSymbol(d_key_permutation, key_permutation, sizeof(key_permutation));
    cudaMemcpyToSymbol(d_initial_permutation, initial_permutation, sizeof(initial_permutation));


    // 分配 GPU 内存
    cudaMalloc(&keys, num_keys * 8 * sizeof(unsigned char));
    cudaMalloc(&test_ciphertexts, num_keys * 8 * sizeof(unsigned char));

    // 生成密钥并复制到 GPU 内存
    for (int i = 0; i < num_keys; i++) {
        for (int j = 0; j < 8; j++) {
            key[j] = (i >> (j * 8)) & 0xFF;
        }
        cudaMemcpy(keys + i * 8, key, 8 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    }

    clock_t start_time = clock();

    // 启动 GPU 核函数
    int threads_per_block = 256;
    int num_blocks = (num_keys + threads_per_block - 1) / threads_per_block;
    des_encrypt_kernel<<<num_blocks, threads_per_block>>>(plaintext, test_ciphertexts, keys, num_keys);

    // 将结果从 GPU 内存复制到 CPU 内存
    unsigned char *host_ciphertexts = (unsigned char *)malloc(num_keys * 8 * sizeof(unsigned char));
    cudaMemcpy(host_ciphertexts, test_ciphertexts, num_keys * 8 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    // 在 CPU 上搜索正确的密钥
    for (int i = 0; i < num_keys; i++) {
        if (memcmp(host_ciphertexts + i * 8, ciphertext, 8) == 0) {
            found = 1;
            memcpy(key, keys + i * 8, 8 * sizeof(unsigned char));
            break;
        }
    }

    // 释放 GPU 内存
    cudaFree(keys);
    cudaFree(test_ciphertexts);

          // 打印 host_ciphertexts 数组
    printf("Ciphertexts:\n");
    for (int i = 0; i < num_keys; i++) {
        printf("%02X %02X %02X %02X %02X %02X %02X %02X\n",
               host_ciphertexts[i * 8 + 0], host_ciphertexts[i * 8 + 1],
               host_ciphertexts[i * 8 + 2], host_ciphertexts[i * 8 + 3],
               host_ciphertexts[i * 8 + 4], host_ciphertexts[i * 8 + 5],
               host_ciphertexts[i * 8 + 6], host_ciphertexts[i * 8 + 7]);
    }

    free(host_ciphertexts);


    if (found) {
        printf("Found the key: ");
        for (int i = 0; i < 8; i++) {
            printf("%02X ", key[i]);
        }
        printf("\n");
        printf("Time taken: %f seconds\n", elapsed_time);
    } else {
        printf("Key not found.\n");
    }

    return 0;
}
