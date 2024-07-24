#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// DES加密函数声明
void des_encrypt(unsigned char *plaintext, unsigned char *key, unsigned char *ciphertext);

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

// 置换函数
void permute(char *data, const char *table, int length) {
    char result[length];
    for (int i = 0; i < length; i++) {
        result[i] = data[table[i] - 1];
    }
    memcpy(data, result, length);
}

void des_encrypt(unsigned char *plaintext, unsigned char *key, unsigned char *ciphertext) {
    // 将密钥和明文转换为二进制字符串
    char key_binary[64 + 1];
    char plaintext_binary[64 + 1];
    for (int i = 0; i < 8; i++) {
        sprintf(key_binary + i * 8, "%08b", key[i]);
        sprintf(plaintext_binary + i * 8, "%08b", plaintext[i]);
    }
    key_binary[64] = '\0';
    plaintext_binary[64] = '\0';

    // 对密钥进行置换
    permute(key_binary, key_permutation, 64);

    // 对明文进行初始置换
    permute(plaintext_binary, initial_permutation, 64);

    // 执行 DES 加密算法的其余部分...

    // 将加密后的二进制字符串复制到密文中
    for (int i = 0; i < 8; i++) {
        ciphertext[i] = (unsigned char)strtoul(plaintext_binary + i * 8, NULL, 2);
    }
}



// // DES加密函数(实际需要实现DES算法)
// void des_encrypt(unsigned char *plaintext, unsigned char *key, unsigned char *ciphertext) {
//     // 实现DES加密算法
// }

int main() {
    unsigned char plaintext[8] = {0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22};
    unsigned char ciphertext[8] = {0x94, 0xA3, 0x3C, 0x50, 0x39, 0x2D, 0x45, 0xE3};
    unsigned char key[8];
    unsigned char test_ciphertext[8];

    clock_t start_time = clock();
    clock_t last_time = start_time;
    long long count = 0;

    int found = 0;
    for (int i = 0; i < 0x100000000; i++) { // 遍历所有可能的密钥
        for (int j = 0; j < 8; j++) {
            key[j] = (i >> (j * 8)) & 0xFF; // 生成密钥
        }

        des_encrypt(plaintext, key, test_ciphertext); // 使用当前密钥加密明文
        count++;

        if (memcmp(test_ciphertext, ciphertext, 8) == 0) { // 检查是否找到正确的密钥
            found = 1;
            break;
        }

        clock_t current_time = clock();
        if (current_time - last_time >= CLOCKS_PER_SEC) {
            double elapsed_time = (double)(current_time - start_time) / CLOCKS_PER_SEC;
            printf("Time elapsed: %.2f seconds, Count: %lld\n", elapsed_time, count);
            last_time = current_time;
        }
    }

    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

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