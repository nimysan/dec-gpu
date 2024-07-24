#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// DES加密函数声明
void des_encrypt(unsigned char *plaintext, unsigned char *key, unsigned char *ciphertext);

// DES加密函数(实际需要实现DES算法)
void des_encrypt(unsigned char *plaintext, unsigned char *key, unsigned char *ciphertext) {
    // 实现DES加密算法
}

int main() {
    unsigned char plaintext[8] = {0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22};
    unsigned char ciphertext[8] = {0x94, 0xA3, 0x3C, 0x50, 0x39, 0x2D, 0x45, 0xE3};
    unsigned char key[8];
    unsigned char test_ciphertext[8];

    clock_t start_time = clock();

    int found = 0;
    for (int i = 0; i < 0x100000000; i++) { // 遍历所有可能的密钥
        for (int j = 0; j < 8; j++) {
            key[j] = (i >> (j * 8)) & 0xFF; // 生成密钥
        }

        des_encrypt(plaintext, key, test_ciphertext); // 使用当前密钥加密明文

        if (memcmp(test_ciphertext, ciphertext, 8) == 0) { // 检查是否找到正确的密钥
            found = 1;
            break;
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