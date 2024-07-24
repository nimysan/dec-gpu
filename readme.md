# Run

## 编译和Run

```bash
GENCODE_FLAGS=--gpu-architecture=compute_75
INCLUDES=-I/home/ubuntu/cuda-samples/Common -I/home/ubuntu/cuda-samples/Common
```

## Benchmark


```bash
# macos MacBook Pro M1--
Time elapsed: 1.00 seconds, Count: 780259
Time elapsed: 2.00 seconds, Count: 1594009
Time elapsed: 3.00 seconds, Count: 2398369
Time elapsed: 4.00 seconds, Count: 3212647
Time elapsed: 5.00 seconds, Count: 4015863
Time elapsed: 6.00 seconds, Count: 4823695
Time elapsed: 7.00 seconds, Count: 5630119
Time elapsed: 8.00 seconds, Count: 6439956

# g4dn.2xlage--
ubuntu@ip-172-31-79-138:~/dec-gpu$ ./sample 
Time elapsed: 1.00 seconds, Count: 272937
Time elapsed: 2.00 seconds, Count: 543818
Time elapsed: 3.00 seconds, Count: 812945
Time elapsed: 4.00 seconds, Count: 1082146
Time elapsed: 5.00 seconds, Count: 1350004
Time elapsed: 6.00 seconds, Count: 1618014
Time elapsed: 7.00 seconds, Count: 1885922
Time elapsed: 8.00 seconds, Count: 2153713
Time elapsed: 9.00 seconds, Count: 2420418
Time elapsed: 10.00 seconds, Count: 2687170

```

## GPU机器信息

```bash
ssh -i us-east-1.pem ubuntu@3.230.162.219
```


## --- benchmark ##

```bash

INCLUDES=-I/home/ubuntu/cuda-samples/Common -I/home/ubuntu/cuda-samples/Common


./gpu-des -k 0x703b021c0e251608 -m 0x123456789ab00000 -g -t 128 -T 8 -p -b 0x703b021c0e251608


./gpu-des -k 0x703b021c0e251608 -m 0x123456789ab00000 -c -t 128 -T 8 -p -b 0x703b021c0e200608 -l 0x703b021c0e250608


./gpu-des -k 0x703b021c0e251608 -m 0x123456789ab00000 -g -t 128 -T 8 -p -b 0x703b021c0e200608 -l 0x703b021c0e250608

G4的最好表现

ubuntu@ip-172-31-79-138:~/gpu-des$ ./gpu-des -k 0x703b021c0e251608 -m 0x123456789ab00000 -g -t 256 -T 1024 -p -b 0x703b021a0c100608 -l 0x703b021c0e250608
Key:            0x85c220b0a48a0e02
Message:        0x123456789ab00000
Encrypted:      0xe2773015a2453950
Decrypted:      0x123456789ab00000
Cracked (GPU):           NOT FOUND      8624865280 keys in   16.30s (throughput: 529.25M/s)

G5测试


ubuntu@ip-172-31-19-130:~/gpu-des$ ./gpu-des -k 0x703b021c0e251608 -m 0x123456789ab00000 -g -t 256 -T 512 -p -b 0x703b021c0e200608 -l 0x703b021c0e250608
Key:            0x85c220b0a48a0e02
Message:        0x123456789ab00000
Encrypted:      0xe2773015a2453950
Decrypted:      0x123456789ab00000
Cracked (GPU):           NOT FOUND          327680 keys in    0.33ms (throughput: 978.59M/s)

```
ubuntu@ip-172-31-19-130:~/gpu-des$ ./gpu-des -k 0x703b021c0e251608 -m 0x123456789ab00000 -g -t 256 -T 1024 -p -b 0x703b021a0c100608 -l 0x703b021c0e250608
Key:            0x85c220b0a48a0e02
Message:        0x123456789ab00000
Encrypted:      0xe2773015a2453950
Decrypted:      0x123456789ab00000


Cracked (GPU):           NOT FOUND      8624865280 keys in 6831.70ms (throughput: 1262.48M/s)


./gpu-des -k 0x703b021c0e251608 -m 0x123456789ab00000 -g -t 256 -T 1024 -p -b 0x703b011a0c100608 -l 0x703b021c0e250608

G5G的表现 t4g的卡

ubuntu@ip-172-31-39-8:~/gpu-des$ ./gpu-des -k 0x703b021c0e251608 -m 0x123456789ab00000 -g -t 256 -T 256 -p -b 0x703b021c0e200608 -l 0x703b021c0e250608
Key:            0x85c220b0a48a0e02
Message:        0x123456789ab00000
Encrypted:      0xe2773015a2453950
Decrypted:      0x123456789ab00000
Cracked (GPU):           NOT FOUND          327680 keys in    0.66ms (throughput: 499.73M/s)
ubuntu@ip-172-31-39-8:~/gpu-des$ 

ubuntu@ip-172-31-39-8:~/gpu-des$ ./gpu-des -k 0x703b021c0e251608 -m 0x123456789ab00000 -c -t 16 -T 1024 -p -b 0x703b021c0e200608 -l 0x703b021c0e250608
Key:            0x85c220b0a48a0e02
Message:        0x123456789ab00000
Encrypted:      0xe2773015a2453950
Decrypted:      0x123456789ab00000
Cracked (CPU):           NOT FOUND          327680 keys in 1001.43ms (throughput: 0.33M/s)


./gpu-des -k 0x703b021c0e251608 -m 0x123456789ab00000 -c -t 256 -T 1024 -p -b 0x703b021c0e200608 -l 0x703b021c0e250608

## G5.12xlarge 
./gpu-des -k 0x703b021c0e251608 -m 0x123456789ab00000 -M -t 256 -T 256 -p -b 0x703b021c0e200608 -l 0x703b021c0e250608 