# Run

## 编译和Run

```bash
gcc sample sample.c
./sample
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