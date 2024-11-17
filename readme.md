Investigation into optimizing matrix multiplication
- (because https://github.com/OpenMathLib/OpenBLAS/tree/develop/kernel/x86_64 is really hard to read)

Benchmarks:

C basic matmul: ~1.4 GFLOP/s
- With -O2 optimization: ~5.6 GFLOP/s

C matmul with basic tiling: ~1.4 GFLOP/s
- I thought efficient caching would have more of an effect but I guess not

Openblas 1-thread (numpy): ~160 GFLOP/s
Openblas: 1000+ GFLOP/s
    - performance is sometimes inconsistent and dependent on size of matrix, I did not bother trying to adjust power management settings to get consistent results
    - will do more benchmarking if needed



System information:
```
            .-/+oossssoo+/-.               aaron@ubuntu
        `:+ssssssssssssssssss+:`           ---------------------- 
      -+ssssssssssssssssssyyssss+-         OS: Ubuntu 22.04.5 LTS x86_64 
    .ossssssssssssssssssdMMMNysssso.       Host: Z690 AERO G -CF 
   /ssssssssssshdmmNNmmyNMMMMhssssss/      Kernel: 6.8.0-48-generic 
  +ssssssssshmydMMMMMMMNddddyssssssss+     Uptime: 3 days, 2 hours, 45 mins 
 /sssssssshNMMMyhhyyyyhmNMMMNhssssssss/    Packages: 1863 (dpkg), 14 (snap) 
.ssssssssdMMMNhsssssssssshNMMMdssssssss.   Shell: bash 5.1.16 
+sssshhhyNMMNyssssssssssssyNMMMysssssss+   Resolution: 3440x1440 
ossyNMMMNyMMhsssssssssssssshmmmhssssssso   Terminal: node 
ossyNMMMNyMMhsssssssssssssshmmmhssssssso   CPU: 13th Gen Intel i9-13900K (32) @ 5.500GHz 
+sssshhhyNMMNyssssssssssssyNMMMysssssss+   GPU: Intel Raptor Lake-S GT1 [UHD Graphics 770] 
.ssssssssdMMMNhsssssssssshNMMMdssssssss.   GPU: NVIDIA GeForce RTX 4090 
 /sssssssshNMMMyhhyyyyhdNMMMNhssssssss/    Memory: 9447MiB / 64048MiB 
  +sssssssssdmydMMMMMMMMddddyssssssss+
   /ssssssssssshdmNNNNmyNMMMMhssssss/                              
    .ossssssssssssssssssdMMMNysssso.                               
      -+sssssssssssssssssyyyssss+-
        `:+ssssssssssssssssss+:`
            .-/+oossssoo+/-.
```
