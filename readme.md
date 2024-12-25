Got access to H100s (hopper) and apple silicon so I'm working on gemm on those platforms!

Investigation into optimizing matrix multiplication
- ( https://github.com/OpenMathLib/OpenBLAS/tree/develop/kernel/x86_64 is really hard to read)

Benchmarks:
- gemm_basic : basic implementation in c
  - around 1 gflops
- gemm_transpose : transpose matrix B
  - reduce cache misses by traversing B through row-majored fashion
  - around 6 gflops
- gemm_tiling
  - reduce cache misses even further through blocked tiling
  - maxes out at around 36 gflops, good block sizes matter a lot here
  - the moment you start using ram the gflops plummet, you want everything to fit on registers
- gemm_simd
  - uses avx SIMD instructions to parallize the computation
  - note that gemm_tiling and gemm_transpose already have some SIMD instructions on data moving from compiler optimization
  - around 34 flops
- gemm_tiled_simd
  - maxes out at around 110 gflops
  - can improve upon cache coherency, but i'm happy with performance
- openblas
  - 160 gflops on one thread on my cpu
  - 1000 gflops on multithreaded

Notes
- use ```objdump -d a.out``` to look at generated assembly
- make sure to set OPENBLAS_NUM_THREADS=1 so that numpy doesn't use all the threads
- valgrind can also be used to check cache hits
  - ```valgrind --tool=cachegrind ./a.out```

System information:
avx
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

cuda machine
```
            .-/+oossssoo+/-.               root@pod-as-vm 
        `:+ssssssssssssssssss+:`           -------------- 
      -+ssssssssssssssssssyyssss+-         OS: Ubuntu 22.04.4 LTS x86_64 
    .ossssssssssssssssssdMMMNysssso.       Host: PowerEdge XE9680 
   /ssssssssssshdmmNNmmyNMMMMhssssss/      Kernel: 5.15.0-124-generic 
  +ssssssssshmydMMMMMMMNddddyssssssss+     Uptime: 50 days, 1 hour, 46 mins 
 /sssssssshNMMMyhhyyyyhmNMMMNhssssssss/    Packages: 504 (dpkg) 
.ssssssssdMMMNhsssssssssshNMMMdssssssss.   Shell: bash 5.1.16 
+sssshhhyNMMNyssssssssssssyNMMMysssssss+   Resolution: 1024x768 
ossyNMMMNyMMhsssssssssssssshmmmhssssssso   Terminal: vscode 
ossyNMMMNyMMhsssssssssssssshmmmhssssssso   CPU: Intel Xeon Platinum 8470 (208) @ 2.972GHz 
+sssshhhyNMMNyssssssssssssyNMMMysssssss+   GPU: NVIDIA H100 SXM5 80GB 
.ssssssssdMMMNhsssssssssshNMMMdssssssss.   GPU: NVIDIA H100 SXM5 80GB 
 /sssssssshNMMMyhhyyyyhdNMMMNhssssssss/    GPU: NVIDIA H100 SXM5 80GB 
  +sssssssssdmydMMMMMMMMddddyssssssss+     GPU: NVIDIA H100 SXM5 80GB 
   /ssssssssssshdmNNNNmyNMMMMhssssss/      GPU: NVIDIA H100 SXM5 80GB 
    .ossssssssssssssssssdMMMNysssso.       GPU: NVIDIA H100 SXM5 80GB 
      -+sssssssssssssssssyyyssss+-         GPU: NVIDIA H100 SXM5 80GB 
        `:+ssssssssssssssssss+:`           GPU: NVIDIA H100 SXM5 80GB 
            .-/+oossssoo+/-.               Memory: 40598MiB / 1031524MiB 
```
