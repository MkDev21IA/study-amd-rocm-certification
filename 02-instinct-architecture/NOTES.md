# Course 2: Instinct Architecture — CDNA/RDNA & MI400 Roadmap

> **Course Focus:** Explore AMD Instinct GPU hardware, from compute units and matrix engines to the memory hierarchy. Learn how GPU kernels map from threads and wavefronts to SIMD hardware and how GPUs hide memory latency. Compare AMD CDNA and RDNA architectures and how HIP enables development across both.

---

## Quick Reference / Table of Contents
1. [Compute Units (CUs) & Hardware Execution Pipeline](#1-compute-units-cus--hardware-execution-pipeline)
2. [Mapping Software to Silicon: Threads, Wavefronts & SIMD](#2-mapping-software-to-silicon-threads-wavefronts--simd)
3. [Memory Hierarchy & Hardware Latency Hiding](#3-memory-hierarchy--hardware-latency-hiding)
4. [Matrix Engines & MFMA (Matrix Fused Multiply-Add)](#4-matrix-engines--mfma)
5. [Architectural Comparison: CDNA vs. RDNA](#5-architectural-comparison-cdna-vs-rdna)
6. [AMD Instinct Family Roadmap (MI100 $\rightarrow$ MI200 $\rightarrow$ MI300 $\rightarrow$ MI400)](#6-amd-instinct-family-roadmap)
7. [Study Notes, Doubts & Hardware Deep Dives](#7-study-notes-doubts--hardware-deep-dives)

---

## 1. Compute Units (CUs) & Hardware Execution Pipeline
* **CU Architecture:** The fundamental multi-threaded compute core of CDNA. Contains 4 SIMD-16 vector pipelines, Matrix Core engines (MFMA), 64 KB of LDS (Local Data Share), Vector Register File (VGPR), and Scalar Register File (SGPR).
* **Hardware Capacity:**
  * **SIMD Units per CU:** 4 SIMD-16 units.
  * **Wavefront Slots per SIMD:** 8 slots.
  * **Theoretical Max Wavefronts per CU:** **32 wavefronts** (standard CDNA/CDNA2, up to 40 in CDNA3).
  * **Max Concurrent Threads per CU:** $32 \times 64 = \mathbf{2,048\text{ threads in flight}}$.
  * **Max Blocks (Workgroups) per CU:** Up to 8 to 16 Workgroups concurrently.

## 2. Mapping Software to Silicon: Threads, Wavefronts & SIMD
* **Hierarchy Mapping:**
  * **Grid (Software):** Entire problem domain $\rightarrow$ Dispatched across all CUs and XCDs on the GPU.
  * **Block / Workgroup (Software):** Group of threads sharing LDS $\rightarrow$ **Mapped to exactly ONE CU** (never split across CUs).
  * **Wavefront (Hardware Quantum):** 64 threads (Wave64) $\rightarrow$ Scheduled onto a SIMD vector unit.
  * **Thread (Software):** Scalar worker $\rightarrow$ Mapped to 1 physical SIMD lane (ALU).
* **SPMD (Software Programming Model):** Programmer writes scalar thread code with ID indexing.
* **SIMD (Silicon Datapath):** Physical 16-lane vector ALU executing lockstep instructions.
* **SIMT (Execution Architecture):** Groups 64 threads into Wavefronts, using the 64-bit `EXEC` mask to serialize divergent `if/else` branches.

## 3. Memory Hierarchy & Hardware Latency Hiding
* **Zero-Overhead Wavefront Scheduling:** Memory access from HBM3 takes 200–400 cycles. To prevent ALUs from idling, the hardware scheduler switches between up to 32 active resident wavefronts with 0 clock cycles of overhead, executing math for other wavefronts while stalled ones wait for memory data.
* **Occupancy Limiters:** The ratio of active wavefronts vs. theoretical maximum (32) is constrained by:
  1. **VGPR Pressure:** High register allocation per thread reduces the number of resident wavefronts.
  2. **LDS Capacity:** 64 KB total per CU. If a block requests 32 KB, only 2 blocks can reside on the CU.
  3. **Workgroup limits:** Hardware ceiling of 8–16 workgroups per CU.

## 4. Matrix Engines & MFMA
* **ALU (Atomic Unit):** 1-lane combinational/pipelined circuit performing arithmetic/logic ($A+B, A \times B + C$).
* **SIMD (1D Vector Processing):** One instruction decoder commanding an array of 16 physical ALUs in lockstep. Used for 1D element-wise operations (activations, vector additions, normalizations).
* **MFMA (2D Matrix Engine):** Dedicated 2D systolic/MAC hardware array inside the Compute Unit. Executes high-density matrix tiles ($D = A \times B + C$) with local accumulation, eliminating register round-trips for partial sums. (AMD's counterpart to NVIDIA Tensor Cores).

## 5. Architectural Comparison: CDNA vs. RDNA
*(Notes will be added here as we discuss the video)*

## 6. AMD Instinct Family Roadmap
* **Instinct Purpose:** Enterprise and data-center GPUs/accelerators designed exclusively for AI and HPC. No display controllers or video outputs; silicon area is dedicated entirely to Compute Units, Matrix Engines, and interconnects.
* **CDNA 1 (MI100 - 2020):** First architecture split from graphics (RDNA). Monolithic 7nm die, 32 GB HBM2. Introduced MFMA (Matrix Fused Multiply-Add).
* **CDNA 2 (MI200 Series / MI250X - 2021):** First multi-chip module (MCM - 2 compute dies connected by Infinity Fabric). 128 GB HBM2e. Powers the Frontier Exascale supercomputer. Full-rate FP64 compute.
* **CDNA 3 (MI300 Series - 2023/2024):** 3D chiplet stacking using TSMC-SoIC (compute dies stacked on base I/O dies).
  * **MI300X:** Pure GPU accelerator with 192 GB HBM3 (5.3 TB/s bandwidth) aimed directly at large LLM training and inference.
  * **MI300A:** First Exascale APU combining 24 Zen 4 CPU cores and CDNA 3 GPU cores with a single unified physical HBM3 memory pool. Powers the El Capitan supercomputer.
* **CDNA 4 / UDNA (MI400 Series - Future):** Next-gen roadmap featuring native FP4/FP6 precisions, unified architecture convergence (UDNA), and HBM3e/HBM4.

---

## 7. Study Notes, Doubts & Hardware Deep Dives

### Q1: What is AMD Instinct? What is MI100 to MI400? Are they processors?
* **Hardware classification:** They are massively parallel vector/matrix processors (enterprise GPUs / APUs).
* **"MI" designation:** Stands for *Machine Intelligence*.
* **Why not regular GPUs?** Consumer GPUs (Radeon / RDNA) balance rasterization, ray tracing, and display engines for graphics. Instinct GPUs (CDNA) strip display pipelines entirely and replace them with Matrix Cores (MFMA hardware MAC arrays), wide HBM buses, and Infinity Fabric links for multi-GPU scaling.
* **EE Analogy:** A CPU is designed for low-latency serial logic (out-of-order execution, deep branch prediction). An Instinct GPU is an array of hundreds of hardware SIMD engines communicating over an ultra-wide (e.g. 8192-bit) memory bus to deliver thousands of TFLOPs of matrix arithmetic ($D = A \cdot B + C$).

### Q2: What are FP64, FP32, FP16, FP8, FP6, and FP4? Why do numbers keep decreasing?
* **Definition:** FP = *Floating Point*. The number denotes the **total bit-width** allocated in hardware: $\text{Total Bits} = \text{Sign (1 bit)} + \text{Exponent } (E \text{ bits}) + \text{Mantissa } (M \text{ bits})$.
* **Hardware Formats Breakdown:**
  * **FP64 (64-bit / 8 bytes):** Double precision ($1 + 11 + 52$). Used for scientific HPC (climate models, astrophysics) where rounding errors compounding over trillions of steps must be avoided.
  * **FP32 (32-bit / 4 bytes):** Single precision ($1 + 8 + 23$). Classic compute & graphics baseline.
  * **FP16 / BF16 (16-bit / 2 bytes):** Half precision & Brain Floating Point ($1 + 8 + 7$ for BF16). Mainstay of Deep Learning training.
  * **FP8 (8-bit / 1 byte):** Standardized as E4M3 (inference weights) and E5M2 (gradients). Halves memory footprint and doubles compute throughput on MI300X / CDNA 3.
  * **FP6 & FP4 (6-bit & 4-bit):** Sub-byte micro-scaling formats (e.g., OCP MX formats). An FP4 number takes only a 4-bit nibble—two weights per byte. Targeted in the MI400 / CDNA 4 roadmap.
* **EE & Hardware Rationale for Smaller Precisions:**
  1. **Multiplier Gate Complexity $O(N^2)$:** An 8-bit or 4-bit MAC circuit requires vastly fewer transistors than a 64-bit multiplier. On identical silicon die area, tens of times more FP8/FP4 MAC units can be integrated into CDNA Matrix Engines.
  2. **Memory Bandwidth & Power:** Moving bits across the HBM bus burns energy ($\text{pJ/bit}$). Halving bits halves memory traffic, allowing models to fit in on-chip SRAM/LDS and VRAM without memory stalling.
  3. **Noise Tolerance in AI:** Deep neural networks are statistical classifiers; they tolerate coarse precision without losing accuracy, unlike differential equation solvers in HPC.

### Q3: What is the difference between the 4 MB L2 cache in each XCD and the 256 MB Infinity Cache? Why is it called "Infinity"?
* **3D Physical Packaging (TSMC 3D SoIC):**
  * **Top Layer:** 8 Accelerator Complex Dies (XCDs) housing compute units.
  * **Bottom Layer:** 4 Base I/O Dies (IODs) stacked vertically under the XCDs via Through-Silicon Vias (TSVs).
* **4 MB L2 Cache (Private per XCD):**
  * Resides physically on each compute die (XCD).
  * Serves the 38 local CUs of that XCD with minimum latency and dedicated bandwidth ($8 \times 4\text{ MB} = 32\text{ MB}$ total L2 across the GPU).
* **256 MB Infinity Cache (Global / Shared):**
  * Resides on the 4 base IODs ($64\text{ MB} \times 4 = 256\text{ MB}$).
  * Acts as a global Last-Level Cache (LLC / L3) shared coherently across all 8 XCDs.
* **Why "Infinity"?**
  * Named after AMD's proprietary coherent interconnect: **AMD Infinity Fabric™**.
  * The cache is integrated directly onto the Infinity Fabric crossbar switches on the base dies, unifying the chiplets into a single virtual GPU.
* **EE Impact (Bandwidth Amplification & Energy):**
  * Accessing SRAM on the Infinity Cache costs $\sim 1\text{--}2\text{ pJ/bit}$, whereas driving external PHY lines to HBM3 costs $\sim 5\text{--}8\text{ pJ/bit}$.
  * Cache hits amplify effective memory bandwidth from $5.3\text{ TB/s}$ (HBM3 physical limit) to $>17\text{ TB/s}$ peak on-package throughput.

### Q4: What is the relation between ALU, SIMD, and MFMA?
* **Hardware Hierarchy:**
  * **ALU (Arithmetic Logic Unit):** The 1-lane atomic hardware datapath doing basic operations ($+ , \times, \text{AND}$).
  * **SIMD (Single Instruction, Multiple Data):** A 1D array of ALUs (in CDNA, 16 physical ALUs per SIMD unit) controlled by a single instruction decoder. Executes 1D vector instructions (e.g., `v_add_f32`) over 4 cycles for a 64-wide wavefront.
  * **MFMA (Matrix Fused Multiply-Add):** A dedicated 2D hardware matrix engine (grid of Multiply-Accumulate units) inside the Compute Unit. Instead of computing single vector lanes, it consumes whole matrix tiles to compute $D = A \times B + C$ in hardware with internal accumulation, saving instruction overhead and register traffic.
* **Workload Division:**
  * SIMD executes pointwise vector ops (ReLU, GeLU, LayerNorm, vector additions).
  * MFMA executes dense GEMM / matrix multiplications (Dense layers, Attention projections, Convolutions).
* **NVIDIA Equivalent:** SIMD $\approx$ CUDA Cores; MFMA $\approx$ Tensor Cores.

### Q5: What is a MAC (Multiply-Accumulate) unit?
* **Definition:** A fundamental digital arithmetic circuit that calculates $\text{Accumulator} \leftarrow \text{Accumulator} + (A \times B)$ or $\text{Result} = (A \times B) + C$ in a single pipelined datapath.
* **Why it dominates AI & DSP:** Over 99% of deep learning arithmetic consists of dot products ($\sum x_i w_i$), convolutions, and matrix multiplications ($C_{ij} = \sum A_{ik} B_{kj}$). All of these are repeated multiply-adds.
* **FMA (Fused Multiply-Add):** The IEEE-standardized implementation of a MAC where the multiplication and addition are performed in one step with **only a single rounding operation at the end**, preserving numerical precision.
* **Connection to MFMA:** AMD's **MFMA** (Matrix Fused Multiply-Add) is an arrayed grid of these FMA circuits on silicon executing matrix dot products concurrently.

### Q6: What is the difference between SPMD, SIMD, and SIMT?
* **Stack Distinction:**
  * **SPMD (Software / Programmer Model):** Write single-thread scalar code that operates concurrently over multiple data points using thread indexing (`c[idx] = a[idx] + b[idx]`).
  * **SIMD (Silicon Datapath):** A physical vector processing unit with one instruction decoder controlling a bank of parallel ALUs in lockstep.
  * **SIMT (Execution Architecture Bridge):** Hardware groups SPMD threads into Wavefronts (64 threads in CDNA) and executes them across SIMD ALUs.
* **EE Hardware Mechanism for Divergence:** In CDNA, when a wavefront branches (`if/else`), threads cannot run separate instructions simultaneously. The hardware utilizes the 64-bit `EXEC` mask register to bit-enable active lanes during the `if` block, then bit-inverts `EXEC` to execute the `else` block while the former lanes are clock-gated. Divergence results in time serialization.

### Q7: How do Grid, Block, and Wavefront relate, and how many Wavefronts can live on a single Compute Unit (CU)?
* **Mapping:**
  * **Grid:** Global kernel invocation spanning all CUs across the GPU.
  * **Block (Workgroup):** Group of threads executing on **exactly ONE Compute Unit**; shares that CU's 64 KB of Local Data Share (LDS).
  * **Wavefront:** 64 threads (Wave64 in CDNA) scheduled as a unit onto one SIMD pipeline.
  * **Thread:** 1 lane in the SIMD vector ALU.
* **Wavefront Capacity per CU:**
  * Each CDNA CU has 4 SIMD-16 units; each can track 8 wavefront slots.
  * Standard maximum = **32 Wavefronts per CU** ($32 \times 64 = 2,048$ concurrent threads per CU).
  * In CDNA 3 (MI300), up to **40 Wavefronts per CU** can be supported under optimal register usage.
* **Latency Hiding:** By maintaining context for up to 32 wavefronts in hardware registers simultaneously, the CU switches among them in 0 clock cycles when one wavefront stalls waiting for high-latency DRAM/HBM access, keeping SIMD ALUs utilized.
* **Occupancy Bottlenecks:** What lowers active wavefronts below 32:
  1. VGPR pressure (threads needing too many vector registers).
  2. LDS consumption (blocks needing $>32\text{ KB}$ of the 64 KB LDS).
  3. Workgroup limits (max 8–16 blocks per CU).
