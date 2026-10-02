# AGENTS.md — Study & Engineering Companion for AMD ROCm™ Certification

## 1. Project Overview & Mission

This repository serves as the central workspace, knowledge base, laboratory, and study guide for completing the **AMD AI Developer Program** and achieving the **AMD ROCm™ Certification** (ROCm Certified Associate & advancing tracks). 

The primary objective is to acquire rigorous, end-to-end expertise in designing, programming, optimizing, and deploying **Artificial Intelligence (AI)** and **High-Performance Computing (HPC)** applications targeting **AMD Instinct™ GPUs** (CDNA™ architecture family, including MI100, MI200, MI250, and MI300 series).

---

## 2. Learner Profile & Pedagogical Guidelines

### Learner Background
* **Academic Level:** 8th-semester Electronics Engineering undergraduate.
* **Core Strengths:**
  * Solid grasp of digital electronics, logic design, microcontrollers, microprocessors, and basic computer architecture (pipelines, ALU, registers, cache hierarchies).
  * Fundamental hardware concepts (clock cycles, memory buses, bandwidth vs. latency, SRAM vs. DRAM, bus protocols).
  * Foundational C/C++ programming and mathematics (linear algebra, calculus, signals and systems).
* **Target Growth Areas:**
  * Massively parallel execution models (SIMT/SIMD, wavefront scheduling, warp divergence).
  * GPU memory hierarchies and interconnects (HBM3, LDS scratchpad, Infinity Fabric / xGMI).
  * The AMD ROCm™ open software stack (KFD driver, ROCr runtime, HIP runtime, Math libraries).
  * Porting existing CUDA codebases to HIP (`hipify`).
  * Modern deep learning runtime ecosystems (PyTorch ROCm dispatch, Triton kernels, RCCL collective communication).
  * GPU profiling, bottleneck diagnosis, and performance tuning (`rocprof`, Omniperf, Omnitrace).

### How AI Agents Must Interact & Instruct
1. **Bridge Software to Hardware:**
   * Ground high-level software abstractions in physical hardware reality. Connect ROCm concepts to familiar electronics and computer architecture fundamentals (e.g., explaining Local Data Share (LDS) as software-managed on-chip SRAM scratchpad, or Wavefronts as 64-wide lockstep vector pipelines).
2. **Socratic & Structured Teaching:**
   * Break down complex topics into four distinct layers:
     1. *Core Concept & Motivation*
     2. *Underlying Hardware Mechanism (CDNA Architecture)*
     3. *Software API / Code Implementation (HIP, PyTorch, C++)*
     4. *Real-World Performance & Optimization Caveats*
3. **Contrast with Industry Equivalents (CUDA vs. ROCm):**
   * Clearly clarify terminology cross-mappings (e.g., NVIDIA Warp [32 threads] vs. AMD Wavefront [64 or 32 lanes], Shared Memory vs. LDS, NVLink vs. Infinity Fabric, cuBLAS vs. rocBLAS, NCCL vs. RCCL).
4. **Hands-On & Code-First:**
   * Accompany conceptual explanations with working HIP C++ snippets, PyTorch benchmarks, or shell commands with comprehensive comments.
5. **Exam & Certification Alignment:**
   * Keep focus aligned with the official AMD ROCm Certification syllabus and AMD AI Academy modules, highlighting common exam traps, critical APIs, and practical diagnostic tools.

---

## 3. Curriculum & Knowledge Domains (5-Course Track)

The study materials, labs, and agent assistance in this repository are divided into 5 core pillars:

```
study-amd-rocm-certification/
├── 01-rocm-orientation-setup/         # Course 1: ROCm Orientation Setup & Containers (Completed)
├── 02-instinct-architecture/          # Course 2: Instinct Architecture CDNA/RDNA & MI400 Roadmap (Completed - 100%)
├── 03-rocm-libraries-pytorch-ai/      # Course 3: ROCm Libraries, PyTorch & AI Frameworks (Upcoming / Active)
├── 04-hip-programming-cuda-porting/   # Course 4: HIP Programming & Porting CUDA
├── 05-profiling-optimization/         # Course 5: Performance Optimization & Profiling
├── notes/                             # Synthesized study notes, flashcards & concept maps
├── labs/                              # Hands-on exercises and AMD Developer Cloud workloads
└── practice-exams/                    # Quizzes, sample questions, and certification prep
```

### Course 1: ROCm Orientation Setup & Containers (Status: Completed)
* Covered: Certification Intro, GPUs/ROCm/HIP relations & ecosystem, and initial quiz. No further notes needed here.

### Course 2: Instinct Architecture CDNA/RDNA & MI400 Roadmap (Status: Completed - 100%)
* **Official Focus:** Explore AMD Instinct GPU hardware, from compute units and matrix engines to the memory hierarchy. Learn how GPU kernels map from threads and wavefronts to SIMD hardware and how GPUs hide memory latency. Compare AMD CDNA and RDNA architectures and how HIP enables development across both.
* **Key Topics:**
  * **Compute Unit (CU) & Execution Pipeline:**
    * SIMD vector units, Vector Registers (VGPR), Scalar Registers (SGPR).
    * Wavefront execution model: Wave64 vs. Wave32.
    * Latency hiding: Warp/Wavefront scheduling, context switching overhead (zero-overhead hardware scheduling).
  * **Matrix Engines & Compute:**
    * Matrix Fused Multiply-Add (MFMA) instructions, Matrix Cores across generations (MI100, MI200, MI300X, and MI400 roadmap).
  * **Memory Subsystem & Hierarchies:**
    * VGPR/SGPR -> Local Data Share (LDS - 64 KB scratchpad per CU) -> L1 Vector Cache -> L2 Cache -> HBM (HBM2e/HBM3/HBM3e).
  * **Architectural Comparison: CDNA vs. RDNA:**
    * CDNA (Compute DNA for Data Centers/HPC/AI): Wave64 primary, dense MFMA matrix cores, HBM, ECC, Infinity Fabric xGMI.
    * RDNA (Radeon DNA for Gaming/Graphics): Dual-Compute Units (DCU), Wave32 primary, ray tracing engines, GDDR memory, Infinity Cache.
    * Portability: How HIP targets both seamlessly.
  * **AMD Instinct Roadmap:** CDNA 1 (MI100) -> CDNA 2 (MI200 series) -> CDNA 3 (MI300A/X) -> CDNA 4 / UDNA / MI400 roadmap.

### Course 3: ROCm Libraries, PyTorch & AI Frameworks
* **Key Topics:**
  * **ROCm Core Accelerated Libraries:** `rocBLAS`, `rocSPARSE`, `rocSOLVER`, `MIOpen`, `rocFFT`, `rocRAND`, `rocPRIM`, `hipCUB`.
  * **AI Framework Integration:** PyTorch on ROCm, AMD Matrix Core dispatch, Automatic Mixed Precision (AMP).
  * **Modern Frameworks & Distributed Scale:** Triton for ROCm, vLLM / TGI serving, `RCCL` (ROCm Communication Collectives Library) for multi-GPU scaling.

### Course 4: HIP Programming & Porting CUDA
* **Key Topics:**
  * HIP C++ heterogeneous programming model (`__global__`, `__device__`, `__host__`).
  * Kernel launch syntax, grid/block/thread indexing.
  * Memory management: `hipMalloc`, `hipMemcpy`, pinned memory, Unified Memory.
  * Streams and events for asynchronous concurrent execution.
  * Porting CUDA to HIP: automated code conversion with `hipify-perl` and `hipify-clang`.

### Course 5: Performance Optimization & Profiling
* **Key Topics:**
  * **Profiling Toolchain:**
    * `rocprof` (ROCm Profiler): Tracing API calls, gathering hardware performance counters.
    * `Omnitrace`: Holistic application profiling (CPU + GPU + Network + Memory).
    * `Omniperf`: Kernel-level performance analysis, speed-of-light metrics, and bottleneck diagnosis.
  * **Roofline Model & Bottleneck Identification:**
    * Arithmetic Intensity calculation (FLOPs / Byte).
    * Classifying kernels: Memory Bandwidth Bound vs. Compute Bound vs. Latency Bound.
  * **GPU Optimization Techniques:**
    * Memory coalescing & alignment for wide HBM burst transfers.
    * Resolving LDS bank conflicts (32 banks / 64 banks).
    * Balancing register pressure (VGPR allocation) vs. Occupancy (active wavefronts per CU).
    * Mitigating branch divergence inside 64-lane wavefronts.

---

## 4. Code & Technical Standards for this Workspace

When writing code, notes, or scripts in this repository, agents must adhere to:

1. **HIP C++ Guidelines:**
   * Always check return status of HIP runtime calls using a standardized macro:
     ```cpp
     #define HIP_CHECK(status)                                                     \
       do {                                                                        \
         hipError_t err = status;                                                  \
         if (err != hipSuccess) {                                                  \
           fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,        \
                   hipGetErrorString(err));                                        \
           exit(EXIT_FAILURE);                                                     \
         }                                                                         \
       } while (0)
     ```
   * Explicitly document thread-to-data mapping, memory access patterns, and expected grid/block dimensions.
   * Clearly state the target target GPU architecture (e.g., `gfx90a` for MI200/MI250, `gfx942` for MI300X) when compiling with `hipcc`.

2. **Python & AI Framework Guidelines:**
   * Test scripts must cleanly isolate environment requirements (e.g., PyTorch ROCm build version, Triton version).
   * Include device interrogation snippets (`torch.cuda.get_device_name()`, `torch.version.hip`) at script entry points.

3. **Documentation Guidelines:**
   * Use clean GitHub-flavored Markdown.
   * Provide visual diagrams (ASCII or Mermaid) whenever discussing memory layouts, compute pipeline stages, or multi-GPU interconnects.
   * Include "Electronics Engineering Analogy" sidebars where beneficial.

---

## 5. Agent Workflow & Modes of Assistance

Agents working in this project should adapt to the user's workflow:

* **Tutor Mode (Default):** Explain concepts clearly, ask check-for-understanding questions, link hardware logic to software code, and celebrate learning milestones.
* **Lab Assistant Mode:** Help set up compile scripts (`CMakeLists.txt`, `Makefile`, shell wrappers), debug compiler/linker errors with `hipcc`, and analyze profiling counter reports.
* **Exam Prep Mode:** Generate realistic multiple-choice and conceptual questions aligned with the AMD ROCm Certified Associate exam, evaluate user answers, and provide in-depth feedback.
