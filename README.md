# AMD ROCm™ Certification Study Hub

Workspace, laboratory, and technical notes for completing the **AMD AI Developer Program** and achieving the **AMD ROCm™ Certified Associate** credential, targeting **AMD Instinct™ GPUs (CDNA™ Architecture)**.

---

## 📌 Curriculum & Progress

| # | Course | Status | Key Topics |
| :---: | :--- | :---: | :--- |
| **01** | **ROCm Orientation Setup & Containers** |  Completed | ROCm ecosystem, KFD driver, ROCr runtime, containers |
| **02** | **Instinct Architecture: CDNA/RDNA & MI400** |  Completed (100%) | CUs, SIMD vs MFMA, Wavefronts, 3D chiplets, Infinity Cache |
| **03** | **ROCm Libraries, PyTorch & AI Frameworks** | ⏳ *Up Next* | rocBLAS, MIOpen, PyTorch ROCm dispatch, Triton, RCCL |
| **04** | **HIP Programming & Porting CUDA** | 📋 Planned | Heterogeneous C++, kernel launches, `hipify` tooling |
| **05** | **Performance Optimization & Profiling** | 📋 Planned | `rocprof`, Omnitrace, Omniperf, Roofline model, latency hiding |

---

## 📂 Repository Structure

```
study-amd-rocm-certification/
├── 01-rocm-orientation-setup/         # Course 1 environment notes
├── 02-instinct-architecture/          # Course 2 CDNA hardware notes & Q&A
│   └── NOTES.md                       # Comprehensive hardware architecture reference
├── 03-rocm-libraries-pytorch-ai/      # Course 3 accelerated libraries & AI
├── 04-hip-programming-cuda-porting/   # Course 4 HIP C++ and CUDA porting labs
├── 05-profiling-optimization/         # Course 5 profiling & performance tuning
├── notes/                             # Global concept maps and flashcards
├── labs/                              # Code benchmarks and cloud experiments
└── practice-exams/                    # Quizzes and exam prep
```

---

## 🛠 Target Hardware & Tech Stack

* **Hardware:** AMD Instinct™ MI100, MI200 (MI250/X), MI300 (MI300X / MI300A APU), and MI400 roadmap.
* **Architectures:** CDNA™ (Compute DNA) & RDNA™ (Radeon DNA).
* **Software Stack:** AMD ROCm™ (AMDGPU, KFD, ROCr, HIP Runtime), PyTorch (ROCm), Triton, and rocBLAS / MIOpen.
