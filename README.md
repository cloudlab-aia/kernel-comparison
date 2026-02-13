# Noise-Robust Quantum Kernels on Real 2D Datasets

This repository contains the full experimental pipeline for evaluating
noise robustness of quantum kernel methods under different classical
feature constructions.

We compare:

- Frequency-aware preprocessing (DFT-based)
- Geometric preprocessing (PCA-based)
- Random projections (RP)
- Diagonal phase embeddings
- Classical Kernels (RBF nd Linear) with DFT Preprocessing.

All experiments are run on homogenized real 2D datasets and evaluated
under controlled additive Gaussian noise.

---

## 📦 Environment Setup

Create the environment using conda or mamba:

```bash
conda env create -f environment.yml
# or
mamba env create -f environment.yml

conda activate kernel-comparison
```
No GPU is required. CPU execution is sufficient.

📁 Repository Structure
kernel-comparison/
│
├── environment.yml
├── .gitignore
│
├── experiments/
│   └── run_real2d_matched.py          # Main experiment script

│
├── scripts/

│   ├── bootstrap_all_groups_best_sigma0_regression.py

│   ├── curves_best_accuracy.py

│   ├── dump_num_classes_real2d.py

│   ├── summarize_multi_sigma.py

│   └── viz_entropy_variants_2d.py

│

├── src/

│   ├── classical/

│   │   ├── baselines_matched.py

│   │   ├── common_eval.py

│   │   ├── dataset_factory.py

│   │   ├── nd_matched.py

│   │   ├── phase_embedding.py

│   │   ├── preprocessing.py

│   │   └── real_2d_datasets.py

│   │

│   ├── quantum/

│   │   ├── ae_featuremap.py

│   │   ├── ae_kernel.py

│   │   ├── angle_featuremap.py

│   │   ├── angle_kernel.py


│   │   ├── common_kernel_ops.py

│   │   ├── pes_featuremap.py

│   │   └── pes_kernel.py

│   │

│   ├── hw/

│   │   ├── analysis_hw_overlap.py

│   │   ├── api_testing.py

│   │   └── run_overlap_benchmark.py

│   │

│   └── utils/

│       ├── __init__.py

│       └── secrets_loader.py


🧪 Running the Main Experiments

The full real-2D experiment can be executed with:
```bash
python experiments/run_real2d_matched.py
```

No arguments are required. This script:
1. Loads real 2D datasets
2. Applies preprocessing (DFT / PCA / RP)
3. Constructs classical and quantum kernels
4. Evaluates matched classical baselines
5. Sweeps noise levels
6. Stores results as CSV files
7. All outputs are written to the configured results directory.

📊 Post-Processing and Analysis
Summarize multi-sigma results
```bash
python scripts/summarize_multi_sigma.py
```
Bootstrap regression analysis (robustness slopes)
```bash
python scripts/bootstrap_all_groups_best_sigma0_regression.py
```
Plot best accuracy curves
```bash
python scripts/curves_best_accuracy.py
```
Entropy visualization (diagnostics)
```bash
python scripts/viz_entropy_variants_2d.py
```
🔬 Hardware Overlap Benchmark

To run hardware-based overlap estimation, you must create a local
configuration file with your IBM Quantum credentials.

Step 1: Create config_secrets.json

Create a file in the repository root:
```json
{
    "qiskit": {
        "api_key": "YOUR_API_KEY",
        "instance": "YOUR_INSTANCE"
    },
    "hardware": {
        "backend_name": "BACKEND_NAME"
    }
}
```

Example:
```
{
    "qiskit": {
        "api_key": "abc123...",
        "instance": "ibm-q/open/main"
    },
    "hardware": {
        "backend_name": "ibm_marrakesh"
    }
}
```

Step 2: Run hardware benchmark
```bash
python src/hw/run_overlap_benchmark.py
```

📈 Reproducibility Guarantees
- Fixed Python version (see environment.yml)
- Explicit Qiskit versions
- Deterministic seeds in experiments
- Unified preprocessing pipeline
- All figures generated from stored CSV outputs
- No manual post-processing

To fully reproduce:
```bash
git clone <repo_url>
cd kernel-comparison
mamba env create -f environment.yml
mamba activate kernel-comparison
python experiments/run_real2d_matched.py
```
⚙️ System Requirements
- Linux / macOS recommended
- CPU execution sufficient
- No GPU required

Internet connection required only for hardware runs and dataset downloading

📜 Citation

If you use this code, please cite the accompanying manuscript.

🧠 Notes

- The environment is intentionally minimal and portable.
- Only necessary dependencies are included.
- Hardware functionality is optional.
- The pipeline is fully automated.
