# Agentic Knowledge Distillation Query Planner

A research implementation for intelligent query optimization using knowledge distillation and reinforcement learning techniques.

## Overview

This project implements and compares multiple approaches to query optimization:

- **Contextual Bandit Optimization** (Bao-lite) with hint selection
- **Learned Cardinality Estimation** (Kipf-style) for join ordering
- **Reinforcement Learning Join Planning** (Neo-lite) with tabular policy
- **Teacher-Student Knowledge Distillation** with UCB1 exploration
- **Baseline Heuristics** and random planning for fair comparison

## Features

- Comprehensive evaluation on NYC Taxi, IMDb, and TPC-H workloads
- Memory and latency constraint handling
- Excel-based result reporting
- Reproducible experiments with seed control
- Performance metrics and analysis tools

## Requirements

- Python 3.7+
- Core dependencies: numpy, pandas, duckdb, sklearn, matplotlib, openpyxl, tqdm

## Quick Start

### Prepare Data
```bash
python agentic_kd_planner.py --prepare
```

### Run Experiments
```bash
python agentic_kd_planner.py --run all --mem_gb 4 --latency_ms 500 --excel artifacts/results.xlsx --seed 42
```

## Project Structure

```
Agent/
├── agentic_kd_planner.py    # Main implementation
├── artifacts/                # Results and outputs
│   ├── data/                # Dataset files
│   ├── figs/                # Generated figures
│   └── results.xlsx         # Experiment results
└── README.md                # This file
```

## Research Areas

- Query optimization
- Knowledge distillation
- Reinforcement learning
- Database systems
- Machine learning for databases

## License

[Add your license here]

## Citation

If you use this code in your research, please cite:

```
[Add citation information here]
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
