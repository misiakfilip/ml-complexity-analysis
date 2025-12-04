# ML Complexity Analysis: Does Complexity Work?

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

> **A comprehensive empirical study comparing simple vs. complex machine learning algorithms across different data scenarios.**

## 📋 Table of Contents

- [Overview](#overview)
- [Research Question](#research-question)
- [Key Findings](#key-findings)
- [Datasets](#datasets)
- [Models Compared](#models-compared)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## 🎯 Overview

This project systematically evaluates whether more complex machine learning algorithms always outperform simpler ones in regression tasks. Through rigorous experimentation across two distinct scenarios, we analyze the **trade-offs between model complexity, accuracy, training time, and interpretability**.

## 🔬 Research Question

**"Does complexity work?"** 

We investigate this by comparing a hierarchy of algorithms from simplest to most advanced:
1. Linear Regression (baseline)
2. Ridge / Lasso / Elastic Net (regularization)
3. Polynomial Regression (degree 2, 3)
4. Support Vector Regression - SVR (kernel trick)
5. Random Forest (ensemble methods)
6. XGBoost / LightGBM (gradient boosting)

## 📊 Datasets

### Scenario 1: Synthetic Linear Data
- **Purpose:** Ideal conditions for linear regression
- **Characteristics:**
  - n = 5,000 samples
  - m = 15 features (all informative)
  - True function: y = β₀ + β₁x₁ + ... + β₁₅x₁₅ + ε, where ε ~ N(0, σ=5)
  - Minimal noise for near-perfect linearity

### Scenario 2: Bike Sharing Dataset
- **Source:** [UCI ML Repository](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
- **Purpose:** Strongly nonlinear relationships
- **Characteristics:**
  - n = 17,389 hourly records (2011-2012)
  - m = 12 features (temporal, weather, seasonal)
  - Target: Hourly bike rental counts
  - **Nonlinearities:**
    - Hour × Weather × Season interactions
    - Temperature thresholds
    - Rush hour peaks (8am, 5pm)
    - Weekend vs. weekday patterns

## 📁 Project Structure

```
ml-complexity-analysis/
│
├── notebooks/
│   └── analysis.ipynb           # Main analysis notebook
│
├── requirements.txt
├── README.md
└── LICENSE
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

```bash
# Clone the repository
git clone https://github.com/misiakfilip/ml-complexity-analysis.git
cd ml-complexity-analysis

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
xgboost>=1.5.0
jupyter>=1.0.0
```

## 📖 Usage

### Running the Analysis

```bash
# Start Jupyter Notebook
jupyter notebook

# Open notebooks/ml-complexity-analysis.ipynb and run all cells
```

### Running from Command Line (if converted to .py)

```bash
python notebooks/ml-complexity-analysis.py
```

The notebook will:
1. ✅ Load and preprocess datasets
2. ✅ Train all models with hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
3. ✅ Evaluate on test sets (MSE, RMSE, MAE, R²)
4. ✅ Measure training and prediction times
5. ✅ Generate comparison visualizations
6. ✅ Run 10× smaller dataset experiment
7. ✅ Output comprehensive analysis report

## 📈 Results


### Training Time vs. Accuracy

![Trade-off Analysis](results/comparison_tradeoff.png)

### Impact of Data Size (10× reduction)

| Model | Scenario 1 ΔR² | Scenario 2 ΔR² |
|-------|---------------|---------------|
| Ridge | **0.027** 🏆 | **0.001** 🏆 |
| Linear | -0.002 | 0.001 |
| XGBoost | 0.061 | 0.037 |
| Random Forest | 0.076 | 0.082 |
| Lasso | **0.590** ❌ | 0.001 |

**Key insight:** Ridge is the safest choice for small datasets!

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Ideas for Extension
- Add more scenarios (high-dimensional, time-series)
- Include deep learning models (MLPs, CNNs for tabular)
- Add cross-validation stability analysis
- Implement SHAP values for interpretability
- Compare with AutoML solutions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this project in your research or work, please cite:

```bibtex
@misc{ml_complexity_analysis_2024,
  author = Filip Misiak, Adam Kowalczyk,
  title = {ML Complexity Analysis: Does Complexity Work?},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/misiakfilip/ml-complexity-analysis}
}
```

## 🙏 Acknowledgments

- **Datasets:**
  - Bike Sharing Dataset: Fanaee-T, H. (2013). [UCI ML Repository](https://doi.org/10.24432/C5W894)
  
- **Libraries:**
  - [scikit-learn](https://scikit-learn.org/)
  - [XGBoost](https://xgboost.readthedocs.io/)
  - [pandas](https://pandas.pydata.org/)
  - [matplotlib](https://matplotlib.org/)
  - [seaborn](https://seaborn.pydata.org/)

## 📧 Contact

**Author:** Filip Misiak, Adam Kowalczyk 
**Email:** filip.misiak11@example.com  
**LinkedIn:** [Filip Misiak](https://linkedin.com/in/filip-misiak-031090281)  

---

⭐ **If you found this project useful, please consider giving it a star!** ⭐
