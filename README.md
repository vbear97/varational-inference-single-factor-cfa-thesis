# Project: Variational Inference Thesis 


This codebase is the refactored version of the codebase used to generate results for my mathematics master's thesis (completed December 2022). It represents an ongoing progress to: 

* Cleanly refactor raw research code into a presentable and sustainable codebase 
* Build on results in my master's thesis and address any limitations 

## Design Choices 

Basically everything is coded from scratch to demonstrate my understanding and also to maximise flexibility.

## Thesis Overview 

### Background Problem: Accurate, Efficient Bayesian Inference 

Traditional Bayesian inference utilises Markov Chain Monte Carlo (MCMC) as the "gold standard" for approximating Bayesian posteriors. However, MCMC can be very slow for multi-dimensional statistical models. 

An alternative method for approximating posteriors is an optimisation technique called **Variational Inference (VI)**. It is based on specifying a family of known, tractable probability distributions - what is termed a "variational family" - and choosing the candidate $q^{*}(\theta)$ that is "closest" to the true posterior $p(\theta | x)$. 

### Thesis/Codebase 

**Purpose** 

**Purpose 1: Improve VI Accuracy**

This codebase explores what happens when we experiment with different "closeness" measures. The traditional "closeness" measure is the KL Divergence, and here we extend these closeness measures to the Renyi Divergence family.

In particular, we seek to solve the problem of variance underestimation: traditional VI that minimises the KL divergence tends to underestimate posterior variances. 

**Purpose 2: Demonstrate that Renyi Divergence VI can be implemented in a black-box manner**

Dang (2022) applied KLVI to a single factor confirmatory factor analysis model using analytical methods. Using this same statistical model, I demonstrate that VI techniques 

**Background Statistical Model**

We focus on applying VI to a single confirmatory factor analysis model. 

# Installation 

### Prerequisites
- Python 3.10 or later 
    - Note that Python 3.10 is the latest Python version that is compatible with the ```pystan``` package, which is required for running Hamiltonian MCMC 
- pip package manager

### Steps 
1. Clone the repository:
```bash
  git clone https://github.com/vbear97/varational-inference-single-factor-cfa-thesis
```
2. Install dependencies 

```bash
pip install -r requirements.txt
```

#### Installation Notes - Pystan on Apple M1 

See additional installation instructions for installing the ```pystan``` package on Apple M1: https://pystan.readthedocs.io/en/latest/faq.html. 

### Optional Steps 
3. Install development dependencies (for code formatting):
```bash 
pip install autopep8

```
4. Set up Git hooks for automatic PEP-8 formatting:
```bash
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash

# Format Python files with autopep8
autopep8 --in-place --aggressive --aggressive $(find . -name "*.py")

# Add the formatted files back to staging
git add $(find . -name "*.py")

echo "Code formatted with autopep8"
EOF

# Make executable
chmod +x .git/hooks/pre-commit
```

# Getting Started

The core workflow is captured in the ```notebooks``` folder: 

* [Appendix: Tests](notebooks/Appendix:Tests.ipynb) outlines how to test our VI implementation using simulated data and a mix of degenerate/uninformative priors 
* [Part 1: KLVI](notebooks/Part1_Black_Box_KLVI.ipynb) implements the black-box version of "vanilla" VI that utilises the KL Divergence, which is the standard measure. 
* [Part 2: Renyi Divergence VI](notebooks/Part2_Renyi_Divergence_VI.ipynb) outlines core thesis results in summarised form and implements black-box Renyi Divergence VI.

# Repository Layout 
```bash
variational-inference-single-factor-cfa-thesis/
├── README.md
├── requirements.txt
├── .gitignore
├── src/                            # Variational inference source code
│   ├── analysis/                   # Analyse fitted models
│   └── analytical_variational_inference/  # VI via MFVB algorithm 
│   └── mcmc/                              # MCMC
├── notebooks/                             # Main workflows 
```


