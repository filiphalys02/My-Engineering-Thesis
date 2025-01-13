
# Engineering Thesis Project

**EN**: Python package for optimizing processes of tabular data analysis, regression, and classification modeling using AI.

**PL**: Pakiet w języku Python do optymalizacji procesów analizy danych tabelarycznych oraz modelowania regresyjnego i klasyfikacyjnego z wykorzystaniem AI.

## Features
This package provides tools to:
- Streamline the analysis of tabular datasets.
- Build and optimize regression and classification models.
- Simplify the implementation process for data mining tasks.

## Installation and Setup Guide

Step 1: Ensure you have [Git](https://git-scm.com/) and [Anaconda](https://www.anaconda.com/) installed on your computer.

Step 2: Clone the Repository:
```bash
git clone https://github.com/filiphalys02/My-Engineering-Thesis.git
```

Step 3: Navigate to the project directory:
```bash
cd My-Engineering-Thesis
```

Step 4: Create a Virtual Environment based on file `environment.yml` or `conda-lock.yml`:
```bash
conda env create -f environment.yml
```
```bash
conda install conda-lock -c conda-forge
conda-lock install --name VIRTUAL-ENV conda-lock.yml
```

Step 5: Activate the Environment:
```bash
conda activate VIRTUAL-ENV
```

Step 6: Install the Package:
```bash
pip install -e .
```

Step 7: Ready to Use, example usage in Python:
```python
from datamining.classification import BestClassification
```

