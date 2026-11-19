# Machine Learning - Labwork 1: Principal Component Analysis (PCA)

**Full name:** Phi Le Bao Linh  
**Student ID:** 2410532  
(If you work in a pair, add the second member.)

## Description
Labwork 1 report (T1 - PCA): study, cleaning and transformation of two datasets from UCI, followed by Principal Component Analysis (PCA).

| Dataset | Source | Samples | Features | Label |
|---|---|---|---|---|
| Raisin | [UCI #850](https://archive.ics.uci.edu/dataset/850/raisin) | 900 | 7 shape features | `Class` (Kecimen / Besni) |
| Abalone | [UCI #1](https://archive.ics.uci.edu/dataset/1/abalone) | 4177 | 8 (including the categorical `Sex`) | `Rings` (age ≈ Rings + 1.5) |

## Folder structure
```
├── README.md
├── requirements.txt
├── data/
│   ├── raw/             # raw data (saved by the notebooks when they run)
│   └── processed/       # data after cleaning
├── notebooks/
│   ├── 01_raisin.ipynb  # Part 1 + 2 for Raisin
│   └── 02_abalone.ipynb # Part 1 + 2 for Abalone
├── figures/             # exported figures for the report
└── report/
    ├── report_outline.md  # report outline
    └── report.pdf         # version submitted to Google Classroom
```

## How to run
```bash
git clone https://github.com/linhplb/machine-learning-labwork1-pca.git
cd machine-learning-labwork1-pca
pip install -r requirements.txt
jupyter notebook
```
Open `notebooks/01_raisin.ipynb` and `notebooks/02_abalone.ipynb` and choose **Run All**. The data is downloaded automatically with the `ucimlrepo` library.

## Main content (mapped to the assignment)
**Part 1 - Study the dataset:** feature types, data quality, cleaning, data integration, transformation, label, statistics (mean/variance/covariance/correlation), strongest correlated pair, summary.  
**Part 2 - PCA:** preprocessing, explained variance, varying the number of components, 2D visualization, comparison between the original data and PCA.

## Main results
*(TODO: fill in after running, for example: "The first 2 PCs keep X% of the variance of Raisin and Y% of Abalone".)*
