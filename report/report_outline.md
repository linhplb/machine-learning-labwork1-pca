# Report outline (PDF, at least 2 pages)

**Full name + Student ID** (mandatory, at the top of the report)

## 1. Datasets
- Short description of Raisin and Abalone: source, number of samples, number of features
- Feature type table (discrete/continuous, quantitative/qualitative, numerical/categorical)

## 2. Data quality and cleaning
- Issues found (missing values, duplicates, Height = 0, outliers) and their impact
- How they were handled + before/after comparison table
- Data integration: needed or not, and why
- Transformation: standardization, one-hot encoding of Sex

## 3. Labels and statistics
- Label, meaning, type of ML task
- Correlation heatmaps, strongest pair for each dataset, comments
- Analysis of the categorical feature (chi-square, ANOVA)

## 4. PCA
- Preprocessing, how the PCs are computed and selected
- Explained variance table/plot, comparison of high- and low-variance PCs
- Effect of the number of components
- 2D scatter PC1-PC2 colored by label, comments on clusters/overlaps

## 5. Comparison and conclusion
- Original vs PCA: information preservation, interpretability, visualization
- Advantages and limitations of PCA

GitHub link: https://github.com/linhplb/machine-learning-labwork1-pca
