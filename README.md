# 📊 Analysis of Fashion MNIST Dataset
**Author:** Wiktoria Maksymiak  
**Album Number:** 418853  

---

## Summary and Discussion of the Data

Fashion-MNIST is a dataset containing grayscale images of fashion items provided by Zalando. Each image represents one of 10 classes:

| Index | Label        |
|-------|--------------|
| 0     | T-shirt/top  |
| 1     | Trouser      |
| 2     | Pullover     |
| 3     | Dress        |
| 4     | Coat         |
| 5     | Sandal       |
| 6     | Shirt        |
| 7     | Sneaker      |
| 8     | Bag          |
| 9     | Ankle boot   |

The dataset is balanced (6,000 images per class), has no missing values, and is divided into training `(60,000, 785)` and test `(10,000, 785)` sets. The first column contains labels; the rest are pixel intensities (0–255).

Images were normalized to [0, 1], and a subset was created using `train_test_split(test_size=0.8, stratify=labels)`.

![Sample Images](https://github.com/w-maks/Analysis-of-Fashion-Mnist-Dataset/blob/main/Images/1/clothes.png)
*Some examples from the dataset*

![Describe Output](https://github.com/w-maks/Analysis-of-Fashion-Mnist-Dataset/blob/main/Images//1/describe.png)  
*Statistics from `describe()` on training data*

---

## Dimensionality Reduction

### PCA (Principal Component Analysis)
- Linear method preserving global structure
- Sensitive to outliers
- Retained 95% variance → 184 dimensions

### t-SNE (t-distributed Stochastic Neighbor Embedding)
- Non-linear method
- Preserves local structure
- Effective for non-linearly separable data

Visualizations after reduction:

| PCA (2D) | PCA (3D) |
|---------|----------|
| ![PCA 2D](https://github.com/w-maks/Analysis-of-Fashion-Mnist-Dataset/blob/main/Images/2/pca%20180d%20onto%202.png) | ![PCA 3D](https://github.com/w-maks/Analysis-of-Fashion-Mnist-Dataset/blob/main/Images/2/pca%20180d%20onto%203.png) |

| PCA vs t-SNE (2D) | PCA vs t-SNE (3D) |
|------------------|-------------------|
| ![2D](https://github.com/w-maks/Analysis-of-Fashion-Mnist-Dataset/blob/main/Images/2/pca%20tsne%202d.png) | ![3D](https://github.com/w-maks/Analysis-of-Fashion-Mnist-Dataset/blob/main/Images/2/pca%20tsne%203d.png) |

---

## Clustering and Outlier Detection

### Evaluation Metrics:
- **Silhouette Score**
- **Adjusted Rand Index (ARI)**
- **Davies-Bouldin Index**
- **Calinski-Harabasz Index**

### Hierarchical Clustering
Tested different linkage methods (`ward`, `average`, etc.) and dimensions (PCA & t-SNE).

Best results:

| Reduction | Dim | Score | Clusters | Linkage |
|-----------|-----|-------|----------|---------|
| PCA       | 184 | 0.1829 | 4        | Ward    |
| t-SNE     | 2   | 0.4134 | 10       | Ward    |

![PCA Dendrogram](https://github.com/w-maks/Analysis-of-Fashion-Mnist-Dataset/blob/main/Images/3/hierarchical/dendograms/180%2010.png)  
![PCA Clusters](https://github.com/w-maks/Analysis-of-Fashion-Mnist-Dataset/blob/main/Images/3/hierarchical/plots/180%2010.png)

![t-SNE Dendrogram](https://github.com/w-maks/Analysis-of-Fashion-Mnist-Dataset/blob/main/Images/3/hierarchical/dendograms/tsne%2010.png)  
![t-SNE Clusters](https://github.com/w-maks/Analysis-of-Fashion-Mnist-Dataset/blob/main/Images/3/hierarchical/plots/tsne%2010.png)

---

### KMeans Clustering

Best model on t-SNE (2D) with `n=10` clusters:

| Reduction | Dim | Clusters | ARI   | SLI   | D-B  | C-H   | Total Score |
|-----------|-----|----------|-------|-------|------|-------|--------------|
| t-SNE     | 2   | 10       | 0.426 | 0.433 | 0.800 | 15854 | **0.9371**   |

![t-SNE + KMeans](https://github.com/w-maks/Analysis-of-Fashion-Mnist-Dataset/blob/main/Images/3/kmeans/best/tsne.png)

---

### DBSCAN Clustering

Used only on t-SNE 2D due to computational cost.

Best params:
- `ε = 3.5`
- `min_samples = 58`
- 10 clusters
- Silhouette Score: 0.3566
- ARI: 0.4233

![DBSCAN Clusters](https://github.com/w-maks/Analysis-of-Fashion-Mnist-Dataset/blob/main/Images/3/dbscan/clusters.png)  
![Outliers](https://github.com/w-maks/Analysis-of-Fashion-Mnist-Dataset/blob/main/Images/3/dbscan/outliers.png)

---

## Classification

### K-Nearest Neighbors (KNN)

- Tested `k = 1–20`, metrics: Euclidean, Manhattan, Cosine
- Best: `k = 4`, metric = Manhattan
- Accuracy, Precision, Recall: **0.832**

![KNN Confusion Matrix](https://github.com/w-maks/Analysis-of-Fashion-Mnist-Dataset/blob/main/Images/5/knn/cm.png)  
![KNN Mismatches](https://github.com/w-maks/Analysis-of-Fashion-Mnist-Dataset/blob/main/Images/5/knn/wrong.png)

---

### Decision Tree

- Tested depths `2–10`
- Best: depth = 10
- Accuracy, Precision, Recall: **0.768**

![Decision Tree Confusion Matrix](https://github.com/w-maks/Analysis-of-Fashion-Mnist-Dataset/blob/main/Images/5/tree/cm.png)  
![Decision Tree Structure](https://github.com/w-maks/Analysis-of-Fashion-Mnist-Dataset/blob/main/Images/5/tree/depth10.png)  
![Decision Tree Mismatches](https://github.com/w-maks/Analysis-of-Fashion-Mnist-Dataset/blob/main/Images/5/tree/wrong.png)

---

## Classification Summary

| Algorithm     | Accuracy | Precision | Recall | Time     |
|---------------|----------|-----------|--------|----------|
| **KNN**       | 0.832    | 0.832     | 0.832  | ~10 min  |
| Decision Tree | 0.768    | 0.769     | 0.768  | ~2 min   |

---

## ChatGPT’s Parallel Analysis

### Dataset Summary

![Chat Summary](https://github.com/w-maks/Analysis-of-Fashion-Mnist-Dataset/blob/main/Images/chat/summary.png)

### Dimensionality Reduction

![Chat Reduction](https://github.com/w-maks/Analysis-of-Fashion-Mnist-Dataset/blob/main/Images/chat/reduction.png)

### Clustering

![Chat Clustering](https://github.com/w-maks/Analysis-of-Fashion-Mnist-Dataset/blob/main/Images/chat/clusters.png)

### Classification

![Chat Report](https://github.com/w-maks/Analysis-of-Fashion-Mnist-Dataset/blob/main/Images/chat/cr.png)  
![Chat Confusion Matrix](https://github.com/w-maks/Analysis-of-Fashion-Mnist-Dataset/blob/main/Images/chat/cm.png)

---

## Conclusion

- **Best combination:** KMeans + t-SNE (2D)
- **Best classification:** Random Forest (ChatGPT), KNN (manual)
- **Best for outliers:** DBSCAN
- **Most efficient:** Decision Tree

---

*Thanks for reading! For full code and visualizations, see the attached notebooks and scripts.*
