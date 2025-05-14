# 🧠 Custom Classifiers and Clustering on Caltech Dataset

This project explores custom implementations of **KNN**, **SVM**, and **K-Means** on the **Caltech 101 dataset** using handcrafted feature extraction methods. The goal is to evaluate their performance in classification and clustering tasks using various features.

---

## 📁 Project Structure

| File              | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `data_filtering.py`     | Preprocessing: filters, scales, and prepares Caltech dataset              |
| `data_augmentation.py` | Applies data augmentation to improve robustness                          |
| `knn.py`                | Custom KNN classifier                                                    |
| `svm.py`                | Custom SVM classifier using basic optimization                           |
| `kmean.py`              | K-Means for unsupervised clustering                                      |
| `kmean__.py`             | K-Means used as classifier via cluster-to-label mapping (classification) |

---

## 🧰 Features Used

- **HOG** (Histogram of Oriented Gradients)  
- **LBP** (Local Binary Patterns)  
- **CLHG** (Custom Local Histogram Gradient)

---

## ✅ Highlights

- Implemented classifiers and clusterers from scratch.
- Custom **KNN** classifier outperformed scikit-learn's KNN in classification accuracy on the extracted features.
- Used **K-Means** both for unsupervised clustering and as a classifier by matching clusters with class labels.
- Conducted preprocessing and augmentation for improved dataset quality.

