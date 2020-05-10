# Clustering Latent Representations for Semi-Supervised Learning
[Final Report](https://drive.google.com/file/d/1eopWAn0TbMRw-Lk3Gz5xSnJmCfkPx2r8) | [Slides](https://drive.google.com/file/d/1SyglGUhVqO3dHjL9lyVg7orZuqIwR32R) | [Video](https://drive.google.com/file/d/1sZk0VWSrgK9xZeF0B7iYqmNXmG4eJ4U9)

Our entry that won the third place award in the semi-supervised learning competition of Yann Lecun's Deep Learning class. It also came in first among the entries that do not use ensembles.

The data for this competition consisted of
- Labeled set 64k images from 1000 different classes from ImageNet 22k (64 images/category) resized to 96x96
- Unlabeled set: 512k images from 1000 classes from ImageNet 22k resized to 96x96 pixels

In this work, we propose using clustering in the latent vector space to impose regularity in the presence of unlabeled data points. Our algorithm uses k-means clustering and learns network parameters simultaneously with the latent cluster centroids in different layers of the network. By making the loss value a function of the deviations from latent cluster centroid, we impose consistent latent representations between the labeled and the unlabeled data to reduce overfitting. 
