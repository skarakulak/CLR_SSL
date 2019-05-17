# Clustering Latent Representations for Semi-Supervised Learning
[Final Report](https://drive.google.com/open?id=1asLwfSbhum7zl_qfLQiJ9RStQF5IuzBD) | [Slides](https://drive.google.com/open?id=1XfpmknrjER2isSSaikX1eAYrwhpw8ajk) | [Video](https://drive.google.com/open?id=15JvtquQE2YYWSpPvByxiRVn7PM9z4IXy)

Our entry that won the third place award in the semi-supervised learning competition of Yann Lecun's Deep Learning class

The data for this competition consisted of
- Labeled set 64k images from 1000 different classes from ImageNet 22k (64 images/category) resized to 96x96
- Unlabeled set: 512k images from 1000 classes from ImageNet 22k resized to 96x96 pixels

In this work, we propose using clustering in the latent vector space to impose regularity in the presence of unlabeled data points. Our algorithm uses k-means clustering and learns network parameters simultaneously with the latent cluster centroids in different layers of the network. By making the loss value a function of the deviations from latent cluster centroid, we impose consistent latent representations between the labeled and the unlabeled data to reduce overfitting. 

 This entry came in third overall and it was first among the entries that do not use ensembles.
