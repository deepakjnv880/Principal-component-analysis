# Principal-component-analysis
Implement principal component analysis without library on mnist handwritten data images

# Problem statement
Dataset: MNIST digit dataset: http://yann.lecun.com/exdb/mnist/


a) Given a set of images of any single digit from the above dataset, compute a covariance matrix and the Eigen vector basis using the vectorized representation of these images. Project each image onto this PCA space using i) all Eigen vectors ii) Selected Eigen vectors with different values of energy thresholds (computed using the top k Eigen values). Reconstruct the original images using the projected data obtained in the cases above and comment on the quality of reconstruction based for different cases.  

b) Now add up to 20% noise to the images, and perform the same experiment as above. Comment on the tradeoff between denoising and reconstruction quality for different cases of no. of principal components. 
# Details
There are three folder. 
<br> All training data digit wise in /training folder
<br> All testing data digit wise in /testing folder
<br> All result for pca reconstruction is in /result folder
# Package required
Install opencv2:  

	sudo apt-get install python3-opencv

# How to run
Go to the folder and run below command:

	python3 main.py
