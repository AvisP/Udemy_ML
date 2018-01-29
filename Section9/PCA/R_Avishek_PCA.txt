# PCA

#setwd("C:/Users/apaul11/Dropbox/Kaggle/Section 9/PCA")

# importing the dataset
dataset = read.csv('Wine.csv')

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Customer_Segment, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[, -14] = scale(training_set[,-14])
test_set[,-14] = scale(test_set[,-14])

# Applying PCA
#install.packages('caret')
library(caret)
#install.packages('e1071')
library(e1071)
pca = preProcess(x = training_set[-14], method = 'pca',pcaComp = 2)
training_set = predict(pca, training_set)
training_set = training_set[c(2,3,1)]
test_set = predict(pca, test_set)
test_set = test_set[c(2,3,1)]

# Fitting SVM to the Training Set
library(e1071)
classifier = svm(formula = Customer_Segment ~.,
			data = training_set,
			type = 'C-classification',
			kernel = 'linear')

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-3])

# Making the confusion matrix
cm = table(test_set[,3], y_pred)



			