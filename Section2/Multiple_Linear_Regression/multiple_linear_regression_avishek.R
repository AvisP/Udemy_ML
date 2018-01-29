# Importing the dataset
dataset = read.csv('50_Startups.csv')
# Encoding
dataset$State = factor(dataset$State,
                         levels = c('New York','California','Florida'),
                         labels = c(1,2,3))
# Splitting data Set
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature scaling is not required as was done in simple linear regression

# Fitting Multiple Linear Regression to training set

#regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State)
regressor = lm(formula = Profit ~ .,
               data=training_set)
# to view summary -> summary(regressor)
# check for p-value. Lower the p value higher is the statistical significance

# Prediciting the Test set Result

y_pred = predict(regressor, newdata = test_set)

# Building the optimal model using backward Elimination
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data=dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data=dataset)
summary(regressor)