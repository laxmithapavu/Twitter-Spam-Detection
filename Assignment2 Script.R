#Load in 3 dataset
testingdata1_data <- read.csv("C:/Users/acer/Documents/Dataset-A21/Dataset-A2/dataset/testing_data1.txt")
testingdata2_data <- read.csv("C:/Users/acer/Documents/Dataset-A21/Dataset-A2/dataset/testing_data2.txt")
trainingdata_data <- read.csv("C:/Users/acer/Documents/Dataset-A21/Dataset-A2/dataset/training_data.txt")

#View testingdata1_data, first 6 rows and it's dimension respectively
View(testingdata1_data)
head(testingdata1_data)
dim(testingdata1_data)

#View dimension of testingdata2_data
dim(testingdata2_data)

#View dimension of trainingdata_data
dim(trainingdata_data)

#Load in column names
colnames(testingdata1_data) <- c("account_age","no_follower","no_following","no_userfavourites","no_lists",
                                 "no_tweets","no_retweets","no_tweetfavourites","no_hashtag", "no_usermention",
                                 "no_urls", "no_char", "no_digits","result")

colnames(testingdata2_data) <- c("account_age","no_follower","no_following","no_userfavourites","no_lists",
                                 "no_tweets","no_retweets","no_tweetfavourites","no_hashtag", "no_usermention",
                                 "no_urls", "no_char", "no_digits","result")

colnames(trainingdata_data) <- c("account_age","no_follower","no_following","no_userfavourites","no_lists",
                                 "no_tweets","no_retweets","no_tweetfavourites","no_hashtag", "no_usermention",
                                 "no_urls", "no_char", "no_digits","result")

#Check one of the datasets to confirm the renaming of the columns
head(trainingdata_data)





#RANDOM FOREST MODEL
#Install dplyr package
install.packages("dplyr") 
library(dplyr)
#Install caret package
install.packages("caret")
library(caret)

#Logical variable for detecting spammer
trainingdata_data <- trainingdata_data %>%
  mutate(nonspammer = result == "spammer") %>%
#remove nonspammer variable
  select(-nonspammer)

#Fit a Random Forest Model (using ranger)
train_data_rf_fit <- train(as.factor(result) ~ .,
                           data = trainingdata_data,
                           method = "ranger")
train_data_rf_fit

#Predict the outcome on a testingdata1 set
trainingdata_rf_pred1 <- predict (train_data_rf_fit, testingdata1_data)
trainingdata_rf_pred1
#Compare predicted outcome and true outcome
confusionMatrix(trainingdata_rf_pred1, as.factor(testingdata1_data$result))

#Predict the outcome on a testingdata2 set
trainingdata_rf_pred2 <- predict (train_data_rf_fit, testingdata2_data)
trainingdata_rf_pred2
#Compare predicted outcome and true outcome
confusionMatrix(trainingdata_rf_pred2, as.factor(testingdata2_data$result))





#NAIVE BAYES MODEL
#Install naivebayes package
install.packages("naivebayes") 
library(naivebayes)

#Fit a Naive Bayes Model (using naive_bayes)
train_data_nb_fit <- train(as.factor(result) ~ .,
                           data = trainingdata_data,
                           method = "naive_bayes")
train_data_nb_fit

#Predict the outcome on a testingdata1 set
trainingdata_nb_pred1 <- predict (train_data_nb_fit, testingdata1_data)
trainingdata_nb_pred1
#Compare predicted outcome and true outcome
confusionMatrix(trainingdata_nb_pred1, as.factor(testingdata1_data$result))

#Predict the outcome on a testingdata2 set
trainingdata_nb_pred2 <- predict (train_data_nb_fit, testingdata2_data)
trainingdata_nb_pred2
#Compare predicted outcome and true outcome
confusionMatrix(trainingdata_nb_pred2, as.factor(testingdata2_data$result))





#KNN MODEL
#Fit a KNN Model (using knn)
train_data_knn_fit <- train(as.factor(result) ~ .,
                           data = trainingdata_data,
                           method = "knn")
train_data_knn_fit

#Predict the outcome on a testingdata1 set
trainingdata_knn_pred1 <- predict (train_data_knn_fit, testingdata1_data)
trainingdata_knn_pred1
#Compare predicted outcome and true outcome
confusionMatrix(trainingdata_knn_pred1, as.factor(testingdata1_data$result))

#Predict the outcome on a testingdata2 set
trainingdata_knn_pred2 <- predict (train_data_knn_fit, testingdata2_data)
trainingdata_knn_pred2
#Compare predicted outcome and true outcome
confusionMatrix(trainingdata_knn_pred2, as.factor(testingdata2_data$result))





#NEURAL NETWORK MODEL
#Install nnet package
install.packages("nnet") 
library(nnet)

#Fit a Neural Netwok Model (using nnet)
train_data_nn_fit <- train(as.factor(result) ~ .,
                           data = trainingdata_data,
                           method = "nnet")
train_data_nn_fit

#Predict the outcome on a testingdata1 set
trainingdata_nn_pred1 <- predict (train_data_nn_fit, testingdata1_data)
trainingdata_nn_pred1
#Compare predicted outcome and true outcome
confusionMatrix(trainingdata_nn_pred1, as.factor(testingdata1_data$result))

#Predict the outcome on a testingdata2 set
trainingdata_nn_pred2 <- predict (train_data_nn_fit, testingdata2_data)
trainingdata_nn_pred2
#Compare predicted outcome and true outcome
confusionMatrix(trainingdata_nn_pred2, as.factor(testingdata2_data$result))






#Stochastic Gradient Boosting Machine (GBM) Model
#Install gbm package
install.packages("gbm") 
library(gbm)

#Fit a gbm Model (using gbm)
train_data_gbm_fit <- train(as.factor(result) ~ .,
                           data = trainingdata_data,
                           method = "gbm")
train_data_gbm_fit

#Predict the outcome on a testingdata1 set
trainingdata_gbm_pred1 <- predict (train_data_gbm_fit, testingdata1_data)
trainingdata_gbm_pred1
#Compare predicted outcome and true outcome
confusionMatrix(trainingdata_gbm_pred1, as.factor(testingdata1_data$result))

#Predict the outcome on a testingdata2 set
trainingdata_gbm_pred2 <- predict (train_data_gbm_fit, testingdata2_data)
trainingdata_gbm_pred2
#Compare predicted outcome and true outcome
confusionMatrix(trainingdata_gbm_pred2, as.factor(testingdata2_data$result))