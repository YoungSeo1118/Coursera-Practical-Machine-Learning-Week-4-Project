library(ggplot2)
library(caret)
library(rattle)
library(corrplot)
library(randomForest)
library(CrossValidate)

set.seed(12345)

#Load and Download Data
train_url<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_url<- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

if (!file.exists("./data")) {
        dir.create("./data")
}
if (!file.exists("./data/pml-training.csv")) {
        download.file(train_url, destfile="./data/pml-training.csv", method="curl")
}
if (!file.exists("./data/pml-testing.csv")) {
        download.file(test_url, destfile="./data/pml-testing.csv", method="curl")
}
traincsv <- read.csv("./data/pml-training.csv")
testcsv <- read.csv("./data/pml-testing.csv")

#Cleaning Data
traincsv<-traincsv[,colSums(is.na(traincsv))==0]
traincsv <- traincsv[,-c(1:7)]

nvz<-nearZeroVar(traincsv)
traincsv<-traincsv[,-nvz]

inTrain<-createDataPartition(y=traincsv$classe,p=.7,list=F)
train_data<-traincsv[inTrain,]
validation_data<-traincsv[-inTrain,]

testcsv<-testcsv[,colSums(is.na(testcsv))==0]
testcsv <- testcsv[,-c(1:7)]
traincsv<-traincsv[,-nvz]
test_data<-testcsv

#Correlation matrix of variables in Training Data Set
cor_data<-cor(train_data[,-length(names(train_data))])
corrplot(cor_data,method="color")

#Testing Models
##Decision Tree
model_tree<-train(classe~.,data=train_data,method="rpart", 
                  trControl=trainControl(method="cv",number=4,verboseIter=F))
fancyRpartPlot(model_tree$finalModel)
predict_tree<-predict(model_tree,validation_data)
confusionMatrix(predict_tree,factor(validation_data$classe))
plot(model_tree)
accuracy_tree<-postResample(predict_tree,factor(validation_data$classe))["Accuracy"]
oose_tree<-1-accuracy_tree

##Random Forest
model_forest<-train(classe~.,data=train_data,method="rf",
                    trControl=trainControl(method="cv",number=4,verboseIter=F))
predict_forest<-predict(model_forest,validation_data)
confusionMatrix(predict_forest,factor(validation_data$classe))
plot(model_forest)
accuracy_forest<-postResample(predict_forest,factor(validation_data$classe))["Accuracy"]
oose_forest<-1-accuracy_forest

##Boosting
model_boosting<-train(classe~.,data=train_data,method="gbm",
                      trControl=trainControl(method="cv",number=4,verboseIter=F))
predict_boosting<-predict(model_boosting,validation_data)
confusionMatrix(predict_boosting,factor(validation_data$classe))
plot(model_boosting)
accuracy_boosting<-postResample(predict_boosting,factor(validation_data$classe))["Accuracy"]
oose_boosting<-1-accuracy_boosting

#error table
error_tree<-data.frame(accuracy_tree,oose_tree)
error_forest<-data.frame(accuracy_forest,oose_forest)
error_boosting<-data.frame(accuracy_boosting,oose_boosting)
error_table<-data.frame(Accuracy=c(error_tree[[1]],error_forest[[1]],error_boosting[[1]]),
           Out_Of_Sample_Error=c(error_tree[[2]],error_forest[[2]],error_boosting[[2]]))
rownames(error_table)<-c("Decision Tree","Random Forest","Gradient Boosted Tree")
error_table

#Prediction
prediction<-prediction(model_forest,test_data)
