library(data.table)
library(dplyr)
library(caret)
library(doParallel)
library(glmnet)
library(party)
dload <- tempfile()
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
download.file(url, dload)
#get column names
cnames <- fread(text = readLines(unzip(dload,"UCI HAR Dataset/features.txt")), col.names = c("colId", "features"))
#get activity names
anames <- fread(text = readLines(unzip(dload,"UCI HAR Dataset/activity_labels.txt")), col.names = c("actId", "activity"))

#observations

dim(cnames)
#1.561 predictors - too large
check <- as.data.frame(unique(cnames$features)) #get unique column names
names(check) <- "features"
check %>% left_join(cnames) %>% group_by(features) %>% summarize(n=n()) %>% filter(n>1) %>% head # get a few duplicates
#2.duplicate feature names, hence we will just use the feature index


#get training data set into three objects
#x containing the predictors
#y containing the activites associated
#sid contains the subject ID associated with the training data which can be used for grouping
x_train <- fread(text = readLines(unzip(dload, "UCI HAR Dataset/train/X_train.txt")))
y_train <- fread(text = readLines(unzip(dload, "UCI HAR Dataset/train/y_train.txt")),col.names = "activity")
sid_train <- fread(text = readLines(unzip(dload,"UCI HAR Dataset/train/subject_train.txt")), col.names = "subjectId")

#now do the same for test data 
#x containing the predictors
#y containing the activites associated
#sid contains the subject ID associated with the test data which can be used for grouping
x_test <- fread(text = readLines(unzip(dload, "UCI HAR Dataset/test/X_test.txt")))
y_test <- fread(text = readLines(unzip(dload, "UCI HAR Dataset/test/y_test.txt")),col.names = "activity")
sid_test <- fread(text = readLines(unzip(dload,"UCI HAR Dataset/test/subject_test.txt")), col.names = "subjectId")

#since y is categorical data, lets convert it into a factor
y_train$activity <- as.factor(y_train$activity)
y_test$activity <- as.factor(y_test$activity)

#combine the x and y training and test data 
data_train <- cbind(x_train,y_train) 
data_valid <- cbind(x_test,y_test)
#make sure activity is a factor
class(data_train$activity)

x_list <- names(x_train)
#observations
clean <- ifelse(sapply(x_list, function(x) 
  any(data_train$x < -1 | data_train$x > 1 | is.na(data_train$x))),"NO", "YES")
all(clean == "YES")
#the range of the values in the predictors appears to have been adjusted to vary between -1 and +1 


#We will now split the train set into training and testing set as we will use the test set as our validation 
set.seed(1,sample.kind = "Rounding")
test_ind <- createDataPartition(data_train$activity,times = 1,p=.2,list = FALSE)
train_set <- data_train[-test_ind,]
test_set <- data_train[test_ind,]

#since there are 561 variables, and one multinominal prediction
#we need to reduce the number of predictors to a minimum and pick a few important variables

#we will use the multinominal LASSO/RIDGE and Elastic net regression

# saved RDS files are available on the google drive
# https://drive.google.com/drive/folders/1BEpMB_7cgVT3JSgxU6sSVaxWwx2ztqjc?usp=sharing

musttrain <- ""
musttrain <- as.character(readline(prompt="Enter Y - if you want to Train or N - if you want to load model from saved rds file provided : "))
if(toupper(musttrain) == "Y") {
     
#set up a Ridge model
control <- trainControl(method = "repeatedcv",
                        number = 10,
                        repeats = 5)
#since the data is regularized, the weights applied are very close to zero for highest accuracy
grid <- expand.grid(alpha=0, #set to 0 for ridge model
                    lambda=10^-5)

#make a cluster of number of processors -1
cl <- makePSOCKcluster(7)
registerDoParallel(cl)




set.seed(1,sample.kind = "Rounding")

ridge_model <- train(activity~.,
                     train_set,
                     method = "glmnet",
                     tuneGrid = grid,
                     trControl = control)

##stop cluster
stopCluster(cl)

#get the details from the model
ridge_model

#as we can see the highest accuracy is obtained with the lowest lambda
plot(ridge_model)

plot(ridge_model$finalModel,xvar = "lambda",label = TRUE)

plot(ridge_model$finalModel,xvar = "dev",label = TRUE)

#get the variables of importance
plot(varImp(ridge_model,scale = TRUE),top = 20)

#Lasso model

grid <- expand.grid(alpha=1, #set to 1 for lasso model
                    lambda=10^-5)
##start cluster
registerDoParallel(cl)

set.seed(1,sample.kind = "Rounding")

lasso_model <- train(activity~.,
                     train_set,
                     method = "glmnet",
                     tuneGrid = grid,
                     trControl = control)
##stop cluster
stopCluster(cl)

#get the details from the model
lasso_model

plot(lasso_model)

plot(lasso_model$finalModel,xvar = "lambda",label = TRUE)

plot(lasso_model$finalModel,xvar = "dev",label = TRUE)

#get the variables of importance
varImp(lasso_model)
plot(varImp(lasso_model,scale = TRUE),top = 20)

#Elasticnet model

grid <- expand.grid(alpha=seq(0,1,length=10), #set 0 to 1 for Elastic model
                    lambda=10^-5)
##start cluster
registerDoParallel(cl)

set.seed(1,sample.kind = "Rounding")

elastic_model <- train(activity~.,
                     train_set,
                     method = "glmnet",
                     tuneGrid = grid,
                     trControl = control)
##stop cluster
stopCluster(cl)

#get the details from the model
elastic_model

plot(elastic_model)

plot(elastic_model$finalModel,xvar = "lambda",label = TRUE)

plot(elastic_model$finalModel,xvar = "dev",label = TRUE)

#get the variables of importance
varImp(elastic_model)
plot(varImp(elastic_model,scale = TRUE),top = 20)

#saving the three models
saveRDS(ridge_model,"Ridge_model.rds")
saveRDS(lasso_model,"Lasso_model.rds")
saveRDS(elastic_model,"Elastic_model.rds")

}
#compare models
if(toupper(musttrain) == "N") {
  ridge_model <-readRDS("Ridge_model.rds")
  lasso_model <-readRDS("Lasso_model.rds")
  elastic_model <-readRDS("Elastic_model.rds")
}

#we can compare the models by combining them to an array and listing out their properties.
models <- list(ridge = ridge_model,lasso = lasso_model,elastic = elastic_model)
resample <- resamples(models)
summary(resample)

#we observe model accuracies from the train set


#Using the test set to predict accuracy
#ridge accuracy
ridge_predict <- predict(ridge_model,test_set)
mean(as.numeric(as.character((ridge_predict))) == as.numeric(as.character(test_set$activity)))

#lasso accuracy
lasso_predict <- predict(lasso_model,test_set)
mean(as.numeric(as.character((lasso_predict))) == as.numeric(as.character(test_set$activity)))

#elasticnet accuracy
elastic_predict <- predict(elastic_model,test_set)
mean(as.numeric(as.character((elastic_predict))) == as.numeric(as.character(test_set$activity)))


#errors

#confusion matricies
cm_ridge <- confusionMatrix(ridge_predict,test_set$activity)
#ridge cm
cm_ridge$table
#ridge error (misclassification %)
1-sum(diag(cm_ridge$table))/sum(cm_ridge$table)

cm_lasso <- confusionMatrix(lasso_predict,test_set$activity)
#lasso cm
cm_lasso$table
#lasso error (misclassification %)
1-sum(diag(cm_lasso$table))/sum(cm_lasso$table)

cm_elastic <- confusionMatrix(elastic_predict,test_set$activity)
#elastic cm
cm_elastic$table
#elastic error (misclassification %)
1-sum(diag(cm_elastic$table))/sum(cm_elastic$table)

#picking the model
elastic_model$bestTune
elastic_best <- elastic_model$finalModel
coefficients <- coef(elastic_best,s=elastic_model$bestTune$lambda)
predictor <- data.frame(variables = names(coefficients[[1]][,1]),y1 = coefficients[[1]][,1],y2 = coefficients[[2]][,1],y3 = coefficients[[3]][,1],y4 = coefficients[[4]][,1],y5 = coefficients[[5]][,1],y6 = coefficients[[6]][,1])
head(predictor)

#plot decision tree
#we will now use ctree from the party package to view the decision tree
#we have used some pruning parameters to make the tree look pretty
# this is for an example purpose only and not actually used for modelling
tree <- ctree(activity~.,data = test_set,controls = ctree_control(mincriterion = 0.99,minsplit = 400))
plot(tree)

#Result
#validation set checks
elastic_predict_valid <- predict(elastic_model,data_valid)
#elastic accuracy 
mean(as.numeric(as.character((elastic_predict_valid))) == as.numeric(as.character(data_valid$activity)))
#create confusion matrix
cm_elastic_valid <- confusionMatrix(elastic_predict_valid,data_valid$activity)
#elastic cm
cm_elastic_valid$table
#ridge error (misclassification %)
1-sum(diag(cm_elastic_valid$table))/sum(cm_elastic_valid$table)



