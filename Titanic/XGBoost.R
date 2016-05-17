setwd("/Users/ewenwang/Dropbox/Data Science/Kaggle/Titanic")

# ============================================================================
## load data
data = read.csv("train.csv", header = T)
test = read.csv("test.csv", header = T)

# ============================================================================
#### preprocess data

# Grab title from passenger names
data$Title <- gsub('(.*, )|(\\..*)', '', data$Name)
test$Title <- gsub('(.*, )|(\\..*)', '', test$Name)

unique(data$Title)

# Titles with very low cell counts to be combined to "rare" level
rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')

# Also reassign mlle, ms, and mme accordingly
data$Title[data$Title == 'Mlle']        <- 'Miss' 
data$Title[data$Title == 'Ms']          <- 'Miss'
data$Title[data$Title == 'Mme']         <- 'Mrs' 
data$Title[data$Title %in% rare_title]  <- 'Rare Title'

test$Title[test$Title == 'Mlle']        <- 'Miss' 
test$Title[test$Title == 'Ms']          <- 'Miss'
test$Title[test$Title == 'Mme']         <- 'Mrs' 
test$Title[test$Title %in% rare_title]  <- 'Rare Title'

data$Title = factor(data$Title)
test$Title = factor(test$Title)

# Create a family size variable including the passenger themselves
data$Fsize = data$SibSp + data$Parch + 1
test$Fsize = test$SibSp + test$Parch + 1

y = data$Survived
X = data[,-c(1,2,4,9,11)]
PassengerID = test[,1]
test = test[,-c(1,3,8,10)]

X$Embarked[which(X$Embarked == "")] = NA

summary(y)

summary(X)
summary(test)

# ============================================================================
## missing data
require(mice)
md.pattern(X)
md.pattern(test)

require(VIM)
aggr_plot <- aggr(X, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, 
                  labels=names(data), cex.axis=.7, gap=3, 
                  ylab=c("Histogram of missing data","Pattern"))
aggr_plot <- aggr(test, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, 
                  labels=names(data), cex.axis=.7, gap=3, 
                  ylab=c("Histogram of missing data","Pattern"))

# imputation
require(randomForest)

y = as.factor(y)
Impt = rfImpute(X, y)

Impt$Embarked = factor(Impt$Embarked)

require(missForest)
test.Impt = missForest(test)
test.Impt = test.Impt$ximp

# ============================================================================
#### XGBoost
require(xgboost)
require(Ckmeans.1d.dp)

Impt$y = as.integer(Impt$y)
Impt$y = Impt$y - 1
Impt_xgb = xgb.DMatrix(data = data.matrix(Impt[,-1]), label = Impt$y)

## grid search depth
depthGrid = seq(1, 10, 1)
for (depth in depthGrid) {
  print("=====================================================")
  cat("depth: ", depth, "\n")
  xgb.cv(data = Impt_xgb, max.depth = depth, eta = 0.01, nthread = 2, 
         nround = 10, objective = "binary:logistic", 
         early.stop.round = 3, maximize = FALSE, nfold = 5)
}
# depth = 8

bst <- xgboost(data = Impt_xgb, max.depth = 8, eta = 0.01, nthread = 2, 
               nround = 10, objective = "binary:logistic", 
               early.stop.round = 3, maximize = FALSE)

xgb.plot.deepness(model = bst)
importance_matrix <- xgb.importance(colnames(X), model = bst)
xgb.plot.importance(importance_matrix)

# ============================================================================
# predict using the test set
test.Impt = xgb.DMatrix(data = data.matrix(test.Impt))

prediction <- predict(bst, test.Impt)
prediction = round(prediction)

# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
submission <- data.frame(PassengerID, Survived = prediction)

# Write the solution to file
write.csv(submission, file = 'Submission2_Ewen.csv', row.names = F)







