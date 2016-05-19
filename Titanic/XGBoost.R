setwd("/Users/ewenwang/Dropbox/Data Science/Kaggle/Titanic")

# ============================================================================
## load data
train = read.csv("train_clean.csv", header = T)
test = read.csv("test_clean.csv", header = T)
full = read.csv("full_clean.csv", header = T)

# ============================================================================
#### XGBoost
require(xgboost)
require(Ckmeans.1d.dp)

data_xgb = xgb.DMatrix(data = data.matrix(train[,-1]), label = train$Survived)

## grid search depth
depthGrid = seq(1, 10, 1)
roundGrid = seq(1, 10, 1)

set.seed(2016)
for (round in roundGrid) {
  for (depth in depthGrid) {
    print("=====================================================")
    cat("round: ", round, "depth: ", depth, "\n")
    xgb.cv(data = data_xgb, max.depth = depth, eta = 0.01, nthread = 2, 
           nround = round, objective = "binary:logistic", 
           early.stop.round = 3, maximize = FALSE, nfold = 5)
  }
}

# depth = 8; round = 5

bst <- xgboost(data = Impt_xgb, max.depth = 8, eta = 0.01, nthread = 2, 
               nround = 2, objective = "binary:logistic", 
               early.stop.round = 3, maximize = FALSE)

xgb.plot.deepness(model = bst)
importance_matrix <- xgb.importance(colnames(train[,-1]), model = bst)
xgb.plot.importance(importance_matrix)

# ============================================================================
#### feature selection

# remain features: Sex, Pclass, Fare, Surname, SibSp, Title
remain_var = c("Sex", "Pclass", "Fare", "Surname", "SibSp", "Title")

full_fc = data.frame()
full_fc <- full[,remain_var]

train_fc <- full_fc[1:891,]
test_fc <- full_fc[892:1309,]


# ============================================================================
#### refit the model

data_xgb_fc = xgb.DMatrix(data = data.matrix(train_fc), label = train$Survived)

## grid search depth
depthGrid = seq(1, 10, 1)
roundGrid = seq(1, 10, 1)

set.seed(2016)
for (round in roundGrid) {
  for (depth in depthGrid) {
    print("=====================================================")
    cat("round: ", round, "depth: ", depth, "\n")
    xgb.cv(data = data_xgb_fc, max.depth = depth, eta = 0.01, nthread = 2, 
           nround = round, objective = "binary:logistic", 
           early.stop.round = 3, maximize = FALSE, nfold = 5)
  }
}

# depth = 7; round = 2

bst_fc <- xgboost(data = data_xgb_fc, max.depth = 7, eta = 0.01, nthread = 2, 
                  nround = 5, objective = "binary:logistic", 
                  early.stop.round = 3, maximize = FALSE)

xgb.plot.deepness(model = bst_fc)
importance_matrix <- xgb.importance(colnames(train_fc), model = bst_fc)
xgb.plot.importance(importance_matrix)


# ============================================================================
# predict using the test set
data_xbg_test = xgb.DMatrix(data = data.matrix(test_fc))

prediction <- predict(bst_fc, data_xbg_test)
prediction = round(prediction)

test.orig = read.csv("test.csv", header = T)
PassengerID = test.orig$PassengerId

# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
submission <- data.frame(PassengerID, Survived = prediction)

# Write the solution to file
write.csv(submission, file = 'Submission3_Ewen.csv', row.names = F)







