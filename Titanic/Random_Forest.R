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
#### Split data into training data and test data

## 75% of the sample size
smp_size <- floor(0.75 * nrow(Impt))

## set the seed to make your partition reproductible
set.seed(123)
train_ind <- sample(seq_len(nrow(Impt)), size = smp_size)

trn <- Impt[train_ind, ]
tst <- Impt[-train_ind, ]

# ============================================================================
#### random forest

## grid search
mtryGrid = c(1, 2, 3, 4)
OOB_rate = c()
for (mtry in mtryGrid) {
  RF = randomForest(y~., data = trn, mtry = mtry, ntree = 1000)
  OOB_rate[mtry] = mean(RF$err.rate[,"OOB"])
}
OOB_rate

mtry = 3
model_RF = randomForest(y~., data = trn, mtry = mtry, ntree = 1000)

mean(model_RF$err.rate[,"OOB"])
mean(model_RF$err.rate[,"0"])
mean(model_RF$err.rate[,"1"])

plot(model_RF)
legend('topright', colnames(model_RF$err.rate), col=1:3, fill=1:3)

## use all data to build the final model
model_RF_final = randomForest(y~., data = Impt, mtry = mtry, ntree = 1000)

mean(model_RF_final$err.rate[,"OOB"])
mean(model_RF_final$err.rate[,"0"])
mean(model_RF_final$err.rate[,"1"])

plot(model_RF_final)
legend('topright', colnames(model_RF_final$err.rate), col=1:3, fill=1:3)

# ============================================================================
# predict using the test set
test.Impt$Pclass = as.integer(test.Impt$Pclass)
test.Impt$SibSp = as.integer(test.Impt$SibSp)
test.Impt$Parch = as.integer(test.Impt$Parch)

prediction <- predict(model_RF_final, test.Impt)

# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
submission <- data.frame(PassengerID, Survived = prediction)

# Write the solution to file
write.csv(submission, file = 'Submission_Ewen.csv', row.names = F)


