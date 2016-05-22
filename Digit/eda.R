setwd("/Users/ewenwang/Dropbox/Data Science/Kaggle/Digit")

# ============================================================================
#### Load packages
require(ggplot2) # visualization
require(ggthemes)
require(proto)
require(readr)

# ============================================================================
#### load the data
train <- read.csv('train.csv', stringsAsFactors = F)
test  <- read.csv('test.csv', stringsAsFactors = F)

head(train[1:10])

# ============================================================================
#### Data Preprocessing
# Create a 28*28 matrix with pixel color values
m = matrix(unlist(train[10,-1]),nrow = 28,byrow = T)
# Plot that matrix
image(m,col=grey.colors(255))

rotate <- function(x) t(apply(x, 2, rev)) # reverses (rotates the matrix)

# Plot a bunch of images
par(mfrow=c(2,3))
lapply(1:6, 
       function(x) image(
         rotate(matrix(unlist(train[x,-1]),nrow = 28,byrow = T)),
         col=grey.colors(255),
         xlab=train[x,1]
       )
)
par(mfrow=c(1,1)) # set plot options back to default


# ============================================================================
#### Train, Predict and Save

require(h2o)

## start a local h2o cluster
localH2O = h2o.init(max_mem_size = '6g', # use 6GB of RAM of *GB available
                    nthreads = -1) # use all CPUs (8 on my personal computer :3)

## MNIST data as H2O
train[,1] = as.factor(train[,1]) # convert digit labels to factor for classification
train_h2o = as.h2o(train)

test_h2o = as.h2o(test)

## set timer
s <- proc.time()

## train model
model =
  h2o.deeplearning(x = 2:785,  # column numbers for predictors
                   y = 1,   # column number for label
                   training_frame = train_h2o, # data in H2O format
                   activation = "RectifierWithDropout", # algorithm
                   input_dropout_ratio = 0.2, # % of inputs dropout
                   hidden_dropout_ratios = c(0.5,0.5), # % for nodes dropout
                   balance_classes = TRUE, 
                   hidden = c(100,100), # two layers of 100 nodes
                   momentum_stable = 0.99,
                   nesterov_accelerated_gradient = T, # use it for speed
                   epochs = 15) # no. of epochs

## print confusion matrix
h2o.confusionMatrix(model)

## print time elapsed
s - proc.time()

## classify test set
h2o_y_test <- h2o.predict(model, test_h2o)

## convert H2O format into data frame and  save as csv
df_y_test = as.data.frame(h2o_y_test)
df_y_test = data.frame(ImageId = seq(1,length(df_y_test$predict)), Label = df_y_test$predict)
write.csv(df_y_test, file = "submission2_Ewen.csv", row.names=F)

## shut down virutal H2O cluster
h2o.shutdown(prompt = F)






