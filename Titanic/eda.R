setwd("/Users/ewenwang/Dropbox/Data Science/Kaggle/Titanic")

# ============================================================================
#### Load packages
require(ggplot2) # visualization
require(ggthemes) # visualization
require(scales) # visualization
require(dplyr) # data manipulation
require(mice) # imputation
require(randomForest) # classification algorithm

# ============================================================================
#### load the data
train <- read.csv('train.csv', stringsAsFactors = F)
test  <- read.csv('test.csv', stringsAsFactors = F)

full  <- bind_rows(train, test) # bind training & test data
rownames(full) <- full$PassengerId

# check data
str(full)

# ============================================================================
#### Feature Engineering

## Survived, Sex, Age ============================
full$Survived = factor(full$Survived)
full$Sex = factor(full$Sex)
full$Age = as.integer(full$Age)

full$Embarked[which(full$Embarked == "")] = NA
full$Embarked = factor(full$Embarked)


## Name ==========================================
# Grab title from passenger names
full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)

# Show title counts by sex
table(full$Sex, full$Title)

# Titles with very low cell counts to be combined to "rare" level
rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')

# Also reassign mlle, ms, and mme accordingly
full$Title[full$Title == 'Mlle']        <- 'Miss' 
full$Title[full$Title == 'Ms']          <- 'Miss'
full$Title[full$Title == 'Mme']         <- 'Mrs' 
full$Title[full$Title %in% rare_title]  <- 'Rare Title'

# Show title counts by sex again
table(full$Sex, full$Title)

# Finally, grab surname from passenger name
full$Surname <- sapply(full$Name,  
                       function(x) strsplit(x, split = '[,.]')[[1]][1])

full$Title = factor(full$Title)
full$Surname = factor(full$Surname)


## family ==========================================
# Create a family size variable including the passenger themselves
full$Fsize <- full$SibSp + full$Parch + 1
full$Fsize = as.integer(full$Fsize)

# Create a family variable 
full$Family <- paste(full$Surname, full$Fsize, sep='_')

full$Family = factor(full$Family)

# Use ggplot2 to visualize the relationship between family size & survival
ggplot(full[1:891,], aes(x = Fsize, fill = factor(Survived))) +
  geom_bar(stat='count', position='dodge') +
  scale_x_continuous(breaks=c(1:11)) +
  labs(x = 'Family Size') +
  theme_few()

# Discretize family size
full$FsizeD[full$Fsize == 1] <- 'singleton'
full$FsizeD[full$Fsize < 5 & full$Fsize > 1] <- 'small'
full$FsizeD[full$Fsize > 4] <- 'large'

full$FsizeD = factor(full$FsizeD)

# Show family size by survival using a mosaic plot
mosaicplot(table(full$FsizeD, full$Survived), main='Family Size by Survival', 
           shade=TRUE)


## Cabin ============================================
# This variable appears to have a lot of missing values
full$Cabin[1:28]

# Create a Deck variable. Get passenger deck A - F:
full$Deck<-factor(sapply(full$Cabin, function(x) strsplit(x, NULL)[[1]][1]))

table(full$Survived, full$Deck)


# ============================================================================
#### Missingness

## check missing data
colSums(is.na(full[1:891,]))
colSums(is.na(full[-c(1:891),]))
# sapply(airquality, function(x) sum(is.na(x)))
# apply(is.na(airquality),2,sum)

which(is.na(full$Embarked)) # 62 830
which(is.na(full$Fare)) # 1044

require(mice)
md.pattern(full[1:891,])
md.pattern(full[-c(1:891),])

require(VIM)
aggr_plot <- aggr(full, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, 
                  labels=names(full), cex.axis=.7, gap=3, 
                  ylab=c("Histogram of missing data","Pattern"))
aggr_plot <- aggr(full[c(1:891),], col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, 
                  labels=names(full), cex.axis=.7, gap=3, 
                  ylab=c("Histogram of missing data","Pattern"))
aggr_plot <- aggr(full[-c(1:891),], col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, 
                  labels=names(full), cex.axis=.7, gap=3, 
                  ylab=c("Histogram of missing data","Pattern"))


## Sensible value imputation

full$Fare[c(62, 830)] # 80 80

full$Embarked[1044] # S
full$Pclass[1044] # 3
full$Sex[1044] # male

# Get rid of our missing passenger IDs
full.rmmv <- full %>%
  filter(PassengerId != 62 & PassengerId != 830 & PassengerId != 1044)

# Use ggplot2 to visualize embarkment, passenger class, & median fare
ggplot(full.rmmv, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +
  geom_boxplot() +
# geom_jitter(size = 0.01, alpha = 0.5) +
  geom_hline(aes(yintercept=80), colour='red', linetype='dashed', lwd=0.5) +
  scale_y_continuous(labels=dollar_format()) +
  theme_few()

ggplot(full.rmmv[full.rmmv$Pclass == '3' & full.rmmv$Embarked == 'S', ], 
       aes(x = Fare)) +
  geom_density(fill = '#99d6ff', alpha=0.4) + 
  geom_vline(aes(xintercept=median(Fare, na.rm=T)),
             colour='red', linetype='dashed', lwd=0.5) +
  scale_x_continuous(labels=dollar_format()) +
  theme_few()

est = median(full.rmmv[full.rmmv$Pclass == '3' & full.rmmv$Embarked == 'S', ]$Fare,
             na.rm = TRUE) # 8.05

# Since c(62, 830) fare was $80 for 1st class, they most likely embarked from 'C'
full$Embarked[c(62, 830)] <- 'C'
# Replace missing fare value with median (8.05) fare for class/embarkment
full$Fare[1044] <- est

# rerun the checking
colSums(is.na(full[1:891,]))
colSums(is.na(full[-c(1:891),]))

aggr_plot <- aggr(full[c(1:891),], col=c('navyblue','yellow'), numbers=TRUE, 
                  sortVars=TRUE, labels=names(full), cex.axis=.7, gap=3, 
                  ylab=c("Histogram of missing data","Pattern"))
aggr_plot <- aggr(full[-c(1:891),], col=c('navyblue','yellow'), numbers=TRUE, 
                  sortVars=TRUE, labels=names(full), cex.axis=.7, gap=3, 
                  ylab=c("Histogram of missing data","Pattern"))


## Predictive imputation

# Perform mice imputation, excluding certain less-than-useful variables:
mice_mod <- mice(full[, !names(full) %in% 
                        c('PassengerId','Name','Ticket','Cabin','Family',
                          'Surname','Survived')], method='rf')

# Save the complete output 
mice_output <- complete(mice_mod)

# Plot age distributions
par(mfrow=c(1,2))
hist(full$Age, freq=F, main='Age: Original Data', 
     col='darkgreen', ylim=c(0,0.04))
hist(mice_output$Age, freq=F, main='Age: MICE Output', 
     col='lightgreen', ylim=c(0,0.04))
par(mfrow=c(1,1))

# Replace Age variable from the mice model.
full$Age <- mice_output$Age

# Show new number of missing Age values
sum(is.na(full$Age))

# ============================================================================
#### Feature Engineering 2

# First we'll look at the relationship between age & survival
ggplot(full[1:891,], aes(Age, fill = factor(Survived))) + 
  geom_histogram() + 
  # I include Sex since we know (a priori) it's a significant predictor
  facet_grid(.~Sex) + 
  theme_few()

# Create the column child, and indicate whether child or adult
full$Child[full$Age < 18] <- 'Child'
full$Child[full$Age >= 18] <- 'Adult'

# Show counts
table(full$Child, full$Survived)

# Adding Mother variable
full$Mother <- 'Not Mother'
full$Mother[full$Sex == 'female' & full$Parch > 0 & full$Age > 18 & full$Title != 'Miss'] <- 'Mother'

# Show counts
table(full$Mother, full$Survived)

# Finish by factorizing our two new factor variables
full$Child  <- factor(full$Child)
full$Mother <- factor(full$Mother)

# remove Deck, Cabin, Name, and Ticket from data
rmv_vars = c("PassengerId", "Name", "Ticket", "Cabin", "Deck")

full[rmv_vars] <- lapply(full[rmv_vars], function(x) NULL)

md.pattern(full)


# ============================================================================
#### Split into training & test sets

# Split the data back into a train set and a test set
train <- full[1:891,]
test <- full[892:1309,]


# ============================================================================
#### Write clean data to file
write.csv(train, file = 'train_clean.csv', row.names = F)
write.csv(test, file = 'test_clean.csv', row.names = F)
write.csv(full, file = "full_clean.csv", row.names = F)






