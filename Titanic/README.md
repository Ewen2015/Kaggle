### Titanic

#### First Try

Method used: Random Forest (mtry = 3)

Predictors used: ``Pclass``, ``Sex``, ``Age``, ``SibSp``, ``Parch``, ``Fare``, ``Embarked``, ``Title``, ``Fsize``

Imputation: on train and test

R packages used: ``randomForest``, ``missForest``, ``VIM``, ``mice``

Kaggle score: 0.77512

#### Second Try

Method used: Boosting Tree (depth = 8)

Predictors used: ``Pclass``, ``Sex``, ``Age``, ``SibSp``, ``Parch``, ``Fare``, ``Embarked``, ``Title``, ``Fsize``

Imputation: on train and test

R packages used: ``xgboost``, ``randomForest``, ``missForest``, ``VIM``, ``mice``

Kaggle score: 0.78947
