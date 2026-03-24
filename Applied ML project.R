library(tidyverse)
library(dplyr)
library(mlr3verse)
library(mlr3learners)
set.seed(1234)

#import the datasets
train_df = read.csv("../Kaggle Project/train.csv")

test_df = read.csv("../Kaggle Project/test.csv")
test_df$Response = 0

head(train_df)
head(test_df)
str(train_df)

testId = test_df$Id
train_df$Id = test_df$Id = NULL

#missing values
na.omit(na_if(colSums(is.na(train_df)),0))

# create tasks
train_task = as_task_regr(train_df, target = "Response")
test_task = as_task_regr(test_df, target = "Response")

# one hot encoding
poe = po("encode", method = "one-hot")
goe = ppl("convert_types", "character", "factor") %>>% po("encode")
train_encode = goe$train(train_task)[[1]]$data()
train_enc = as_task_regr(train_encode, target = "Response")
test_encode = goe$train(test_task)[[1]]$data()
test_enc = as_task_regr(test_encode, target = "Response")

# create learner that imputes missing values
imp_xgboost = as_learner(po("imputemean") %>>%  lrn("regr.xgboost"))
imp_rpart = as_learner(po("imputemean") %>>% lrn("regr.rpart"))
imp_ranger = as_learner(po("imputemean") %>>% lrn("regr.ranger"))

# Create autotuner
afs_ranger = auto_fselector(
  fselector = fs("random_search"),
  learner = imp_ranger,
  resampling = rsmp("holdout"),
  measure = msr("regr.mse"),
  terminator = trm("evals", n_evals = 10)
)
afs_xg = auto_fselector(
  fselector = fs("random_search"),
  learner = imp_xgboost,
  resampling = rsmp("holdout"),
  measure = msr("regr.mse"),
  terminator = trm("evals", n_evals = 10)
)
afs_rpart = auto_fselector(
  fselector = fs("random_search"),
  learner = imp_rpart,
  resampling = rsmp("holdout"),
  measure = msr("regr.mse"),
  terminator = trm("evals", n_evals = 10)
)
grid = benchmark_grid(train_enc, list(afs_ranger, afs_xg, afs_rpart),
                      rsmp("cv", folds = 3))

bmr = benchmark(grid)$aggregate(msr("regr.mse"))
as.data.table(bmr)[, .(learner_id, regr.mse)]

#train the model
model = afs_xg$train(train_enc)

afs_ranger$fselect_result$features

pred = afs_xg$predict_newdata(test_df)
as.data.table(pred)
submission = data.frame(Id = testId)
submission$Response = as.integer(pred$response)
as.data.table(submission)
write.csv(submission, "Prudential_life_insurance_submission", row.names = FALSE)
