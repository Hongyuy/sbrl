\name{predict.sbrl}
\alias{predict}
\alias{predict.sbrl}
\title{
PREDICT THE POSITIVE PROBABILITY FOR THE OBSERVATIONS
}
\description{
Returns a list of probabilities.
}
\usage{
\method{predict}{sbrl}(object, tdata, \dots)
}
\arguments{
  \item{object}{sbrl model returned from the \code{\link{sbrl}} function.
}
  \item{tdata}{test data
}
  \item{\dots}{
    further arguments passed to or from other methods.
  }
}
\value{
return a list containing 2 lists of probablities for the rule list, corresponding to probability being 0 and 1 for each observation. The two probabilities for each rule add up to 1, P(y=0 | rule r) + p(y=1 | rule r) = 1
}
\examples{
# Let us use the titactoe dataset
data(tictactoe)
for (name in names(tictactoe)) {tictactoe[name] <- as.factor(tictactoe[,name])}

# Train on two-thirds of the data
b = round(2*nrow(tictactoe)/3, digit=0)
data_train <- tictactoe[1:b, ]
# Test on the remaining one third of the data
data_test <- tictactoe[(b+1):nrow(tictactoe), ]
# data_train, data_test are dataframes with factor columns
# The class column is "label"

# Run the sbrl algorithm on the training set
  sbrl_model <- sbrl(data_train, iters=20000, pos_sign="1",
   neg_sign="0", rule_minlen=1, rule_maxlen=3, 
   minsupport_pos=0.10, minsupport_neg=0.10, 
   lambda=10.0, eta=1.0, nchain=25)
  print(sbrl_model)

# Make predictions on the test set
  yhat <- predict(sbrl_model, data_test)
# yhat will be a list of predicted negative and positive probabilities for the test data. 

#clean up
rm(list = ls())
gc()
}
