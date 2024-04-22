\name{print.sbrl}
\alias{show.sbrl}
\alias{print.sbrl}
\title{
INTERPRETABLE VERSION OF A SBRL MODEL
}
\description{
This function prints an sbrl object. It is a method for the generic function print of class "sbrl".
}
\usage{
# S3 method for class 'sbrl'
# This complies with the form of the standard generic method print
\method{print}{sbrl}(x, useS4=FALSE, \dots)
\method{show}{sbrl}(x, useS4=FALSE, \dots)
}
\arguments{
  \item{x}{ A sbrl model returned from \code{\link{sbrl}} function }
  \item{useS4}{ An argument used to match showDefault function. Fixed as FALSE. }
  \item{\dots}{
    further arguments passed to or from other methods.
  }
}
\details{
	 This function is a method for the generic function print for class "sbrl". It can be invoked by calling print for an object of the appropriate class, or directly by calling print.sbrl regardless of the class of the object.
}
\value{No return value. This function prints out the logical structure of the sbrl model as a sequence of IF-THEN rules, identical to a decision list or one-sided decision tree.}
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

#clean up
rm(list = ls())
gc()
}
