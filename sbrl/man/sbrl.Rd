\name{sbrl}
\alias{sbrl}
\title{
fit the scalable bayesian rule lists model
}
\description{
Fit the scalable bayesian rule lists model with given data and parameters. It generates a model that is a probabilistic classifier that optimizes the posterior of a Bayesian hierarchical model over pre-mined association rules.
}
\usage{
sbrl(tdata, iters=30000, pos_sign="1", 
 neg_sign="0", rule_minlen=1, rule_maxlen=1, 
 minsupport_pos=0.10, minsupport_neg=0.10, 
 lambda=10.0, eta=1.0, alpha=c(1,1), nchain=10)
}

\arguments{
  \item{tdata}{a dataframe, with a "label" column specifying the correct labels for each observation.}
  
  \item{iters}{the number of iterations for each MCMC chain.} 
  
  \item{pos_sign}{the sign for the positive labels in the "label" column.}
  
  \item{neg_sign}{the sign for the negative labels in the "label" column.}
  
  \item{rule_minlen}{the minimum number of cardinality for rules to be mined from the dataframe.}
  
  \item{rule_maxlen}{the maximum number of cardinality for rules to be mined from the dataframe.}
  
  \item{minsupport_pos}{a number between 0 and 1, for the minimum percentage support for the positive observations.}
  
  \item{minsupport_neg}{a number between 0 and 1, for the minimum percentage support for the negative observations.}
  
  \item{lambda}{a hyperparameter for the expected length of the rule list.}
  
  \item{eta}{a hyperparameter for the expected cardinality of the rules in the optimal rule list.}
  
  \item{alpha}{a prior pseudo-count for the positive and negative classes. fixed at 1's}
  
  \item{nchain}{an integer for the number of the chains that MCMC will be running.}
}
\value{
Return a list of :
	\item{rs}{a ruleset which contains the rule indices and their positive probabilities for the best rule list by training sbrl with the given data and parameters.}
	
	\item{rulenames}{a list of all the rule names mined with \code{arules}.}
	
	\item{featurenames}{a list of all the feature names.}
	
	\item{mat_feature_rule}{a binary matrix representing which features are included in which rules.}
}
\references{
  Hongyu Yang, Cynthia Rudin, Margo Seltzer (2017)
  \emph{Scalable Bayesian Rule Lists.}
  Proceedings of the 34th International Conference on Machine Learning, PMLR 70:3921-3930, 2017.

  Benjamin Letham, Cynthia Rudin, Tyler McCormick and David Madigan (2015)
  \emph{Building Interpretable Classifiers with Rules using Bayesian Analysis.}
  Annals of Applied Statistics, 2015.
}
\author{
Hongyu Yang, Morris Chen, Cynthia Rudin, Margo Seltzer
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
