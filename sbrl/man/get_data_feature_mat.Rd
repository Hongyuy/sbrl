\name{get_data_feature_mat}
\alias{get_data_feature_mat}
\title{
	GET BINARY MATRIX REPRESENTATION OF THE DATA-FEATURE RELAITONSHIP
}
\description{
	Given some features in the form "feature1=x1", "feature2=x2"..., this function will generate a matrix representation of which data are captured by which features. 
}
\usage{
get_data_feature_mat(data, featurenames)
}
\arguments{
  \item{data}{ a data.frame representing the observations.
}
  \item{featurenames}{
a character vector representing the features in the form: "feature1=x1", "feature2=x2"...
}
}
\value{
a binary matrix of size #observations-by-#featurenames
}
\author{
Hongyu Yang, Morris Chen, Cynthia Rudin, Margo Seltzer
}

\examples{
data(tictactoe)
featurenames <- c("c1=b", "c1=o", "c1=x")
get_data_feature_mat(tictactoe, featurenames)
#it will generate a binary matrix representing which observations are captured by which features.
}
