\name{tictactoe}
\alias{tictactoe}
\docType{data}
\title{
SHUFFLED TIC-TAC-TOE-ENDGAME DATASET
}
\description{
This is a shuffled version of the Tic-Tac-Toe Endgame Data Set on UCI Machine Learning Repository.
}
\usage{data("tictactoe")}
\format{
  A data frame with 958 observations on the following 10 variables.
  \describe{
    \item{\code{c1}}{a factor with levels \code{b}, \code{o}, \code{x}}
    \item{\code{c2}}{a factor with levels \code{b}, \code{o}, \code{x}}
    \item{\code{c3}}{a factor with levels \code{b}, \code{o}, \code{x}}
    \item{\code{c4}}{a factor with levels \code{b}, \code{o}, \code{x}}
    \item{\code{c5}}{a factor with levels \code{b}, \code{o}, \code{x}}
    \item{\code{c6}}{a factor with levels \code{b}, \code{o}, \code{x}}
    \item{\code{c7}}{a factor with levels \code{b}, \code{o}, \code{x}}
    \item{\code{c8}}{a factor with levels \code{b}, \code{o}, \code{x}}
    \item{\code{c9}}{a factor with levels \code{b}, \code{o}, \code{x}}
    \item{\code{label}}{an integer with values \code{0}, \code{1}}
  }
}
\details{
This database encodes the complete set of possible board configurations at the end of tic-tac-toe games, where "x" is assumed to have played first. The target concept is "win for x" (i.e., true when "x" has one of 8 possible ways to create a "three-in-a-row"). 
}
\source{
https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame
}
\examples{
data(tictactoe)
## maybe str(tictactoe) ; plot(tictactoe) ...
}
