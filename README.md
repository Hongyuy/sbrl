# sbrl - the development repository for the R package [sbrl](https://cran.r-project.org/web/packages/sbrl/index.html)

The full details of the algorithm are in the paper:<br>
Hongyu Yang, Cynthia Rudin, Margo Seltzer (2017) <https://proceedings.mlr.press/v70/yang17h.html>

# ChangeLog
## current HEAD
* Further speedup (2x) the runtime of the core functions.
## v1.4
* Resolved [reported](https://stackoverflow.com/questions/51324870/memory-leak-and-c-wrapper) memory leak issues originated from the improper memory management in the C implementation.
## v1.3
* Refactored the C implementation to C++ implementation.
* Fixed various bugs and corner cases.
## v1.2
* Completed the R package and published it on [CRAN](https://github.com/cran/sbrl/commits/master/).
## v1.0
* The first implementation of the algorithm. It is now moved to the branch [master-old-c](https://github.com/Hongyuy/sbrl/tree/master-old-c)
