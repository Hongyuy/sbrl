#include <Rcpp.h>
#include <assert.h>
#if 0
#include <fstream>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif

#include "mytime.h"
#if 0
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_sf.h>
gsl_rng *RAND_GSL;

/* Convenient macros. */
#define RANDOM_RANGE(lo, hi) \
    (unsigned)(lo + (unsigned)((random() / (float)RAND_MAX) * (hi - lo + 1)))
#define DEFAULT_RULESET_SIZE  3
#define DEFAULT_RULE_CARDINALITY 3
#define MAX_RULE_CARDINALITY 3
#define NLABELS 2
#endif

// extern "C" {
// // https://stackoverflow.com/questions/1793800/can-i-redefine-a-c-macro-then-define-it-back
// #pragma push_macro("__cplusplus")
// #undef __cplusplus
#include "rule.h"
// #define __cplusplus
// #pragma pop_macro("__cplusplus")

PredModel train(Data &, int, int, const Params &);
int load_data(std::string &, std::string &, int &, int &, std::vector<Rule> &, std::vector<Rule> &);
int load_data2(Data &data, Rcpp::StringVector ruleNames, Rcpp::StringVector labelNames, Rcpp::IntegerMatrix ruleTruthTables, Rcpp::IntegerMatrix labelTruthTables)
{
        data.nrules = ruleNames.size();
        data.nsamples = ruleTruthTables.ncol();
        return 0;
}

// #if 0
// int debug;
// #endif
// }

    Rcpp::List _train(int initialization, int method, Rcpp::List paramList, Rcpp::CharacterVector dataFile, Rcpp::CharacterVector labelFile,
                      Rcpp::StringVector ruleNames, Rcpp::StringVector labelNames, Rcpp::IntegerMatrix ruleTruthTables, Rcpp::IntegerMatrix labelTruthTables) {
//        Rprintf("training!\n");

	Data	data, data2;
	int	ret;
	struct timeval tv_acc, tv_start, tv_end;
	std::string df = Rcpp::as<std::string>(dataFile[0]);
	std::string lf = Rcpp::as<std::string>(labelFile[0]);

        /*
	 * We treat the label file as a separate ruleset, since it has
 	 * a similar format.
         */
        INIT_TIME(tv_acc);
        START_TIME(tv_start);
        if ((ret = load_data(df, lf,
		data.nsamples, data.nrules, data.rules, data.labels)) != 0)
                return (ret);
        END_TIME(tv_start, tv_end, tv_acc);
        REPORT_TIME("Initialize time", "per rule", tv_end, data.nrules);

        if ((ret = load_data2(data2, ruleNames, labelNames, ruleTruthTables, labelTruthTables)) != 0)
                return (ret);
//#if 0
//        if (debug)
//                printf("%d rules %d samples\n\n", nrules, nsamples);
//
//        if (debug > 100)
//                rule_print_all(rules, nrules, nsamples);
//
//        if (debug > 100) {
//                printf("Labels for %d samples\n\n", nsamples);
//                rule_print_all(labels, nsamples, nsamples);
//        }
//#endif
 
	Params params;
        Rcpp::NumericVector nv;
        Rcpp::IntegerVector iv;
        nv = paramList[0];
        params.lambda = nv[0];
        nv = paramList[1];
        params.eta = nv[0];
        nv = paramList[2];
        params.threshold = nv[0];
        nv = paramList[3];
        params.alpha[0] = nv[0];
        params.alpha[1] = nv[1];
        iv = paramList[4];
        params.iters = iv[0];
        iv = paramList[5];
        params.nchain = iv[0];

        INIT_TIME(tv_acc);
        START_TIME(tv_start);
	PredModel pred_model_sbrl = train(data, initialization, method, params);
        END_TIME(tv_start, tv_end, tv_acc);
        REPORT_TIME("Time to train", "", tv_end, 1);

        Rcpp::IntegerVector id;
	for (int i=0; i<pred_model_sbrl.rs.n_rules; i++)
                id.push_back(pred_model_sbrl.rs.entries[i].rule_id);
	
        Rcpp::NumericVector prob;
	for (int i=0; i<pred_model_sbrl.rs.n_rules; i++)
                prob.push_back(pred_model_sbrl.theta[i]);

#if 0
        Rcpp::NumericVector ci_low;
	for (int i=0; i<pred_model_brl->rs->n_rules; i++)
        	ci_low.push_back(pred_model_brl->confIntervals->a);

        Rcpp::NumericVector ci_high;
	for (int i=0; i<pred_model_brl->rs->n_rules; i++)
        	ci_high.push_back(pred_model_brl->confIntervals->b);
#endif

        // Rcpp::DataFrame brl =  Rcpp::DataFrame::create(Rcpp::Named("clause")=clause, Rcpp::Named("prob")=prob, Rcpp::Named("ci_low")=ci_low, Rcpp::Named("ci_high")=ci_high);
        Rcpp::DataFrame rs =  Rcpp::DataFrame::create(Rcpp::Named("V1")=id, Rcpp::Named("V2")=prob);

        return(Rcpp::List::create(Rcpp::Named("rs")=rs));
    }

//using namespace Rcpp;

// Rcpp::List _train(int initialization, int method, Rcpp::List paramList, Rcpp::CharacterVector dataFile, Rcpp::CharacterVector labelFile)
// fastLR_
// Rcpp::List fastLR_(Rcpp::NumericMatrix x, Rcpp::NumericVector y, Rcpp::NumericVector start, double eps_f, double eps_g, int maxit);
RcppExport SEXP sbrl_train(SEXP initSEXP, SEXP methodSEXP, SEXP paramListSEXP, SEXP dataFileSEXP, SEXP labelFileSEXP,
                           SEXP ruleNamesSEXP, SEXP labelNamesSEXP, SEXP ruleTruthTablesSEXP, SEXP labelTruthTablesSEXP) {
    BEGIN_RCPP
    Rcpp::traits::input_parameter< int >::type init(initSEXP);
    Rcpp::traits::input_parameter< int >::type method(methodSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type params(paramListSEXP);
    Rcpp::traits::input_parameter< Rcpp::CharacterVector >::type dataFile(dataFileSEXP);
    Rcpp::traits::input_parameter< Rcpp::CharacterVector >::type labelFile(labelFileSEXP);
    Rcpp::traits::input_parameter< Rcpp::StringVector >::type ruleNames(ruleNamesSEXP);
    Rcpp::traits::input_parameter< Rcpp::StringVector >::type labelNames(labelNamesSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerMatrix >::type ruleTruthTables(ruleTruthTablesSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerMatrix >::type labelTruthTables(labelTruthTablesSEXP);
    //__result = Rcpp::wrap(_train(x, y, start, eps_f, eps_g, maxit));
    //return __result;
    return Rcpp::wrap(_train(init, method, params, dataFile, labelFile, ruleNames, labelNames, ruleTruthTables, labelTruthTables));
    END_RCPP
}
// Fortran code and Found no calls to: 'R_registerRoutines', 'R_useDynamicSymbols'
// https://stackoverflow.com/questions/43101032/fortran-code-and-found-no-calls-to-r-registerroutines-r-usedynamicsymbols
#include <R.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>
#include <Rcpp.h>

/* FIXME:
Check these declarations against the C/Fortran source code.
*/

// extern void F77_NAME(cf)(int *r, int *cd, double *loci);

static const R_FortranMethodDef FortranEntries[] = {
  {"sbrl_train", (DL_FUNC) &sbrl_train,  3},
  {NULL, NULL, 0}
};

void R_init_sbrl(DllInfo *dll)
{
  R_registerRoutines(dll, NULL, NULL, FortranEntries, NULL);
  R_useDynamicSymbols(dll, FALSE);
}

int
load_data(std::string &data_file, std::string &label_file,
    int &ret_samples, int &ret_nrules, std::vector<Rule> &rules, std::vector<Rule> &labels)
{
        int nlabels, ret, samples_chk;

        /* Load data. */
        if ((ret = rules_init(data_file, ret_nrules, ret_samples, rules, 1)) != 0)
                return (ret);

        /* Load labels. */
        if ((ret =
            rules_init(label_file, nlabels, samples_chk, labels, 0)) != 0) {
                // free (*rules);
                return (ret);
        }

        assert(nlabels == 2);
        assert(samples_chk == ret_samples);
        return (0);
}
