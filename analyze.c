/*
 * Program to read in lines of rule/evaluation pairs where rule is a string
 * representing a rule and the evaluation is a vector of 1's and 0's indicating
 * whether the ith sample evaluates to true or false for the given rule.
 *
 * Once we have a collection of rules, we create random rulesets so that we
 * can apply basic transformations to those rule sets: add a rule, remove a
 * rule, swap the order of two rules.
 */

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include "mytime.h"
#include "rule.h"
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

//#define LAMBDA 20.0
//#define ETA 1.0
//#define THRESHOLD 0.5
//double ALPHA[2] = {1, 1};

void run_experiment(int, int, int, int, rule_t *);
ruleset_t* run_mcmc_single_core(int, int, int, int, rule_t *, rule_t *, params_t, double);
ruleset_t* run_simulated_annealing_single_core(int, int, int, int, rule_t *, rule_t *, params_t);
void ruleset_proposal(ruleset_t *, int, int *, int *, char *, double *);
void ruleset_assign(ruleset_t **, ruleset_t *);
double compute_log_posterior(ruleset_t *, rule_t *, int, rule_t *, params_t, int, int, double *);
void init_gsl_rand_gen();
int gen_poission(double);
void gsl_ran_poisson_test();
void sanity_check_with_python(int, int, int, int, rule_t *, rule_t *, params_t);
double* get_theta(ruleset_t*, rule_t *, rule_t *, params_t);
double get_accuracy(ruleset_t*, double *, rule_t *, rule_t *, params_t);
pred_model_t *train(data_t *, int, int, params_t); // data, initialization, method, prior
double* predict(data_t *, pred_model_t *, params_t);

int debug;

/*
 * Usage: analyze <file> -s <ruleset-size> -i <input operations> -S <seed>
 */
int
usage(void)
{
	(void)fprintf(stderr,
	    "Usage: analyze [-d level] [-s ruleset-size] %s\n",
	    "[-c cmdfile] [-i iterations] [-S seed]");
	return (-1);
}

int
main (int argc, char *argv[])
{
    int  len, rc;
	int iters, nrules, nsamples, nlabels, nsamples_dup, ntestsamples, ntestsamples_dup, nchains;
    rule_t *rules, *labels, *test_rules, *test_labels;
    params_t params = {9.0, 1.0, {1.0, 1.0}, 0.5, 1000, 2, 11};
    
    extern char *optarg;
    extern int optind, optopt, opterr, optreset;
    int ret, size = DEFAULT_RULESET_SIZE;
    char ch, *cmdfile = NULL, *infile;
    struct timeval tv_acc, tv_start, tv_end;

    debug = 0;
    iters = 2;
    nchains=11;
    srandom( time(0)+clock()+random() );
    while ((ch = getopt(argc, argv, "d:i:s:S:n:b:")) != EOF) {
        switch (ch) {
        case 'c':
            cmdfile = optarg;
            break;
        case 'd':
            debug = atoi(optarg);
            break;
        case 'i':
            params.iters = atoi(optarg);
            break;
        case 's':
            params.init_size = atoi(optarg);
            if ((ret = (params.init_size <= 1)) != 0)
                return (ret);
            break;
        case 'S':
            srandom((unsigned)(atoi(optarg)));
                break;
        case 'n':
            params.nchain = atoi(optarg);
                break;
        case 'b':
            params.LAMBDA = atof(optarg);
            break;
        case '?':
        default:
            return (usage());
        }
    }
    //printf("iter here = %d\n", iters);
    argc -= optind;
    argv += optind;
    
    if (argc == 0)
        return (usage());
    /* read in the training data-rule relations */
    infile = argv[0];

    INIT_TIME(tv_acc);
    START_TIME(tv_start);
    if ((ret = rules_init(infile, &nrules, &nsamples, &rules, 1)) != 0)
        return (ret);
    
    //rule_print_all(rules, nrules, nsamples);
//    for (int i=0; i<nrules; i++) if (rules[i].cardinality==1) {
//        printf("rule[%d] is %s\n", i, rules[i].features);
//    }
    
    if (debug > 0)
	    printf("\n%d rules %d samples\n\n", nrules, nsamples);

    if (debug > 100)
        rule_print_all(rules, nrules, nsamples);

    /* Read in the training data-label relations. */
    infile = argv[1];
    if ((ret = rules_init(infile, &nlabels, &nsamples_dup, &labels, 0)) != 0)
        return (ret);
    assert (nsamples == nsamples_dup);

    /* read in the test data-rule relations */
//    if (argc >= 4) {
//        infile = argv[2];
//        printf("infile here = %s\n", infile);
//        if ((ret = rules_init(infile, &nlabels, &ntestsamples, &test_rules, 1)) != 0)
//            return (ret);
//        printf("#test points = %d\n", ntestsamples);
//        printf("%d\n", count_ones_vector(test_rules[0].truthtable, test_rules[0].support));
//        /* read in the test data-label relations */
//        infile = argv[3];
//        printf("infile here = %s\n", infile);
//        if ((ret = rules_init(infile, &nlabels, &ntestsamples_dup, &test_labels, 0)) != 0)
//            return (ret);
//        printf("#test points = %d\n", ntestsamples_dup);
//        
//        rule_print_all(labels, nlabels, nsamples);
//        printf("\n%d labels %d samples\n\n", nlabels, nsamples);
//    }
    //rule_print_all(labels, nlabels, nsamples);
    //printf("\n%d labels %d samples\n\n", nlabels, nsamples);
    
	/*
	 * Add number of iterations for first parameter
	 */

//    int init_size = 2; //size;
//    ruleset_t *rs;
//    sanity_check_with_python(5, init_size, nsamples, nrules, rules, labels, params);
    
    data_t train_data, test_data;
    train_data.rules = rules;
    train_data.labels = labels;
    train_data.nrules = nrules;
    train_data.nsamples = nsamples;
    
//    rs = run_mcmc_single_core(iters, init_size, nsamples, nrules, rules, labels, params);
    pred_model_t *pred_model_brl;
    pred_model_brl = train(&train_data, 0, 0, params);
//    double *theta = NULL;
//    printf("accuracy = %.8f\n", get_predict_accuracy(rs, rules, labels, test_rules, test_labels, &theta, params) );
//    if (argc >= 4) {
//        test_data.rules = test_rules;
//        test_data.labels = test_labels;
//        test_data.nrules = nrules;
//        test_data.nsamples = ntestsamples;
//        double* probs = predict(&test_data, pred_model_brl, params);
//    }
    if (argc >= 3) {
        FILE *outfile;
        outfile = fopen (argv[2], "w+");
        for (int i=0; i<pred_model_brl->rs->n_rules; i++)
            fprintf(outfile, "%d,%.8f\n", pred_model_brl->rs->rules[i].rule_id, pred_model_brl->theta[i]);
    	fflush(outfile);
    }
    
    END_TIME(tv_start, tv_end, tv_acc);
    REPORT_TIME("analyze", "per rule", tv_acc, nrules);

    printf("Lambada = %.6f\n", params.LAMBDA);
    printf("Eta = %.6f\n", params.ETA);
    printf("Alpha[0] = %.6f\n", params.ALPHA[0]);
    printf("Alpha[1] = %.6f\n", params.ALPHA[1]);
    printf("nchain = %d\n", params.nchain);
    printf("Iterations = %d\n", params.iters);
    printf("Init_size = %d\n", params.init_size);

}

pred_model_t *
train(data_t *train_data, int initialization, int method, params_t params) {
    pred_model_t *pred_model = calloc(1, sizeof(pred_model_t));
    ruleset_t *rs, *rs_temp;
    double max_pos = -1e9, pos_temp, null_bound;
    rs = run_mcmc_single_core(params.iters, params.init_size, train_data->nsamples, train_data->nrules, train_data->rules, train_data->labels, params, max_pos);
    max_pos = compute_log_posterior(rs, train_data->rules, train_data->nrules, train_data->labels, params, 1, -1, &null_bound);
    
    for (int chain = 1; chain < params.nchain; chain++) {
        rs_temp = run_mcmc_single_core(params.iters, params.init_size, train_data->nsamples, train_data->nrules, train_data->rules, train_data->labels, params, max_pos);
        pos_temp = compute_log_posterior(rs_temp, train_data->rules, train_data->nrules, train_data->labels, params, 1, -1, &null_bound);
        if (pos_temp >= max_pos) {
            rs = rs_temp;
            max_pos = pos_temp;
        }
    }
    printf("\n\n/*----The best rule list out of %d MCMC chains is: */\n", params.nchain);
//    for (int i=0; i < rs->n_rules; i++) printf("rule[%d]_id = %d\n", i, rs->rules[i].rule_id);
    printf("max_log_posterior = %6f\n\n", max_pos);
    printf("max_log_posterior = %6f\n\n", compute_log_posterior(rs, train_data-> rules, train_data-> nrules,train_data-> labels, params, 1, -1, &null_bound));
    ruleset_print(rs, train_data->rules);
    
    double *theta = get_theta(rs, train_data->rules, train_data->labels, params);
    pred_model->rs = rs;
    pred_model->theta = theta;
    pred_model->rs_str = calloc(rs->n_rules, sizeof(char*));
    for (int i=0; i < rs->n_rules; i++) {
        pred_model->rs_str[i] = train_data->rules[rs->rules[i].rule_id].features;
        printf("%s\n", pred_model->rs_str[i]);
    }
    return pred_model;
}

double *
predict(data_t *test_data, pred_model_t *pred_model, params_t params) {
    double * prob = calloc(test_data->nsamples, sizeof(double));
    for (int j=0; j<pred_model->rs->n_rules; j++) {
        int cnt = 0;
        int rule_id = pred_model->rs->rules[j].rule_id;
        for (int i = 0; i < test_data->nsamples; i++) {
            if (prob[i]<1e-5 &&
	        rule_isset(test_data->rules[rule_id].truthtable, i)) {
			prob[i] = pred_model->theta[j];
			cnt ++;
            }
        }
//        printf(" rule %d captures %d samples, out of %d samples\n", rule_id, cnt, test_data->nsamples);
    }
    for (int i = 0; i < test_data->nsamples; i++) printf("%.6f\n", prob[i]);
//    printf("test accuracy = %.6f \n", get_accuracy(pred_model->rs, pred_model->theta, test_data->rules, test_data->labels, params));
    return prob;
}

double*
get_theta(ruleset_t *rs, rule_t *rules, rule_t *labels, params_t params) {
    /* calculate captured 0's and 1's */
    VECTOR v0;
    rule_vinit(rs->n_samples, &v0);
    double *theta = NULL;
    if (theta==NULL)
        theta = malloc(rs->n_rules*sizeof(double));
    for (int j=0; j < rs->n_rules; j++) {
        int n0, n1;
        rule_vand(v0, rs->rules[j].captures, labels[0].truthtable, rs->n_samples, &n0);
        n1 = rs->rules[j].ncaptured - n0;
        theta[j] = (n1+params.ALPHA[1])*1.0/(n1+n0+params.ALPHA[0]+params.ALPHA[1]);
        if (theta[j] >= params.THRESHOLD) {
            printf("n0=%d,  n1=%d, captured=%d, training accuracy = %.8f\n", n0, n1, rs->rules[j].ncaptured, n1*1.0/rs->rules[j].ncaptured);
        }
        else {
            printf("n0=%d,  n1=%d, captured=%d, training accuracy = %.8f\n", n0, n1, rs->rules[j].ncaptured, n0*1.0/rs->rules[j].ncaptured);
        }
        printf("theta[%d] = %.8f\n", j, theta[j]);
    }
    return theta;
}

double
get_accuracy(ruleset_t *rs, double *theta, rule_t *test_rules, rule_t *test_labels, params_t params) {
    VECTOR v0;
    rule_vinit(rs->n_samples, &v0);
    int *idarray = NULL;
    ruleset_t *rs_test;

    ruleset_backup(rs, &idarray);
    ruleset_init(rs->n_rules, test_rules[0].support, idarray, test_rules, &rs_test);
    ruleset_print(rs_test, test_rules);
//    for (int j=0; j< rs_test->n_rules; j++)
//        printf("theta[%d] = %d\n", j, theta[j]);
    rule_vinit(rs->n_samples, &v0);
    int nwrong = 0;
    for (int j=0; j< rs_test->n_rules; j++) {
        int n1_correct=0, n0_correct=0;
        if (theta[j] >= params.THRESHOLD) {
            rule_vand(v0, rs_test->rules[j].captures, test_labels[1].truthtable, rs_test->n_samples, &n1_correct);
            nwrong += abs(n1_correct - rs_test->rules[j].ncaptured);
        } else {
            rule_vand(v0, rs_test->rules[j].captures, test_labels[0].truthtable, rs_test->n_samples, &n0_correct);
            nwrong += abs(n0_correct - rs_test->rules[j].ncaptured);
        }
        printf("rules[%d] captures %d samples, correct n0=%d, n1=%d, test Probability=%.6f\n", j, rs_test->rules[j].ncaptured, n0_correct, n1_correct, (n0_correct+n1_correct)*1.0/rs_test->rules[j].ncaptured);
    }
    printf("ntotal = %d,  n0 = %d,  n1 = %d\n", rs_test->n_samples, test_labels[0].support, test_labels[1].support);
    printf("#incorrect predictions = %d,  #total predictions = %d\n", nwrong, rs_test->n_samples);
    return 1-1.0*nwrong/rs_test->n_samples; //1-1.0*(test_labels[1].support - cnt1_correct)/rs_test->n_samples;
}

int
create_random_ruleset(int size,
    int nsamples, int nrules, rule_t *rules, ruleset_t **rs)
{
	int i, j, *ids, next, ret;

	ids = calloc(size, sizeof(int));
	for (i = 0; i < (size - 1); i++) {
try_again:	next = RANDOM_RANGE(1, (nrules - 1));
		/* Check for duplicates. */
		for (j = 0; j < i; j++)
			if (ids[j] == next)
				goto try_again;
		ids[i] = next;
	}

	/* Always put rule 0 (the default) as the last rule. */
	ids[i] = 0;
    
	return(ruleset_init(size, nsamples, ids, rules, rs));
}

/*
 * Given a rule set, pick a random rule (not already in the set) and
 * add it at the ndx-th position.
 */
int
add_random_rule(rule_t *rules, int nrules, ruleset_t *rs, int ndx)
{
	int j, new_rule;

pickrule:
	new_rule = RANDOM_RANGE(1, (nrules-1));
	for (j = 0; j < rs->n_rules; j++)
		if (rs->rules[j].rule_id == new_rule)
			goto pickrule;
	if (debug > 10)
		printf("\nAdding rule: %d\n", new_rule);
	return(ruleset_add(rules, nrules, rs, new_rule, ndx));
}

/*
 * Generate a random ruleset and then do some number of adds, removes,
 * swaps, etc.
 */
void
run_experiment(int iters, int size, int nsamples, int nrules, rule_t *rules)
{
	int i, j, k, ret;
	ruleset_t *rs;
	struct timeval tv_acc, tv_start, tv_end;

	for (i = 0; i < iters; i++) {
		ret = create_random_ruleset(size, nsamples, nrules, rules, &rs);
		if (ret != 0)
			return;
		if (debug > 0) {
			printf("Initial ruleset\n");
			ruleset_print(rs, rules);
		}

		/* Now perform-(size-2) squared swaps */
		INIT_TIME(tv_acc);
		START_TIME(tv_start);
		for (j = 0; j < size; j++)
			for (k = 1; k < (size-1); k++) {
				if (debug > 0)
					printf("\nSwapping rules %d and %d\n",
					    rs->rules[k-1].rule_id,
					    rs->rules[k].rule_id);
				if (ruleset_swap(rs, k - 1, k, rules))
					return;
				if (debug > 0)
					ruleset_print(rs, rules);
			}
		END_TIME(tv_start, tv_end, tv_acc);
		REPORT_TIME("analyze", "per swap", tv_acc, ((size-1) * (size-1)));

		/*
		 * Now remove a rule from each position, replacing it
		 * with a random rule at the end.
		 */
		INIT_TIME(tv_acc);
		START_TIME(tv_start);
		for (j = 0; j < (size - 1); j++) {
			if (debug > 0)
				printf("\nDeleting rule %d\n", j);
			ruleset_delete(rules, nrules, rs, j);
			if (debug > 0) 
				ruleset_print(rs, rules);
			add_random_rule(rules, nrules, rs, j);
			if (debug > 0)
				ruleset_print(rs, rules);
		}
		END_TIME(tv_start, tv_end, tv_acc);
		REPORT_TIME("analyze", "per add/del", tv_acc, ((size-1) * 2));
	}

}

ruleset_t*
run_mcmc_single_core(int iters, int init_size, int nsamples, int nrules, rule_t *rules, rule_t *labels, params_t params, double v_star) {
    int i,j,t, ndx1, ndx2;
    char stepchar;
    ruleset_t *rs, *rs_proposal=NULL, *rs_temp=NULL;
    double jump_prob, log_post_rs=0.0, log_post_rs_proposal=0.0;
    int *rs_idarray=NULL, len, length4bound=0, nsuccessful_rej=0, n_add=0, n_delete=0, n_swap=0;
    double max_log_posterior = -1e9, prefix_bound = -1e10;
    /* initialize random number generator for some distrubitions */
    init_gsl_rand_gen();
    //gsl_ran_poisson_test();
    
//    for (int i=0; i<20; i++)
//        printf("%u , ", gen_poisson(7.2));
//    printf("\n");
    
    //initialize_ruleset(&rs, rules, nrules)
    
    /* initialize the ruleset */
    printf("%.8f, %.8f\n", prefix_bound, v_star);
    while (prefix_bound < v_star) {
        create_random_ruleset(init_size, nsamples, nrules, rules, &rs);
//        ruleset_print(rs, rules);
        log_post_rs = compute_log_posterior(rs, rules, nrules, labels, params, 0, 0, &prefix_bound);
    }
    ruleset_backup(rs, &rs_idarray);
    max_log_posterior = log_post_rs;
    len = rs->n_rules;
//
//    for (int i=0; i<10; i++) {
//        ruleset_proposal(rs, nrules, &ndx1, &ndx2, &stepchar, &jump_prob);
//        printf("\n%d, %d, %d, %c, %f\n", nrules, ndx1, ndx2, stepchar, log(jump_prob));
//    }
    
    //printf("%d\n", cou)
    ruleset_assign(&rs_proposal, rs); // rs_proposel <-- rs
//    printf("\n*****************************************\n");
//    printf("\n %p %p %d\n", rs, rs_proposal, rs_proposal==NULL);
    
//    ruleset_assign(&rs_proposal, rs);  // rs_proposel <-- rs
//    ruleset_print(rs_proposal, rules);
//    printf("\n %p %p \n", rs, rs_proposal);
    printf("iters = %d", iters);
    for (int i=0; i<iters; i++) {
        
        
        ruleset_proposal(rs, nrules, &ndx1, &ndx2, &stepchar, &jump_prob);
//        printf("\nnrules=%d, ndx1=%d, ndx2=%d, action=%c, relativeProbability=%f\n", nrules, ndx1, ndx2, stepchar, log(jump_prob));
//        printf("%d rules currently in the ruleset, they are:\n", rs->n_rules);
//        for (int j=0; j<rs->n_rules; j++) printf("%u ", rs->rules[j].rule_id); printf("\n");
        
        ruleset_assign(&rs_proposal, rs); // rs_proposel <-- rs
        
        switch (stepchar) {
            case 'A':
                ruleset_add(rules, nrules, rs_proposal, ndx1, ndx2); //add ndx1 rule to ndx2
                length4bound = ndx2;
                n_add ++;
                break;
            case 'D':
                ruleset_delete(rules, nrules, rs_proposal, ndx1);
                length4bound = ndx1;
                n_delete ++;
                break;
            case 'S':
                ruleset_swap_any(rs_proposal, ndx1, ndx2, rules);
                length4bound = ndx1;
                n_swap ++;
                break;
            default:
                break;
        }
////        if (stepchar!='S')
//        int cnt=0;
//        for (int j=0; j<rs_proposal->n_rules; j++) {
//            printf("%d\n", rs_proposal->rules[j].ncaptured);
//            cnt += rs_proposal->rules[j].ncaptured;
//        }
//            printf("############Been here! %d\n", cnt);
//        for (int j=0; j<rs_proposal->n_rules; j++) printf("%u ", rs_proposal->rules[j].rule_id);
//        printf("\n");
        
        log_post_rs_proposal = compute_log_posterior(rs_proposal, rules, nrules, labels, params, 0, length4bound, &prefix_bound);
//        printf("%f\n", jump_prob);
        if (prefix_bound < max_log_posterior) nsuccessful_rej++;
        if (prefix_bound > max_log_posterior &&
            log((random() / (float)RAND_MAX)) < log_post_rs_proposal-log_post_rs+log(jump_prob)) {
            free(rs);
            rs = rs_proposal;
            log_post_rs = log_post_rs_proposal;
            rs_proposal = NULL;
            if (log_post_rs>max_log_posterior) {
                ruleset_backup(rs, &rs_idarray);
		max_log_posterior = log_post_rs;
                len = rs->n_rules;
            }
        }
    }
    /* regenerate the best rule list */
    printf("\n\n/*----The best rule list is: */\n");
    printf("#rejections = %d\n", nsuccessful_rej);
    printf("#add, #delete, #swap = %d, %d, %d\n", n_add, n_delete, n_swap);
    ruleset_init(len, nsamples, rs_idarray, rules, &rs);
//    for (int i=0; i < len; i++) printf("rule[%d]_id = %d\n", i, rs_idarray[i]);
    printf("max_log_posterior = %6f\n\n", max_log_posterior);
    printf("max_log_posterior = %6f\n\n", compute_log_posterior(rs, rules, nrules, labels, params, 1, -1, &prefix_bound));
    ruleset_print(rs, rules);
    
    ruleset_init(len, nsamples, rs_idarray, rules, &rs);
    return rs;
}

ruleset_t*
run_simulated_annealing_single_core(int iters, int init_size, int nsamples, int nrules, rule_t *rules, rule_t *labels, params_t params) {
    int i,j,t, ndx1, ndx2;
    char stepchar;
    ruleset_t *rs, *rs_proposal=NULL, *rs_temp=NULL;
    double jump_prob, log_post_rs=0.0, log_post_rs_proposal=0.0;
    int *rs_idarray=NULL, len, length4bound;
    double max_log_posterior = -1e9, prefix_bound = 0.0;
    /* initialize random number generator for some distrubitions */
    init_gsl_rand_gen();
    
    /* initialize the ruleset */
    create_random_ruleset(init_size, nsamples, nrules, rules, &rs);
    log_post_rs = compute_log_posterior(rs, rules, nrules, labels, params, 0, -1, &prefix_bound);
    ruleset_backup(rs, &rs_idarray);
    max_log_posterior = log_post_rs;
    len = rs->n_rules;
    
    ruleset_assign(&rs_proposal, rs); // rs_proposel <-- rs
    
    /* pre-compute the cooling schedule*/
    double T[100000], tmp[50];
    int ntimepoints = 0;
    tmp[0] = 1;
    for (int i=1; i < 28; i++) {
        tmp[i] = tmp[i-1] + exp(0.25*(i+1));
        for (int j = (int)tmp[i-1]; j < (int)tmp[i]; j++)
            T[ntimepoints++] = 1.0/(i+1);
    }
    int itersPerStep = 200;
    printf("itersPerStep = %d,    #timepoints = %d\n", itersPerStep, ntimepoints);
    for (int k=0; k<ntimepoints; k++) {
        double tk = T[k];
        for (int iter=0; iter < itersPerStep; iter++) {
            ruleset_proposal(rs, nrules, &ndx1, &ndx2, &stepchar, &jump_prob);
            ruleset_assign(&rs_proposal, rs); // rs_proposel <-- rs
            
            switch (stepchar) {
                case 'A':
                    ruleset_add(rules, nrules, rs_proposal, ndx1, ndx2); //add ndx1 rule to ndx2
                    length4bound = ndx2;
                    break;
                case 'D':
                    ruleset_delete(rules, nrules, rs_proposal, ndx1);
                    length4bound = ndx1;
                    break;
                case 'S':
                    ruleset_swap_any(rs_proposal, ndx1, ndx2, rules);
                    length4bound = ndx1;
                    break;
                default:
                    break;
            }
            
            log_post_rs_proposal = compute_log_posterior(rs_proposal, rules, nrules, labels, params, 0, length4bound, &prefix_bound);
//            printf("%f\n", jump_prob);
//            if (log((random() / (float)RAND_MAX)) < log_post_rs_proposal-log_post_rs+log(jump_prob)) {
            if (prefix_bound > max_log_posterior &&
                ( log_post_rs_proposal > log_post_rs || log((random() / (float)RAND_MAX)) < (log_post_rs_proposal-log_post_rs)/tk) ) {
                free(rs);
                rs = rs_proposal;
                log_post_rs = log_post_rs_proposal;
                rs_proposal = NULL;
                if (log_post_rs>max_log_posterior) {
                    ruleset_backup(rs, &rs_idarray);
		    max_log_posterior = log_post_rs;
                    len = rs->n_rules;
                }
            }
        }
    }
    /* regenerate the best rule list */
    printf("\n\n/*----The best rule list is: */\n");
    ruleset_init(len, nsamples, rs_idarray, rules, &rs);
//    for (int i=0; i < len; i++) printf("rule[%d]_id = %d\n", i, rs_idarray[i]);
    printf("max_log_posterior = %6f\n\n", max_log_posterior);
    printf("max_log_posterior = %6f\n\n", compute_log_posterior(rs, rules, nrules, labels, params, 1, -1, &prefix_bound));
    ruleset_print(rs, rules);
    
    ruleset_init(len, nsamples, rs_idarray, rules, &rs);
    return rs;
}

double compute_log_posterior(ruleset_t *rs, rule_t *rules, int nrules, rule_t *labels, params_t params, int ifPrint, int length4bound, double *prefix_bound) {
    double log_prior = 0.0, log_likelihood = 0.0, norm_constant;
    static double eta_norm = 0;
    static double *log_lambda_pmf=NULL, *log_eta_pmf=NULL;
    int i,j,k,li;
    /* prior pre-calculation */
    if (log_lambda_pmf == NULL) {
        log_lambda_pmf = malloc(nrules*sizeof(double));
        log_eta_pmf = malloc((1+MAX_RULE_CARDINALITY)*sizeof(double));
        for (int i=0; i < nrules; i++) {
            log_lambda_pmf[i] = log(gsl_ran_poisson_pdf(i, params.LAMBDA));
//            printf("log_lambda_pmf[ %d ] = %6f\n", i, log_lambda_pmf[i]);
        }
        for (int i=0; i <= MAX_RULE_CARDINALITY; i++) {
            log_eta_pmf[i] = log(gsl_ran_poisson_pdf(i, params.ETA));
//            printf("log_eta_pmf[ %d ] = %6f\n", i, log_eta_pmf[i]);
        }
        /* 
         for simplicity, assume that all the cardinalities <= MAX_RULE_CARDINALITY 
         appear in the mined rules
         */
        eta_norm = gsl_cdf_poisson_P(MAX_RULE_CARDINALITY, params.ETA) - gsl_ran_poisson_pdf(0, params.ETA);
//        printf("eta_norm(Beta_Z) = %6f\n", eta_norm);
    }
    /* calculate log_prior */
    int card_count[1 + MAX_RULE_CARDINALITY];
    for (i=0; i <= MAX_RULE_CARDINALITY; i++) card_count[i] = 0;
    int maxcard = 0;
    
    for (i=0; i < nrules; i++){
        card_count[rules[i].cardinality]++;
        if (rules[i].cardinality > maxcard)
            maxcard = rules[i].cardinality;
    }
    
//    for (i=0; i<=MAX_RULE_CARDINALITY; i++)
//        printf("there are %d rules with cardinality %d\n", card_count[i], i);
    /*
     For simplicity, this was not used in BRL_code.py,
     eta_norm was pre-calculated in the prior pre-calculation section
     */
    //eta_norm = gsl_cdf_poisson_P(maxcard, ETA) - gsl_ran_poisson_pdf(0, ETA);
    norm_constant = eta_norm;
    log_prior += log_lambda_pmf[rs->n_rules-1];
    double prefix_prior = 0.0; // added for prefix_bound
    if (rs->n_rules-1 > params.LAMBDA)
        prefix_prior += log_lambda_pmf[rs->n_rules - 1];
    else
        prefix_prior += log_lambda_pmf[(int)params.LAMBDA];
    for (i=0; i < rs->n_rules-1; i++){ //don't compute the last rule(default rule).
        li = rules[rs->rules[i].rule_id].cardinality;
        if (log(norm_constant) != log(norm_constant)) printf("\n NAN here log(eta_norm) at i= %d \t eta_norm = %6f",i, eta_norm);
        log_prior += log_eta_pmf[li] - log(norm_constant);
        if (log_prior != log_prior) printf("\n NAN here at i= %d, aa ",i);
        log_prior += -log(card_count[li]);
        if (log_prior != log_prior) printf("\n NAN here at i= %d, bb ",i);
        if (j <= length4bound)
            prefix_prior += log_eta_pmf[li] - log(norm_constant) - log(card_count[li]); // added for prefix_boud
        card_count[li]--;
        if (card_count[li] == 0)
            norm_constant -= exp(log_eta_pmf[li]);
    }
    /* calculate log_likelihood */
    VECTOR v0;
    rule_vinit(rs->n_samples, &v0);
    double prefix_log_likelihood = 0.0;
    int left0 = labels[0].support, left1 = labels[1].support;
    for (int j=0; j < rs->n_rules; j++) {
        int n0, n1;
        rule_vand(v0, rs->rules[j].captures, labels[0].truthtable, rs->n_samples, &n0);
//        rule_vand(v0, v0, labels[0].truthtable, rs->n_samples, &n0);
        n1 = rs->rules[j].ncaptured - n0;
        log_likelihood += gsl_sf_lngamma(n0+params.ALPHA[0]) + gsl_sf_lngamma(n1+params.ALPHA[1]) - gsl_sf_lngamma(n0+n1+params.ALPHA[0]+params.ALPHA[1]);
        /*
         added for prefix_bound
         */
        left0 -= n0;
        left1 -= n1;
        if (j <= length4bound) {
            prefix_log_likelihood += gsl_sf_lngamma(n0+1) + gsl_sf_lngamma(n1+1) - gsl_sf_lngamma(n0+n1+2);
            if ( j == length4bound ){
                prefix_log_likelihood += gsl_sf_lngamma(1) + gsl_sf_lngamma(left0+1) - gsl_sf_lngamma(left0+2)
                                        + gsl_sf_lngamma(1) + gsl_sf_lngamma(left1+1) - gsl_sf_lngamma(left1+2);
            }
        }
    }
    *prefix_bound = prefix_prior + prefix_log_likelihood;
    if (ifPrint) printf("log_prior = %6f \t log_likelihood = %6f \n", log_prior, log_likelihood);
    return log_prior + log_likelihood;
}

int
ruleset_swap_any(ruleset_t *rs, int i, int j, rule_t *rules)
{
    int temp, cnt, ndx, nset, ret;
    VECTOR caught;
    
    assert(i <= rs->n_rules);
    assert(j <= rs->n_rules);
    if (i>j) { temp=i; i=j; j=temp; }
    assert(i <= j);
    /*
     * Compute newly caught in two steps: first compute everything
     * caught in rules i to j, then compute everything from scratch
     * for rules between rule i and rule j, both inclusive.
     */
    if ((ret = rule_vinit(rs->n_samples, &caught)) != 0)
        return (ret);
    for (int k=i; k<=j; k++)
        rule_vor(caught, caught, rs->rules[k].captures, rs->n_samples, &cnt);
    
    //printf("cnt = %d\n", cnt);
    temp = rs->rules[i].rule_id;
    rs->rules[i].rule_id = rs->rules[j].rule_id;
    rs->rules[j].rule_id = temp;
    
    for (int k=i; k<=j; k++) {
        rule_vand(rs->rules[k].captures, caught, rules[rs->rules[k].rule_id].truthtable, rs->n_samples, &rs->rules[k].ncaptured);
        rule_vandnot(caught, caught, rs->rules[k].captures, rs->n_samples, &cnt);
    //    printf("cnt = %d, captured by this rule = %d\n", cnt, rs->rules[k].ncaptured);
    }
    //printf("cnt = %d\n", cnt);
    assert(cnt == 0);
    
#ifdef GMP
    mpz_clear(caught);
#else
    free(caught);
#endif
    return (0);
}


void
ruleset_proposal(ruleset_t *rs, int nrules_mined, int *ndx1, int *ndx2, char *stepchar, double *jumpRatio) {
    static double MOVEPROBS[15] = {
        0.0, 1.0, 0.0,
        0.0, 0.5, 0.5,
        0.5, 0.0, 0.5,
        1.0/3.0, 1.0/3.0, 1.0/3.0,
        1.0/3.0, 1.0/3.0, 1.0/3.0
    };
    static double JUMPRATIOS[15] = {
        0.0, 0.5, 0.0,
        0.0, 2.0/3.0, 2.0,
        1.0, 0.0, 2.0/3.0,
        1.0, 1.5, 1.0,
        1.0, 1.0, 1.0
    };
    double moveProbs[3], jumpRatios[3];
//    double moveProbDefault[3] = {1.0/3.0, 1.0/3.0, 1.0/3.0};
    int offset = 0;
    if (rs->n_rules == 1){
        offset = 0;
    } else if (rs->n_rules == 2){
        offset = 3;
    } else if (rs->n_rules == nrules_mined-1){
        offset = 6;
    } else if (rs->n_rules == nrules_mined-2){
        offset = 9;
    } else {
        offset = 12;
    }
    memcpy(moveProbs, MOVEPROBS+offset, 3*sizeof(double));
    memcpy(jumpRatios, JUMPRATIOS+offset, 3*sizeof(double));
    
    double u = ((double) rand()) / (RAND_MAX);
    int index1, index2;
    if (u < moveProbs[0]){
        /* swap rules */
        index1 = rand() % (rs->n_rules-1); // can't swap with the default rule
        do {
            index2 = rand() % (rs->n_rules-1);
        }
        while (index2==index1);
        *jumpRatio = jumpRatios[0];
        *stepchar = 'S';
    } else if (u < moveProbs[0]+moveProbs[1]) {
        /* add a new rule */
        index1 = rs->n_rules + 1 + rand() % (nrules_mined-rs->n_rules);
        int *allrules = calloc(nrules_mined, sizeof(int));
        for (int i=0; i < rs->n_rules; i++) allrules[rs->rules[i].rule_id] = -1;
        int cnt=0;
        for (int i=0; i < nrules_mined; i++)
            if (allrules[i] != -1)
                allrules[cnt++] = i;
        index1 = allrules[rand() % cnt];
        free(allrules);
        index2 = rand() % rs->n_rules; // can add a rule at the default rule position
        *jumpRatio = jumpRatios[1] * (nrules_mined - 1 - rs->n_rules);
        *stepchar = 'A';
    } else if (u < moveProbs[0]+moveProbs[1]+moveProbs[2]){
        /* delete an existing rule */
        index1 = rand() % (rs->n_rules-1); //cannot delete the default rule
        index2 = 0;// index2 doesn't matter in this case
        *jumpRatio = jumpRatios[2] * (nrules_mined - rs->n_rules);
        *stepchar = 'D';
    } else{
        //should raise exception here.
    }
    *ndx1 = index1;
    *ndx2 = index2;
}

void
ruleset_assign(ruleset_t **ret_dest, ruleset_t *src) {
    ruleset_t *dest = *ret_dest;
    if (dest != NULL){
//        printf("wrong here\n");
//        realloc(dest, sizeof(ruleset_t) + (src->n_rules+1) * sizeof(ruleset_entry_t));
        free(dest);
    }
    dest = malloc(sizeof(ruleset_t) + (src->n_rules+1) * sizeof(ruleset_entry_t));
    dest->n_alloc = src->n_rules + 1;
    dest->n_rules = src->n_rules;
    dest->n_samples = src->n_samples;
    
//    printf("\n%d, %p\n", dest==NULL, dest);
   
    for (int i=0; i<src->n_rules; i++){
        dest->rules[i].rule_id = src->rules[i].rule_id;
        dest->rules[i].ncaptured = src->rules[i].ncaptured;
        rule_vinit(src->n_samples, &(dest->rules[i].captures));
//          this line is wrong below
//        VECTOR_ASSIGN(dest->rules[i].captures, src->rules[i].captures);
        rule_copy(dest->rules[i].captures, src->rules[i].captures, src->n_samples);
    }
    /* initialize the extra assigned space */
    rule_vinit(src->n_samples, &(dest->rules[src->n_rules].captures));
    *ret_dest = dest;
}


void
init_gsl_rand_gen() {
    gsl_rng_env_setup();
    RAND_GSL = gsl_rng_alloc(gsl_rng_default);
}

int
gen_poisson(double mu) {
    return (int) gsl_ran_poisson(RAND_GSL, mu);
}

double
gen_poission_pdf(int k, double mu) {
    return gsl_ran_poisson_pdf(k, mu);
}

double
gen_gamma_pdf(double x, double a, double b) {
    return gsl_ran_gamma_pdf(x, a, b);
}

void
gsl_ran_poisson_test() {
    unsigned int k1 = gsl_ran_poisson(RAND_GSL, 5);
    unsigned int k2 = gsl_ran_poisson(RAND_GSL, 5);
    printf("k1 = %u , k2 = %u\n", k1, k2);
    
    const int nrolls = 10000; // number of experiments
    const int nstars = 100;   // maximum number of stars to distribute
    
    int p[10]={};
    for (int i=0; i<nrolls; ++i) {
        unsigned int number = gsl_ran_poisson(RAND_GSL, 4.1);
        if (number<10) ++p[number];
    }
    
    printf("poisson_distribution (mean=4.1):\n" );
    for (int i=0; i<10; ++i) {
        printf("%d, : ", i);
        for (int j=0; j< p[i]*nstars/nrolls; j++)
            printf("*");
        printf("\n");
    }
}
/*
 randomly generate @iters rulesets and calculate
 the log_prior and log_posterior probability
 from C and Python separately. Output to screen
 */
void
sanity_check_with_python(int iters, int init_size, int nsamples, int nrules, rule_t *rules, rule_t *labels, params_t params) {
    int i,j,t, rand_size;
    double log_post_rs=0.0, null_bound;
    const char *pythonfile = "/Users/hongyuy/Desktop/code/rulelib-fork/rulelib-fork/rulelib-fork/cross_check.py";
    const char *basicoutfile = "cross_check_list.csv";
    char outfile[50];
    FILE *fout;
    ruleset_t *rs;
    int ret;
    
    snprintf(outfile, sizeof(outfile), "cross_check_list.csv");
             
    fout = fopen(outfile, "w");
    for (i = 0; i<iters; i++) {
        rand_size = RANDOM_RANGE(1, nrules);
        printf("ruleset[ %d ], size = %d:\n", i, rand_size);
        create_random_ruleset(rand_size, nsamples, nrules, rules, &rs);
        log_post_rs = compute_log_posterior(rs, rules, nrules, labels, params, 0, -1, &null_bound);
        for (j=0; j<rand_size-1; j++)
            fprintf(fout, "%d,", rs->rules[j].rule_id);
        fprintf(fout, "%d\n", rs->rules[rand_size-1].rule_id);
    }
    fclose(fout);
    
    char buffer[200];
    int cx;
    cx = snprintf(buffer, sizeof(buffer), "python %s", pythonfile);
    snprintf(buffer+cx, 50-cx, " %s", outfile);
    ret = system(buffer);
    free(rs);
    // option: generate outputs from C code and Python code, then diff.
}
