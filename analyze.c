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
#define DEFAULT_RULESET_SIZE  4
#define DEFAULT_RULE_CARDINALITY 3
#define MAX_RULE_CARDINALITY 10
#define NLABELS 2

#define gen_poisson(MU) (int)gsl_ran_poisson(RAND_GSL, MU)
#define gen_poission_pdf(K, MU) gsl_ran_poisson_pdf(K, MU)
#define gen_gamma_pdf(X, A, B) gsl_ran_gamma_pdf(X, A, B)

#define LAMBDA 3.0
#define ETA 1.0
double ALPHA[2] = {1, 1};
double compute_log_posterior(ruleset_t *, rule_t *, int, rule_t *);

void run_experiment(int, int, int, int, rule_t *);
void run_mcmc(int, int, int, int, rule_t *, rule_t *);
void ruleset_proposal(ruleset_t *, int, int *, int *, char *, double *);

int debug;

int
usage(void)
{
	(void)fprintf(stderr, "Usage: analyze [-d] [-s ruleset-size] %s %s\n",
	    "[-c cmdfile] [-i iterations] [-t test] [-S seed]",
	    "data.out [data.label]");
	return (-1);
}

#define	TEST_TEST	1
#define	TEST_MCMC	2

int
main (int argc, char *argv[])
{
	extern char *optarg;
	extern int optind, optopt, opterr, optreset;
	int ret, size = DEFAULT_RULESET_SIZE;
	int iters, nrules, nsamples, nlabels, nsamples_chk, tnum;
	char ch, *cmdfile = NULL, *infile;
	rule_t *rules, *labels;
	struct timeval tv_acc, tv_start, tv_end;

	debug = 0;
	iters = 10;
	tnum = TEST_TEST;
	while ((ch = getopt(argc, argv, "di:s:S:t:")) != EOF)
		switch (ch) {
		case 'c':
			cmdfile = optarg;
			break;
		case 'd':
			debug = 1;
			break;
		case 'i':
			iters = atoi(optarg);
			break;
		case 's':
			size = atoi(optarg);
			break;
		case 'S':
			srandom((unsigned)(atoi(optarg)));
			break;
		case 't':
			tnum = atoi(optarg);
			break;
		case '?':
		default:
			return (usage());
		}

	argc -= optind;
	argv += optind;

	if (argc != 2)
		return (usage());

	/* read in the data-rule relations */
	infile = argv[0];

	INIT_TIME(tv_acc);
	START_TIME(tv_start);
	if ((ret = rules_init(infile, &nrules, &nsamples, &rules, 1)) != 0)
		return (ret);
	END_TIME(tv_start, tv_end, tv_acc);
	REPORT_TIME("analyze", "per rule", tv_acc, nrules);
    
	rule_print_all(rules, nrules, nsamples);
    
	printf("\n%d rules %d samples\n\n", nrules, nsamples);
	if (debug)
		rule_print_all(rules, nrules, nsamples);

	/*
	 * We treat the label file as a separate ruleset, since it has
	 * a similar format.
	 */
	infile = argv[1];
	if ((ret = rules_init(infile, &nlabels, &nsamples_chk, &labels, 0)) != 0)
		return (ret);
    	assert(nlabels == 2);
	assert(nsamples_chk == nsamples);
    
	rule_print_all(labels, nlabels, nsamples);
	printf("\n%d labels %d samples\n\n", nlabels, nsamples);
    
	switch(tnum) {
		case 1:
			run_experiment(iters, size, nsamples, nrules, rules);
			break;
		case 2:
			run_mcmc(iters, size, nsamples, nrules, rules, labels);
			break;
		default:
			usage();
			break;
	}
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
	if (debug)
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
		if (debug) {
			printf("Initial ruleset\n");
			ruleset_print(rs, rules);
		}

		/* Now perform-(size-2) squared swaps */
		INIT_TIME(tv_acc);
		START_TIME(tv_start);
		for (j = 0; j < size; j++)
			for (k = 1; k < (size-1); k++) {
				if (debug)
					printf("\nSwapping rules %d and %d\n",
					    rs->rules[k-1].rule_id,
					    rs->rules[k].rule_id);
				if (ruleset_swap(rs, k - 1, k, rules))
					return;
				if (debug)
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
			if (debug)
				printf("\nDeleting rule %d\n", j);
			ruleset_delete(rules, nrules, rs, j);
			if (debug) 
				ruleset_print(rs, rules);
			add_random_rule(rules, nrules, rs, j);
			if (debug)
				ruleset_print(rs, rules);
		}
		END_TIME(tv_start, tv_end, tv_acc);
		REPORT_TIME("analyze", "per add/del", tv_acc, ((size-1) * 2));
		ruleset_destroy(rs);
	}

}

void
init_gsl_rand_gen() {
	gsl_rng_env_setup();
	RAND_GSL = gsl_rng_alloc(gsl_rng_default);
}

void
run_mcmc(int iters,
    int init_size, int nsamples, int nrules, rule_t *rules, rule_t *labels) {
	char stepchar;
	double jump_prob, log_post_rs=0.0, log_post_rs_proposal=0.0;
	double max_log_posterior = 1e-9;
	int i,j,t, ndx1, ndx2;
	int len, ret, *rs_idarray;
	ruleset_t *rs, *rs_proposal, *rs_temp;

	rs_idarray = NULL;
	rs_proposal = NULL;
	rs_temp = NULL;

	/* initialize random number generator for some distrubitions */
	init_gsl_rand_gen();
    
	/* Create a random rule set and set up initial parameters. */
	if ((ret =
	    create_random_ruleset(init_size, nsamples, nrules, rules, &rs)) != 0)
		return;

	log_post_rs = compute_log_posterior(rs, rules, nrules, labels);
	if (ruleset_backup(rs, &rs_idarray) != 0)
		return;
	max_log_posterior = log_post_rs;

	len = rs->n_rules;

	/* MIS I guessed this was for testing/debugging and put it under debug. */
	if (debug) {
		for (int i = 0; i < 10; i++) {
			ruleset_proposal(rs,
			    nrules, &ndx1, &ndx2, &stepchar, &jump_prob);
			printf("\n%d, %d, %d, %c, %f\n",
			    nrules, ndx1, ndx2, stepchar, log(jump_prob));
		}
    	}
    
    
	if (ruleset_copy(&rs_proposal, rs) != 0)
		return;

	if (debug) {
		ruleset_print(rs, rules);

		printf("\n*****************************************\n");
		printf("\n %p %p %d\n", rs, rs_proposal, rs_proposal==NULL);
   	} 

	if (ruleset_copy(&rs_proposal, rs) != 0)
		return;

	if (debug) {
		ruleset_print(rs_proposal, rules);
		printf("\n %p %p \n", rs, rs_proposal);
		printf("iters = %d", iters);
		for (int i=0; i<iters; i++) {
			ruleset_print(rs, rules);
		}
	}
        
        ruleset_proposal(rs, nrules, &ndx1, &ndx2, &stepchar, &jump_prob);
	if (debug) {
		printf("\nnrules=%d, ndx1=%d, ndx2=%d, action=%c, %s=%f\n",
		    nrules, ndx1, ndx2, stepchar, "relativeProbability",
		    log(jump_prob));
		printf("%d rules currently in the ruleset, they are:\n", rs->n_rules);

        	for (int j=0; j<rs->n_rules; j++)
			printf("%u ", rs->rules[j].rule_id); printf("\n");
	}
        
        ruleset_copy(&rs_proposal, rs);
        
        switch (stepchar) {
            case 'A':
                ruleset_add(rules, nrules, rs_proposal, ndx1, ndx2);
                break;
            case 'D':
                ruleset_delete(rules, nrules, rs_proposal, ndx1);
                break;
            case 'S':
                // ruleset_swap_any(rs_proposal, ndx1, ndx2, rules);
                break;
            default:
                break;
        }

	if (debug)
		ruleset_print(rs_proposal, rules);
        
        log_post_rs_proposal =
	    compute_log_posterior(rs_proposal, rules, nrules, labels);

	if (log((random() / (float)RAND_MAX)) <
	    log_post_rs_proposal-log_post_rs+log(jump_prob)) {
		free(rs);
		rs = rs_proposal;
		log_post_rs = log_post_rs_proposal;
		rs_proposal = NULL;

		if (ruleset_backup(rs, &rs_idarray) != 0)
			return;
		max_log_posterior = log_post_rs;

		len = rs->n_rules;
	}
    
	/* regenerate the best rule list */
	ruleset_init(len, nsamples, rs_idarray, rules, &rs);

	for (int i=0; i < len; i++)
		printf("rule[%d]_id = %d\n", i, rs_idarray[i]);
	printf("nmax_log_posterior = %6f\n\n", max_log_posterior);
	ruleset_print(rs, rules);
}

/* Utilities for MCMC. */
double
compute_log_posterior(ruleset_t *rs, rule_t *rules, int nrules, rule_t *labels) {
	double log_prior = 0.0, log_likelihood = 0.0;
	static double eta_norm = 0;
	static double *log_lambda_pmf=NULL, *log_eta_pmf=NULL;
	int i,j,k,li;
	int card_count[1 + MAX_RULE_CARDINALITY];
	int maxcard = 0;
	int n0, n1;
	VECTOR v0;

	/* prior pre-calculation */
	if (log_lambda_pmf == NULL) {
		log_lambda_pmf = malloc(nrules*sizeof(double));
		log_eta_pmf = malloc((1+MAX_RULE_CARDINALITY)*sizeof(double));
		for (i = 0; i < nrules; i++)
			log_lambda_pmf[i] = log(gsl_ran_poisson_pdf(i, LAMBDA));
		for (i = 0; i <= MAX_RULE_CARDINALITY; i++)
			log_eta_pmf[i] = log(gsl_ran_poisson_pdf(i, ETA));
	}

	/* Calculate log_prior. */
	for (i = 0; i <= MAX_RULE_CARDINALITY; i++)
		card_count[i] = 0;

	for (i=0; i < rs->n_rules; i++) {
		card_count[rules[rs->rules[i].rule_id].cardinality]++;
		if (rules[rs->rules[i].rule_id].cardinality > maxcard)
			maxcard = rules[rs->rules[i].rule_id].cardinality;
	}
    
	log_prior += log_lambda_pmf[rs->n_rules-1];
	eta_norm = gsl_cdf_poisson_P(maxcard, ETA) - gsl_ran_poisson_pdf(0, ETA);
	for (i=0; i < rs->n_rules-1; i++){ //don't compute the last rule(default rule).
		li = rules[rs->rules[i].rule_id].cardinality;
		log_prior += log_eta_pmf[li] - log(eta_norm);
		log_prior += -log(card_count[li]);
		card_count[li]--;
		if (card_count[li] == 0)
			eta_norm -= exp(log_eta_pmf[li]);
	}

	/* calculate log_likelihood */
	rule_vinit(rs->n_samples, &v0);
	for (int j=0; j < rs->n_rules; j++) {
		rule_vand(v0, rs->rules[j].captures,
		    labels[0].truthtable, rs->n_samples, &n0);
		n1 = rs->rules[j].ncaptured - n0;
		log_likelihood += gsl_sf_lngamma(n0+1) + gsl_sf_lngamma(n1+1) - gsl_sf_lngamma(n0+n1+2);
	}
	return (log_prior + log_likelihood);
}


void
ruleset_proposal(ruleset_t *rs,
    int nrules_mined, int *ndx1, int *ndx2, char *stepchar, double *jumpRatio) {

	static double MOVEPROBS[15] = {
		0.0, 1.0, 0.0,
		0.0, 0.5, 0.5,
		0.5, 0.0, 0.5,
		1.0/3.0, 1.0/3.0, 1.0/3.0,
		1.0/3.0, 1.0/3.0, 1.0/3.0
	};
	static double JUMPRATIOS[15] = {
		0.0, 1/2, 0.0,
		0.0, 2.0/3.0, 2.0,
		1.0, 0.0, 2.0/3.0,
		1.0, 1.5, 1.0,
		1.0, 1.0, 1.0
	};

	double moveProbs[3], jumpRatios[3];
	double u;
	int *allrules, cnt, i, index1, index2, offset;


	if (rs->n_rules == 0){
		offset = 0;
	} else if (rs->n_rules == 1){
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
    
	u = ((double) rand()) / (RAND_MAX);
	if (u < moveProbs[0]) {
		/* Swap rules -- but never the default (last) one. */
        	index1 = rand() % (rs->n_rules-1);
		do {
		    index2 = rand() % (rs->n_rules-1);
		} while (index2 == index1);

		*jumpRatio = jumpRatios[0];
		*stepchar = 'S';
	} else if (u < moveProbs[0] + moveProbs[1]) {
		/* Add a new rule. */
		index1 = rs->n_rules + 1 + rand() % (nrules_mined-rs->n_rules);
		allrules = calloc(nrules_mined, sizeof(int));

		for (i = 0; i < rs->n_rules; i++)
			allrules[rs->rules[i].rule_id] = -1;

		for (cnt = 0, i = 0; i < nrules_mined; i++)
			if (allrules[i] != -1)
				allrules[cnt++] = i;

		index1 = allrules[rand() % cnt];
		free(allrules);
		/* We do allow addition of a rule at default position. */
		index2 = rand() % rs->n_rules;
		*jumpRatio = jumpRatios[1] * (nrules_mined - 1 - rs->n_rules);
		*stepchar = 'A';
	} else if (u < moveProbs[0] + moveProbs[1] + moveProbs[2]) {
		/* Delete an existing rule; not the default one. */
		index1 = rand() % (rs->n_rules - 1);
		index2 = 0;
		*jumpRatio = jumpRatios[2] * (nrules_mined - rs->n_rules);
		*stepchar = 'D';
	} else{
		return;
	}

	*ndx1 = index1;
	*ndx2 = index2;
}

void
gsl_ran_poisson_test() {
	int i, j, p[10] = {};
	unsigned int number;
	unsigned int k1 = gsl_ran_poisson(RAND_GSL, 5);
	unsigned int k2 = gsl_ran_poisson(RAND_GSL, 5);

	printf("k1 = %u , k2 = %u\n", k1, k2);
    
	const int nrolls = 10000; // number of experiments
	const int nstars = 100;   // maximum number of stars to distribute
    
	for (i = 0; i < nrolls; ++i) {
		number = gsl_ran_poisson(RAND_GSL, 4.1);
		if ( number < 10)
			++p[number];
	}
    
	printf("poisson_distribution (mean=4.1):\n" );
	for (i = 0; i < 10; ++i) {
		printf("%d, : ", i);
		for (j = 0; j < p[i]*nstars/nrolls; j++)
			printf("*");
		printf("\n");
	}
}
