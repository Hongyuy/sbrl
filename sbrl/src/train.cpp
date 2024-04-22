/*
 * Copyright (c) 2016 Hongyu Yang, Cynthia Rudin, Margo Seltzer, and
 * The President and Fellows of Harvard College
 * 
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
/*
 * Scalable Bayesian Rulelist training
 */

#include <assert.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "mytime.h"
#include "rule.h"

#define EPSILON 1e-9
#define MAX_RULE_CARDINALITY 10

/*
 * File global variables.
 * These make the library not thread safe. If we want to be thread safe during
 * training, then we should reference count these global tables.
 */
static double *log_lambda_pmf, *log_eta_pmf;
static double *log_gammas;
static double eta_norm;
static int n_add, n_delete, n_swap;
static int maxcard;
static int card_count[1 + MAX_RULE_CARDINALITY];

/* These hold the alpha parameter values to speed up log_gamma lookup. */
static int a0, a1, a01;

typedef struct _permute {
	int val;
	int ndx;
} permute_t;
static permute_t *rule_permutation;
static int permute_ndx;

int debug;

// void _quicksort (void *const, size_t, size_t, int (const void *, const void *));
double compute_log_posterior(ruleset_t *,
    std::vector<rule_t> &, int, std::vector<rule_t> &, params_t *, int, int, double *);
int gen_poission(double);
std::vector<double> get_theta(ruleset_t *, std::vector<rule_t> &, std::vector<rule_t> &, params_t *);
void gsl_ran_poisson_test(void);
void init_gsl_rand_gen(gsl_rng**);

int my_rng(gsl_rng * RAND_GSL)
{
    return (unsigned)(gsl_rng_uniform (RAND_GSL) * RAND_MAX);
}
/****** These are the heart of both MCMC and SA ******/
/*
 * Once we encapsulate the acceptance critera, we can use the same routine,
 * propose, to make proposals and determine acceptance. This leaves a lot
 * of the memory management nicely constrained in this routine.
 */

int
mcmc_accepts(double new_log_post, double old_log_post,
             double prefix_bound, double max_log_post, double *extra, gsl_rng *RAND_GSL)
{
    /* Extra = jump_prob */
    return (prefix_bound > max_log_post &&
            log((my_rng(RAND_GSL) / (float)RAND_MAX)) <
            (new_log_post - old_log_post + log(*extra)));
}

int
sa_accepts(double new_log_post, double old_log_post,
           double prefix_bound, double max_log_post, double *extra, gsl_rng *RAND_GSL)
{
    /* Extra = tk */
    return (prefix_bound > max_log_post &&
            (new_log_post > old_log_post ||
             (log((my_rng(RAND_GSL) / (float)RAND_MAX)) <
              (new_log_post - old_log_post) / *extra)));
}


/*
 * Create a proposal; used both by simulated annealing and MCMC.
 * 1. Compute proposal parameters
 * 2. Create the new proposal ruleset
 * 3. Compute the log_posterior
 * 4. Call the appropriate function to determine acceptance criteria
 */
ruleset_t *
propose(ruleset_t *rs, std::vector<rule_t> &rules, std::vector<rule_t> &labels, int nrules,
    double *jump_prob, double *ret_log_post, double max_log_post,
    int *cnt, double *extra, params_t *params, gsl_rng *RAND_GSL,
    int (*accept_func)(double, double, double, double, double *, gsl_rng *))
{
	char stepchar;
	double new_log_post, prefix_bound;
	int change_ndx, ndx1, ndx2;
	ruleset_t *rs_new, *rs_ret;
	rs_new = NULL;

	if (ruleset_copy(&rs_new, rs) != 0)
		goto err;

	if (ruleset_proposal(rs_new,
	    nrules, &ndx1, &ndx2, &stepchar, jump_prob, RAND_GSL) != 0)
	    	goto err;

//	if (debug > 10) {
//		printf("Given ruleset: \n");
//		ruleset_print(rs, rules, (debug > 100));
//		printf("Operation %c(%d)(%d) produced proposal:\n",
//		    stepchar, ndx1, ndx2);
//	}
	switch (stepchar) {
	case 'A':
		/* Add the rule whose id is ndx1 at position ndx2 */
		if (ruleset_add(rules, nrules, &rs_new, ndx1, ndx2) != 0)
			goto err;
		change_ndx = ndx2 + 1;
		n_add++;
		break;
	case 'D':
		/* Delete the rule at position ndx1. */
		change_ndx = ndx1;
		ruleset_delete(rules, nrules, rs_new, ndx1);
		n_delete++;
		break;
	case 'S':
		/* Swap the rules at ndx1 and ndx2. */
		ruleset_swap_any(rs_new, ndx1, ndx2, rules);
		change_ndx = 1 + (ndx1 > ndx2 ? ndx1 : ndx2);
		n_swap++;
		break;
	default:
		goto err;
		break;
	}

	new_log_post = compute_log_posterior(rs_new,
	    rules, nrules, labels, params, 0, change_ndx, &prefix_bound);

//	if (debug > 10) {
//		ruleset_print(rs_new, rules, (debug > 100));
//		printf("With new log_posterior = %0.6f\n", new_log_post);
//	}
	if (prefix_bound < max_log_post)
		(*cnt)++;

	if (accept_func(new_log_post,
	    *ret_log_post, prefix_bound, max_log_post, extra, RAND_GSL)) {
//	    	if (debug > 10)
//			printf("Accepted\n");
		rs_ret = rs_new;
		*ret_log_post = new_log_post;
		ruleset_destroy(rs);
	} else {
//	    	if (debug > 10)
//			printf("Rejected\n");
		rs_ret = rs;
		ruleset_destroy(rs_new);
	}

	return (rs_ret);
err:
	if (rs_new != NULL)
		ruleset_destroy(rs_new);
	ruleset_destroy(rs);
	return (NULL);
}

/********** End of proposal routines *******/
int
compute_log_gammas(int nsamples, params_t *params)
{
	int i, max;

	/* Pre-compute alpha sum for accessing the log_gammas. */
	a0 = params->alpha[0];
	a1 = params->alpha[1];
	a01 = a0 + a1;

	max = nsamples + 2 * (1 + a01);
	log_gammas = (double*)malloc(sizeof(double) * max);
	if (log_gammas == NULL)
		return (-1);

	for (i = 1; i < max; i++)
		log_gammas[i] = gsl_sf_lngamma((double)i);
	return (0);
}

int
compute_pmf(int nrules, params_t *params)
{
	int i;
	if ((log_lambda_pmf = (double*)malloc(nrules * sizeof(double))) == NULL)
		return (errno);
	for (i = 0; i < nrules; i++) {
		log_lambda_pmf[i] =
		    log(gsl_ran_poisson_pdf(i, params->lambda));
//		if (debug > 100)
//			printf("log_lambda_pmf[ %d ] = %6f\n",
//			    i, log_lambda_pmf[i]);
	}

	if ((log_eta_pmf =
	    (double*)malloc((1 + MAX_RULE_CARDINALITY) * sizeof(double))) == NULL)
		return (errno);
	for (i = 0; i <= MAX_RULE_CARDINALITY; i++) {
		log_eta_pmf[i] =
		    log(gsl_ran_poisson_pdf(i, params->eta));
//		if (debug > 100)
//			printf("log_eta_pmf[ %d ] = %6f\n",
//			    i, log_eta_pmf[i]);
	}

	/*
	 * For simplicity, assume that all the cardinalities
	 * <= MAX_RULE_CARDINALITY appear in the mined rules
	 */
	eta_norm = gsl_cdf_poisson_P(MAX_RULE_CARDINALITY, params->eta)
	    - gsl_ran_poisson_pdf(0, params->eta);

//	if (debug > 10)
//		printf("eta_norm(Beta_Z) = %6f\n", eta_norm);

	return (0);
}

void
compute_cardinality(std::vector<rule_t> &rules, int nrules)
{
	int i;
	for (i = 0; i <= MAX_RULE_CARDINALITY; i++)
		card_count[i] = 0;

	for (i = 0; i < nrules; i++) {
		card_count[rules[i].cardinality]++;
		if (rules[i].cardinality > maxcard)
			maxcard = rules[i].cardinality;
	}

//	if (debug > 10)
//		for (i = 0; i <= MAX_RULE_CARDINALITY; i++)
//			printf("There are %d rules with cardinality %d.\n",
//			    card_count[i], i);
}

int
permute_cmp(const void *v1, const void *v2)
{
	return ((permute_t *)v1)->val - ((permute_t *)v2)->val;
}

int
permute_rules(int nrules, gsl_rng *RAND_GSL)
{
	int i;
	if ((rule_permutation = (permute_t*)malloc(sizeof(permute_t) * nrules)) == NULL)
		return (-1);
	for (i = 1; i < nrules; i++) {
		rule_permutation[i].val = my_rng(RAND_GSL);
		rule_permutation[i].ndx = i;
	}
//    _quicksort(rule_permutation, nrules, sizeof(permute_t), permute_cmp);
    qsort(rule_permutation, nrules, sizeof(permute_t), permute_cmp);
	permute_ndx = 1;
	return (0);

}

PredModel
train(Data &train_data, int initialization, int method, params_t *params)
{
	PredModel pred_model;
	int chain, default_rule;
	ruleset_t *rs, *rs_temp;
	double max_pos, pos_temp, null_bound;
    
    gsl_rng *RAND_GSL=NULL;
    /* initialize random number generator for some distributions. */
    init_gsl_rand_gen(&RAND_GSL);
    
	rs = NULL;
	if (compute_pmf(train_data.nrules, params) != 0)
		goto err;
	compute_cardinality(train_data.rules, train_data.nrules);

	if (compute_log_gammas(train_data.nsamples, params) != 0)
		goto err;


	default_rule = 0;
	if (ruleset_init(1,
	    train_data.nsamples, &default_rule, train_data.rules, &rs) != 0)
	    	goto err;

	max_pos = compute_log_posterior(rs, train_data.rules,
	    train_data.nrules, train_data.labels, params, 1, -1, &null_bound);
	if (permute_rules(train_data.nrules, RAND_GSL) != 0)
		goto err;

	for (chain = 0; chain < params->nchain; chain++) {
		rs_temp = run_mcmc(params->iters,
		    train_data.nsamples, train_data.nrules,
		    train_data.rules, train_data.labels, params, max_pos, RAND_GSL);
		pos_temp = compute_log_posterior(rs_temp, train_data.rules,
		    train_data.nrules, train_data.labels, params, 1, -1,
		    &null_bound);

		if (pos_temp >= max_pos) {
			ruleset_destroy(rs);
			rs = rs_temp;
			max_pos = pos_temp;
		} else {
			ruleset_destroy(rs_temp);
		}
	}

	pred_model.theta =
	    get_theta(rs, train_data.rules, train_data.labels, params);
	pred_model.rs = *rs;
	rs = NULL;

	/*
	 * THIS IS INTENTIONAL -- makes error handling localized.
	 * If we branch to err, then we want to free an allocated model;
	 * if we fall through naturally, then we don't.
	 */
err:
	/* Free allocated memory. */
	if (log_lambda_pmf != NULL)
		free(log_lambda_pmf);
	if (log_eta_pmf != NULL)
		free(log_eta_pmf);
	if (rule_permutation != NULL)
		free(rule_permutation);
	if (log_gammas != NULL)
		free(log_gammas);
	if (rs != NULL)
		ruleset_destroy(rs);
    
    gsl_rng_free(RAND_GSL);
    
	return (pred_model);
}

std::vector<double>
get_theta(ruleset_t * rs, std::vector<rule_t> & rules, std::vector<rule_t> & labels, params_t *params)
{
	/* calculate captured 0's and 1's */
	VECTOR v0;
	std::vector<double> theta;
	int j;

	rule_vinit(rs->n_samples, &v0);
	theta.reserve(rs->n_rules);

	for (j = 0; j < rs->n_rules; j++) {
		int n0, n1;

		rule_vand(v0, rs->rules[j].captures,
		    labels[0].truthtable, rs->n_samples, &n0);
		n1 = rs->rules[j].ncaptured - n0;
		theta[j] = (n1 + params->alpha[1]) * 1.0 /
		    (n1 + n0 + params->alpha[0] + params->alpha[1]);
//		if (debug) {
//			printf("n0=%d, n1=%d, captured=%d, training accuracy =",
//			    n0, n1, rs->rules[j].ncaptured);
//			if (theta[j] >= params->threshold)
//				printf(" %.8f\n",
//				    n1 * 1.0 / rs->rules[j].ncaptured);
//			else
//				printf(" %.8f\n",
//				    n0 * 1.0 / rs->rules[j].ncaptured);
//			printf("theta[%d] = %.8f\n", j, theta[j]);
//		}
	}
	rule_vfree(&v0);
	return (theta);
}

ruleset_t *
run_mcmc(int iters, int nsamples, int nrules,
    std::vector<rule_t> &rules, std::vector<rule_t> &labels, params_t *params, double v_star, gsl_rng *RAND_GSL)
{
	ruleset_t *rs;
	double jump_prob, log_post_rs;
	int *rs_idarray, len, nsuccessful_rej;
	int i, rarray[2], count;
	double max_log_posterior, prefix_bound;

	rs = NULL;
	rs_idarray = NULL;
	log_post_rs = 0.0;
	nsuccessful_rej = 0;
	prefix_bound = -1e10; // This really should be -MAX_DBL
	n_add = n_delete = n_swap = 0;

	/* Initialize the ruleset. */
//	if (debug > 10)
//		printf("Prefix bound = %10f v_star = %f\n",
//		    prefix_bound, v_star);
	/*
	 * Construct rulesets with exactly 2 rules -- one drawn from
	 * the permutation and the default rule.
	 */
	rarray[1] = 0;
	count = 0;
	while (prefix_bound < v_star) {
		// TODO Gather some stats on how much we loop in here.
		if (rs != NULL) {
			ruleset_destroy(rs);
			count++;
			if (count == (nrules - 1))
				return (NULL);
		}
		rarray[0] = rule_permutation[permute_ndx++].ndx;
		if (permute_ndx >= nrules)
			permute_ndx = 1;
		ruleset_init(2, nsamples, rarray, rules, &rs);
		log_post_rs = compute_log_posterior(rs, rules,
		    nrules, labels, params, 0, 1, &prefix_bound);
//		if (debug > 10) {
//			printf("Initial random ruleset\n");
//			ruleset_print(rs, rules, 1);
//			printf("Prefix bound = %f v_star = %f\n",
//			    prefix_bound, v_star);
//		}
	}

	/*
	 * The initial ruleset is our best ruleset so far, so keep a
	 * list of the rules it contains.
	 */
	if (ruleset_backup(rs, &rs_idarray) != 0)
		goto err;
	max_log_posterior = log_post_rs;
	len = rs->n_rules;

	for (i = 0; i < iters; i++) {
		if ((rs = propose(rs, rules, labels, nrules, &jump_prob,
		    &log_post_rs, max_log_posterior, &nsuccessful_rej,
		    &jump_prob, params, RAND_GSL, mcmc_accepts)) == NULL)
		    	goto err;

		if (log_post_rs > max_log_posterior) {
			if (ruleset_backup(rs, &rs_idarray) != 0)
				goto err;
			max_log_posterior = log_post_rs;
			len = rs->n_rules;
		}
	}

	/* Regenerate the best rule list */
	ruleset_destroy(rs);
	ruleset_init(len, nsamples, rs_idarray, rules, &rs);
	free(rs_idarray);

//	if (debug) {
//		printf("\n%s%d #add=%d #delete=%d #swap=%d):\n",
//		    "The best rule list is (#reject=", nsuccessful_rej,
//		    n_add, n_delete, n_swap);
//
//		printf("max_log_posterior = %6f\n", max_log_posterior);
//		printf("max_log_posterior = %6f\n",
//		    compute_log_posterior(rs, rules,
//		    nrules, labels, params, 1, -1, &prefix_bound));
//		ruleset_print(rs, rules, (debug > 100));
//	}
	return (rs);

err:
	if (rs != NULL)
		ruleset_destroy(rs);
	if (rs_idarray != NULL)
		free(rs_idarray);
	return (NULL);
}

ruleset_t *
run_simulated_annealing(int iters, int init_size, int nsamples,
    int nrules, std::vector<rule_t> & rules, std::vector<rule_t> & labels, params_t *params, gsl_rng *RAND_GSL)
{
	ruleset_t *rs;
	double jump_prob;
	int dummy, i, j, k, iter, iters_per_step, *rs_idarray = NULL, len;
	double log_post_rs, max_log_posterior = -1e9, prefix_bound = 0.0;
	double T[100000], tmp[50];
	int ntimepoints = 0;

	log_post_rs = 0.0;
	iters_per_step = 200;

	/* Initialize the ruleset. */
	if (create_random_ruleset(init_size, nsamples, nrules, rules, &rs, RAND_GSL) != 0)
		return (NULL);

	log_post_rs = compute_log_posterior(rs,
	    rules, nrules, labels, params, 0, -1, &prefix_bound);
	if (ruleset_backup(rs, &rs_idarray) != 0)
		goto err;
	max_log_posterior = log_post_rs;
	len = rs->n_rules;

//	if (debug > 10) {
//		printf("Initial ruleset: \n");
//		ruleset_print(rs, rules, (debug > 100));
//	}

	/* Pre-compute the cooling schedule. */

	tmp[0] = 1;
	for (i = 1; i < 28; i++) {
		tmp[i] = tmp[i - 1] + exp(0.25 * (i + 1));
		for (j = (int)tmp[i - 1]; j < (int)tmp[i]; j++)
			T[ntimepoints++] = 1.0 / (i + 1);
	}

//	if (debug > 0)
//		printf("iters_per_step = %d, #timepoints = %d\n",
//		    iters_per_step, ntimepoints);

	for (k = 0; k < ntimepoints; k++) {
		double tk = T[k];
		for (iter = 0; iter < iters_per_step; iter++) {
    			if ((rs = propose(rs, rules, labels, nrules, &jump_prob,
			    &log_post_rs, max_log_posterior, &dummy, &tk,
			    params, RAND_GSL, sa_accepts)) == NULL)
			    	goto err;

			if (log_post_rs > max_log_posterior) {
				if (ruleset_backup(rs, &rs_idarray) != 0)
					goto err;
				max_log_posterior = log_post_rs;
				len = rs->n_rules;
			}
		}
	}
	/* Regenerate the best rule list. */
	ruleset_destroy(rs);
//	printf("\n\n/*----The best rule list is: */\n");
//	ruleset_init(len, nsamples, rs_idarray, rules, &rs);
//	printf("max_log_posterior = %6f\n\n", max_log_posterior);
//	printf("max_log_posterior = %6f\n\n",
//	    compute_log_posterior(rs, rules,
//	    nrules, labels, params, 1, -1, &prefix_bound));
//	free(rs_idarray);
//	ruleset_print(rs, rules, (debug > 100));

	return (rs);
err:
	if (rs != NULL)
		ruleset_destroy(rs);
	if (rs_idarray != NULL)
		free(rs_idarray);
	return (NULL);
}

double
compute_log_posterior(ruleset_t *rs, std::vector<rule_t> &rules, int nrules, std::vector<rule_t> &labels,
    params_t *params, int ifPrint, int length4bound, double *prefix_bound)
{

	double log_prior;
	double log_likelihood = 0.0;
	double prefix_prior = 0.0;
	double norm_constant;
	int i, j, li;
	int local_cards[1 + MAX_RULE_CARDINALITY];

	for (i = 0; i < (1 + MAX_RULE_CARDINALITY); i++)
		local_cards[i] = card_count[i];

	/* Calculate log_prior. */
	norm_constant = eta_norm;
	log_prior = log_lambda_pmf[rs->n_rules - 1];

	if (rs->n_rules - 1 > params->lambda)
		prefix_prior += log_lambda_pmf[rs->n_rules - 1];
	else
		prefix_prior += log_lambda_pmf[(int)params->lambda];

	// Don't compute the last (default) rule.
	for (i = 0; i < rs->n_rules - 1; i++) {
		li = rules[rs->rules[i].rule_id].cardinality;
		log_prior += log_eta_pmf[li] - log(norm_constant);

		log_prior -= log(local_cards[li]);
		if (i < length4bound) {
			// added for prefix_bound
			prefix_prior += log_eta_pmf[li] - 
			    log(norm_constant) - log(local_cards[li]);
		}

		local_cards[li]--;
		if (local_cards[li] == 0)
			norm_constant -= exp(log_eta_pmf[li]);
	}
	/* Calculate log_likelihood */
	VECTOR v0;
	double prefix_log_likelihood = 0.0;
	int left0 = labels[0].support, left1 = labels[1].support;

	rule_vinit(rs->n_samples, &v0);
	for (j = 0; j < rs->n_rules; j++) {
		int n0, n1;	 // Count of 0's; count of 1's

		rule_vand(v0, rs->rules[j].captures,
		    labels[0].truthtable, rs->n_samples, &n0);
		n1 = rs->rules[j].ncaptured - n0;
		log_likelihood += log_gammas[n0 + a0] +
		    log_gammas[n1 + a1] - 
		    log_gammas[n0 + n1 + a01];
		// Added for prefix_bound.
		left0 -= n0;
		left1 -= n1;
		if (j < length4bound) {
			prefix_log_likelihood += log_gammas[n0 + a0] +
			    log_gammas[n1 + a1] - log_gammas[n0 + n1 + a01];
			if (j == (length4bound - 1)) {
				prefix_log_likelihood += log_gammas[a1] + 
				    log_gammas[left0 + a0] - 
				    log_gammas[left0 + a01] + 
				    log_gammas[a0] + 
				    log_gammas[left1 + a1] - 
				    log_gammas[left1 + a01];
			}
		}
	}
	*prefix_bound = prefix_prior + prefix_log_likelihood;
//	if (debug > 20)
//		printf("log_prior = %6f\t log_likelihood = %6f\n",
//		    log_prior, log_likelihood);
	rule_vfree(&v0);
	return (log_prior + log_likelihood);
}

int
ruleset_proposal(ruleset_t * rs, int nrules,
    int *ndx1, int *ndx2, char *stepchar, double *jumpRatio, gsl_rng *RAND_GSL){
	static double MOVEPROBS[15] = {
		0.0, 1.0, 0.0,
		0.0, 0.5, 0.5,
		0.5, 0.0, 0.5,
		1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0,
		1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
	};
	static double JUMPRATIOS[15] = {
		0.0, 0.5, 0.0,
		0.0, 2.0 / 3.0, 2.0,
		1.0, 0.0, 2.0 / 3.0,
		1.0, 1.5, 1.0,
		1.0, 1.0, 1.0
	};

	double moveProbs[3], jumpRatios[3];
	int offset = 0;
	if (rs->n_rules == 1) {
		offset = 0;
	} else if (rs->n_rules == 2) {
		offset = 3;
	} else if (rs->n_rules == nrules - 1) {
		offset = 6;
	} else if (rs->n_rules == nrules - 2) {
		offset = 9;
	} else {
		offset = 12;
	}
	memcpy(moveProbs, MOVEPROBS + offset, 3 * sizeof(double));
	memcpy(jumpRatios, JUMPRATIOS + offset, 3 * sizeof(double));

	double u = gsl_rng_uniform (RAND_GSL);
	int index1, index2;

	if (u < moveProbs[0]) {
		// Swap rules: cannot swap with the default rule
		index1 = my_rng(RAND_GSL) % (rs->n_rules - 1);

		// Make sure we do not swap with ourselves
		do {
			index2 = my_rng(RAND_GSL) % (rs->n_rules - 1);
		} while (index2 == index1);

		*jumpRatio = jumpRatios[0];
		*stepchar = 'S';
	} else if (u < moveProbs[0] + moveProbs[1]) {
		/* Add a new rule */
		index1 = pick_random_rule(nrules, rs, RAND_GSL);
		index2 = my_rng(RAND_GSL) % rs->n_rules;
		*jumpRatio = jumpRatios[1] * (nrules - 1 - rs->n_rules);
		*stepchar = 'A';
	} else if (u < moveProbs[0] + moveProbs[1] + moveProbs[2]) {
		/* delete an existing rule */
		index1 = my_rng(RAND_GSL) % (rs->n_rules - 1);
		//cannot delete the default rule
			index2 = 0;
		//index2 doesn 't matter in this case
			* jumpRatio = jumpRatios[2] * (nrules - rs->n_rules);
		*stepchar = 'D';
	} else {
		//should raise exception here.
		return -1;
	}
	*ndx1 = index1;
	*ndx2 = index2;
	return (0);
}

void
init_gsl_rand_gen(gsl_rng **p_RAND_GSL)
{
    if (*p_RAND_GSL == NULL) {
        gsl_rng_env_setup();
        *p_RAND_GSL = gsl_rng_alloc(gsl_rng_default);
        gsl_rng_set(*p_RAND_GSL, 0);
    }
}

//int
//gen_poisson(double mu)
//{
//	return ((int)gsl_ran_poisson(RAND_GSL, mu));
//}

double
gen_poission_pdf(int k, double mu)
{
	return (gsl_ran_poisson_pdf(k, mu));
}

double
gen_gamma_pdf (double x, double a, double b)
{
	return (gsl_ran_gamma_pdf(x, a, b));
}

