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
static std::vector<double> log_lambda_pmf;
static std::vector<double> log_eta_pmf;
static std::vector<double> log_gammas;
static double eta_norm;
static int n_add, n_delete, n_swap;
static int maxcard;
static int card_count[1 + MAX_RULE_CARDINALITY];

/* These hold the alpha parameter values to speed up log_gamma lookup. */
static int a0, a1, a01;

int debug;

// void _quicksort (void *const, size_t, size_t, int (const void *, const void *));
double compute_log_posterior(Ruleset &,
    std::vector<Rule> &, int, std::vector<Rule> &, const Params &, int, int, double &);
int gen_poission(double);
std::vector<double> get_theta(Ruleset &, std::vector<Rule> &, std::vector<Rule> &, const Params &);
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
             double prefix_bound, double max_log_post, double &extra, gsl_rng *RAND_GSL)
{
    /* Extra = jump_prob */
    return (prefix_bound > max_log_post &&
            log((my_rng(RAND_GSL) / (float)RAND_MAX)) <
            (new_log_post - old_log_post + log(extra)));
}

int
sa_accepts(double new_log_post, double old_log_post,
           double prefix_bound, double max_log_post, double &extra, gsl_rng *RAND_GSL)
{
    /* Extra = tk */
    return (prefix_bound > max_log_post &&
            (new_log_post > old_log_post ||
             (log((my_rng(RAND_GSL) / (float)RAND_MAX)) <
              (new_log_post - old_log_post) / extra)));
}


/*
 * Create a proposal; used both by simulated annealing and MCMC.
 * 1. Compute proposal parameters
 * 2. Create the new proposal ruleset
 * 3. Compute the log_posterior
 * 4. Call the appropriate function to determine acceptance criteria
 */
void
propose(Ruleset &rs, std::vector<Rule> &rules, std::vector<Rule> &labels, int nrules,
    double &jump_prob, double &ret_log_post, double max_log_post,
    int &cnt, double &extra, const Params &params, gsl_rng *RAND_GSL,
    int (*accept_func)(double, double, double, double, double &, gsl_rng *))
{
	Step stepchar;
	double new_log_post, prefix_bound;
	int change_ndx, ndx1, ndx2;
	Ruleset rs_new = rs.ruleset_copy();

	rs_new.ruleset_proposal(nrules, ndx1, ndx2, stepchar, jump_prob, RAND_GSL);

//	if (debug > 10) {
//		printf("Given ruleset: \n");
//		ruleset_print(rs, rules, (debug > 100));
//		printf("Operation %c(%d)(%d) produced proposal:\n",
//		    stepchar, ndx1, ndx2);
//	}
	switch (stepchar) {
	case Step::Add:
		/* Add the rule whose id is ndx1 at position ndx2 */
		rs_new.ruleset_add(rules, nrules, ndx1, ndx2);
		change_ndx = ndx2 + 1;
		n_add++;
		break;
	case Step::Delete:
		/* Delete the rule at position ndx1. */
		change_ndx = ndx1;
		rs_new.ruleset_delete(rules, nrules, ndx1);
		n_delete++;
		break;
	case Step::Swap:
		/* Swap the rules at ndx1 and ndx2. */
		rs_new.ruleset_swap_any(ndx1, ndx2, rules);
		change_ndx = 1 + (ndx1 > ndx2 ? ndx1 : ndx2);
		n_swap++;
		break;
	default:
		break;
	}

	new_log_post = compute_log_posterior(rs_new,
	    rules, nrules, labels, params, 0, change_ndx, prefix_bound);

//	if (debug > 10) {
//		ruleset_print(rs_new, rules, (debug > 100));
//		printf("With new log_posterior = %0.6f\n", new_log_post);
//	}
	if (prefix_bound < max_log_post)
		cnt++;

	if (accept_func(new_log_post,
	    ret_log_post, prefix_bound, max_log_post, extra, RAND_GSL)) {
//	    	if (debug > 10)
//			printf("Accepted\n");
		ret_log_post = new_log_post;
		// return rs_new;
		rs = rs_new;
	} else {
//	    	if (debug > 10)
//			printf("Rejected\n");
		// return rs;
	}
}

/********** End of proposal routines *******/
void
compute_log_gammas(int nsamples, const Params &params)
{
	int i, max;

	/* Pre-compute alpha sum for accessing the log_gammas. */
	a0 = params.alpha[0];
	a1 = params.alpha[1];
	a01 = a0 + a1;

	max = nsamples + 2 * (1 + a01);
	// log_gammas = (double*)malloc(sizeof(double) * max);
	// if (log_gammas == NULL)
	// 	return (-1);
	log_gammas = std::vector<double>(max);
	for (i = 1; i < max; i++)
		log_gammas[i] = gsl_sf_lngamma((double)i);
}

void
compute_pmf(int nrules, const Params &params)
{
	int i;
	log_lambda_pmf = std::vector<double>(nrules);
	for (i = 0; i < nrules; i++) {
		log_lambda_pmf[i] =
		    log(gsl_ran_poisson_pdf(i, params.lambda));
//		if (debug > 100)
//			printf("log_lambda_pmf[ %d ] = %6f\n",
//			    i, log_lambda_pmf[i]);
	}

	log_eta_pmf = std::vector<double>(1 + MAX_RULE_CARDINALITY);
	for (i = 0; i <= MAX_RULE_CARDINALITY; i++) {
		log_eta_pmf[i] =
		    log(gsl_ran_poisson_pdf(i, params.eta));
//		if (debug > 100)
//			printf("log_eta_pmf[ %d ] = %6f\n",
//			    i, log_eta_pmf[i]);
	}

	/*
	 * For simplicity, assume that all the cardinalities
	 * <= MAX_RULE_CARDINALITY appear in the mined rules
	 */
	eta_norm = gsl_cdf_poisson_P(MAX_RULE_CARDINALITY, params.eta)
	    - gsl_ran_poisson_pdf(0, params.eta);

//	if (debug > 10)
//		printf("eta_norm(Beta_Z) = %6f\n", eta_norm);
}

void
compute_cardinality(std::vector<Rule> &rules, int nrules)
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

PredModel
train(Data &train_data, int initialization, int method, const Params &params)
{
	PredModel pred_model;
	int chain;
	double max_pos, pos_temp, null_bound;
	std::vector<int> default_rule(1, 0);
	Ruleset rs = Ruleset::ruleset_init(train_data.nsamples, default_rule, train_data.rules);
    
    gsl_rng *RAND_GSL=NULL;
    /* initialize random number generator for some distributions. */
    init_gsl_rand_gen(&RAND_GSL);
    
	compute_pmf(train_data.nrules, params);
	compute_cardinality(train_data.rules, train_data.nrules);

	compute_log_gammas(train_data.nsamples, params);

	max_pos = compute_log_posterior(rs, train_data.rules,
	    train_data.nrules, train_data.labels, params, 1, -1, null_bound);

	Permutations rule_permutation(train_data.nrules, RAND_GSL);

	for (chain = 0; chain < params.nchain; chain++) {
		auto rs_temp = run_mcmc(params.iters,
		    train_data.nsamples, train_data.nrules,
		    train_data.rules, train_data.labels, params, rule_permutation, max_pos, RAND_GSL);
		pos_temp = compute_log_posterior(rs_temp, train_data.rules,
		    train_data.nrules, train_data.labels, params, 1, -1,
		    null_bound);

		if (pos_temp >= max_pos) {
			rs = rs_temp;
			max_pos = pos_temp;
		}
	}

	pred_model.theta =
	    get_theta(rs, train_data.rules, train_data.labels, params);
	pred_model.rs = rs;

	/*
	 * THIS IS INTENTIONAL -- makes error handling localized.
	 * If we branch to err, then we want to free an allocated model;
	 * if we fall through naturally, then we don't.
	 */
err:
	/* Free allocated memory. */
    gsl_rng_free(RAND_GSL);
    
	return (pred_model);
}

std::vector<double>
get_theta(Ruleset &rs, std::vector<Rule> & rules, std::vector<Rule> & labels, const Params &params)
{
	/* calculate captured 0's and 1's */
	BitVec v0;
	std::vector<double> theta;
	int j;

	v0.rule_vinit(rs.n_samples);

	for (j = 0; j < rs.length(); j++) {
		int n0, n1;

		rule_vand(v0, rs.entries[j].captures,
		    labels[0].truthtable, rs.n_samples, n0);
		n1 = rs.entries[j].ncaptured - n0;
		theta.push_back((n1 + params.alpha[1]) * 1.0 /
		    (n1 + n0 + params.alpha[0] + params.alpha[1]));
//		if (debug) {
//			printf("n0=%d, n1=%d, captured=%d, training accuracy =",
//			    n0, n1, rs->rules[j].ncaptured);
//			if (theta[j] >= params.threshold)
//				printf(" %.8f\n",
//				    n1 * 1.0 / rs->rules[j].ncaptured);
//			else
//				printf(" %.8f\n",
//				    n0 * 1.0 / rs->rules[j].ncaptured);
//			printf("theta[%d] = %.8f\n", j, theta[j]);
//		}
	}
	return (theta);
}

Ruleset
run_mcmc(int iters, int nsamples, int nrules,
    std::vector<Rule> &rules, std::vector<Rule> &labels, const Params &params, Permutations &rule_permutation, double v_star, gsl_rng *RAND_GSL)
{
	Ruleset rs;
	double jump_prob, log_post_rs;
	int nsuccessful_rej;
	int i, count;
	double max_log_posterior, prefix_bound;
	std::vector<int> rarray(2, 0);
	std::vector<int> rs_idarray;

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
		// if (rs != NULL) {
		if (rs.entries.size()) {
			count++;
			if (count == (nrules - 1))
				throw std::runtime_error("exausted rules");
		}
		rarray[0] = rule_permutation[rule_permutation.permute_ndx++].ndx;
		if (rule_permutation.permute_ndx >= nrules)
			rule_permutation.permute_ndx = 1;
		rs = Ruleset::ruleset_init(nsamples, rarray, rules);
		log_post_rs = compute_log_posterior(rs, rules,
		    nrules, labels, params, 0, 1, prefix_bound);
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
	rs_idarray = rs.backup();
	max_log_posterior = log_post_rs;

	for (i = 0; i < iters; i++) {
		propose(rs, rules, labels, nrules, jump_prob,
		    log_post_rs, max_log_posterior, nsuccessful_rej,
		    jump_prob, params, RAND_GSL, mcmc_accepts);

		if (log_post_rs > max_log_posterior) {
			rs_idarray = rs.backup();
			max_log_posterior = log_post_rs;
		}
	}

	/* Regenerate the best rule list */
	rs = Ruleset::ruleset_init(nsamples, rs_idarray, rules);

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
}

Ruleset
run_simulated_annealing(int iters, int init_size, int nsamples,
    int nrules, std::vector<Rule> & rules, std::vector<Rule> & labels, const Params &params, gsl_rng *RAND_GSL)
{
	Ruleset rs = Ruleset::create_random_ruleset(init_size, nsamples, nrules, rules, RAND_GSL);
	double jump_prob;
	int dummy, i, j, k, iter, iters_per_step;
	double log_post_rs, max_log_posterior = -1e9, prefix_bound = 0.0;
	double T[100000], tmp[50];
	int ntimepoints = 0;
	std::vector<int> rs_idarray;

	log_post_rs = 0.0;
	iters_per_step = 200;

	log_post_rs = compute_log_posterior(rs,
	    rules, nrules, labels, params, 0, -1, prefix_bound);
	rs_idarray = rs.backup();
	max_log_posterior = log_post_rs;

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
    		propose(rs, rules, labels, nrules, jump_prob,
			    log_post_rs, max_log_posterior, dummy, tk,
			    params, RAND_GSL, sa_accepts);

			if (log_post_rs > max_log_posterior) {
				rs_idarray = rs.backup();
				max_log_posterior = log_post_rs;
			}
		}
	}
	/* Regenerate the best rule list. */
//	printf("\n\n/*----The best rule list is: */\n");
//	ruleset_init(len, nsamples, rs_idarray, rules, &rs);
//	printf("max_log_posterior = %6f\n\n", max_log_posterior);
//	printf("max_log_posterior = %6f\n\n",
//	    compute_log_posterior(rs, rules,
//	    nrules, labels, params, 1, -1, &prefix_bound));
//	free(rs_idarray);
//	ruleset_print(rs, rules, (debug > 100));

	return (rs);
}

double
compute_log_posterior(Ruleset &rs, std::vector<Rule> &rules, int nrules, std::vector<Rule> &labels,
    const Params &params, int ifPrint, int length4bound, double &prefix_bound)
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
	log_prior = log_lambda_pmf[rs.length() - 1];

	if (rs.length() - 1 > params.lambda)
		prefix_prior += log_lambda_pmf[rs.length() - 1];
	else
		prefix_prior += log_lambda_pmf[(int)params.lambda];

	// Don't compute the last (default) rule.
	for (i = 0; i < rs.length() - 1; i++) {
		li = rules[rs.entries[i].rule_id].cardinality;
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
	BitVec v0;
	double prefix_log_likelihood = 0.0;
	int left0 = labels[0].support, left1 = labels[1].support;

	v0.rule_vinit(rs.n_samples);
	for (j = 0; j < rs.length(); j++) {
		int n0, n1;	 // Count of 0's; count of 1's

		rule_vand(v0, rs.entries[j].captures,
		    labels[0].truthtable, rs.n_samples, n0);
		n1 = rs.entries[j].ncaptured - n0;
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
	prefix_bound = prefix_prior + prefix_log_likelihood;
//	if (debug > 20)
//		printf("log_prior = %6f\t log_likelihood = %6f\n",
//		    log_prior, log_likelihood);
	return (log_prior + log_likelihood);
}

void
Ruleset::ruleset_proposal(int nrules,
    int &ndx1, int &ndx2, Step &stepchar, double &jumpRatio, gsl_rng *RAND_GSL) const {
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
	if (this->length() == 1) {
		offset = 0;
	} else if (this->length() == 2) {
		offset = 3;
	} else if (this->length() == nrules - 1) {
		offset = 6;
	} else if (this->length() == nrules - 2) {
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
		index1 = my_rng(RAND_GSL) % (this->length() - 1);

		// Make sure we do not swap with ourselves
		do {
			index2 = my_rng(RAND_GSL) % (this->length() - 1);
		} while (index2 == index1);

		jumpRatio = jumpRatios[0];
		stepchar = Step::Swap;
	} else if (u < moveProbs[0] + moveProbs[1]) {
		/* Add a new rule */
		index1 = this->pick_random_rule(nrules, RAND_GSL);
		index2 = my_rng(RAND_GSL) % this->length();
		jumpRatio = jumpRatios[1] * (nrules - 1 - this->length());
		stepchar = Step::Add;
	} else if (u < moveProbs[0] + moveProbs[1] + moveProbs[2]) {
		/* delete an existing rule */
		index1 = my_rng(RAND_GSL) % (this->length() - 1);
		//cannot delete the default rule
			index2 = 0;
		//index2 doesn 't matter in this case
			jumpRatio = jumpRatios[2] * (nrules - this->length());
		stepchar = Step::Delete;
	} else {
		//should raise exception here.
		throw std::runtime_error(std::string("unexpected: u = ") + std::to_string(u));
	}
	ndx1 = index1;
	ndx2 = index2;
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

/*
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
*/

unsigned RANDOM_RANGE(int lo, int hi, gsl_rng *RAND_GSL) { return (unsigned)(lo + (unsigned)((my_rng(RAND_GSL) / (float)RAND_MAX) * (hi - lo + 1))); }

int permute_cmp(const void *v1, const void *v2) { return ((permute_t *)v1)->val - ((permute_t *)v2)->val; }

Permutations::Permutations(int nrules, gsl_rng *RAND_GSL): ptr{nullptr}, permute_ndx{0}
{
	if (ptr != NULL)
		throw std::runtime_error("Permutations: double initialization");
	if ((ptr = (permute_t*)malloc(sizeof(permute_t) * nrules)) == NULL)
		throw std::runtime_error("Permutations: malloc failed");
	for (int i = 0; i < nrules; i++) {
		ptr[i].val = my_rng(RAND_GSL);
		ptr[i].ndx = i;
	}
	qsort(ptr+1, nrules-1, sizeof(permute_t), permute_cmp);
	permute_ndx = 1;
}
