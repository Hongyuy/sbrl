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
#include <memory>

#define EPSILON 1e-9
#define MAX_RULE_CARDINALITY 10

/*
 * File global variables.
 * These make the library not thread safe. If we want to be thread safe during
 * training, then we should reference count these global tables.
 */
extern Ruleset g_rs_new;
extern std::unique_ptr<BitVec> g_v0;
static std::vector<double> log_lambda_pmf;
static std::vector<double> log_eta_pmf;
static std::vector<double> log_gammas;
static std::vector<double> eta_pmf;
static double eta_norm;
static int n_add, n_delete, n_swap;
static int maxcard;
static int card_count[1 + MAX_RULE_CARDINALITY];
static std::vector<double> log_card_count;

/* These hold the alpha parameter values to speed up log_gamma lookup. */
static int a0, a1, a01;

int debug;

double compute_log_posterior(Ruleset &,
                             const std::vector<Rule> &, const int, std::vector<Rule> &, const Params &, const int, const int, double &);
int gen_poission(double);
std::vector<double> get_theta(Ruleset &, std::vector<Rule> &, std::vector<Rule> &, const Params &);
void gsl_ran_poisson_test(void);
void init_gsl_rand_gen(gsl_rng **);

int my_rng(gsl_rng *RAND_GSL)
{
    return (unsigned)(gsl_rng_uniform(RAND_GSL) * RAND_MAX);
}
/****** These are the heart of both MCMC and SA ******/
/*
 * Once we encapsulate the acceptance critera, we can use the same routine,
 * propose, to make proposals and determine acceptance. This leaves a lot
 * of the memory management nicely constrained in this routine.
 */

bool mcmc_accepts(double new_log_post, double old_log_post,
                  double prefix_bound, double max_log_post, double &extra, gsl_rng *RAND_GSL)
{
    return (prefix_bound > max_log_post &&
            log((my_rng(RAND_GSL) / (double)RAND_MAX)) <
                (new_log_post - old_log_post + log(extra)));
}

bool sa_accepts(double new_log_post, double old_log_post,
                double prefix_bound, double max_log_post, double &extra, gsl_rng *RAND_GSL)
{
    return (prefix_bound > max_log_post &&
            (new_log_post > old_log_post ||
             (log((my_rng(RAND_GSL) / (double)RAND_MAX)) <
              (new_log_post - old_log_post) / extra)));
}

/*
 * Create a proposal; used both by simulated annealing and MCMC.
 * 1. Compute proposal parameters
 * 2. Create the new proposal ruleset
 * 3. Compute the log_posterior
 * 4. Call the appropriate function to determine acceptance criteria
 */
template <typename Func>
void propose(Ruleset &rs, std::vector<Rule> &rules, std::vector<Rule> &labels, int nrules,
             double &jump_prob, double &ret_log_post, double max_log_post,
             int &cnt, double &extra, const Params &params, gsl_rng *RAND_GSL,
             Func accept_func)
{
    Step stepchar;
    double new_log_post, prefix_bound;
    int change_ndx, ndx1, ndx2;
    rs.ruleset_copy_to(g_rs_new);

    g_rs_new.ruleset_proposal(nrules, ndx1, ndx2, stepchar, jump_prob, RAND_GSL);

    switch (stepchar)
    {
    case Step::Add:
        /* Add the rule whose id is ndx1 at position ndx2 */
        g_rs_new.ruleset_add(rules, nrules, ndx1, ndx2);
        change_ndx = ndx2 + 1;
        n_add++;
        break;
    case Step::Delete:
        /* Delete the rule at position ndx1. */
        change_ndx = ndx1;
        g_rs_new.ruleset_delete(rules, nrules, ndx1);
        n_delete++;
        break;
    case Step::Swap:
        /* Swap the rules at ndx1 and ndx2. */
        g_rs_new.ruleset_swap_any(ndx1, ndx2, rules);
        change_ndx = 1 + (ndx1 > ndx2 ? ndx1 : ndx2);
        n_swap++;
        break;
    default:
        break;
    }

    new_log_post = compute_log_posterior(g_rs_new,
                                         rules, nrules, labels, params, 0, change_ndx, prefix_bound);

    if (prefix_bound < max_log_post)
        cnt++;

    if (accept_func(new_log_post,
                    ret_log_post, prefix_bound, max_log_post, extra, RAND_GSL))
    {
        ret_log_post = new_log_post;
        // rs = std::move(g_rs_new);
        while (!rs.entries.empty())
            rs.recycle_to_pool();
        rs.entries.swap(g_rs_new.entries);
    }
}

/********** End of proposal routines *******/
void compute_log_gammas(int nsamples, const Params &params)
{
    int i, max;

    /* Pre-compute alpha sum for accessing the log_gammas. */
    a0 = params.alpha[0];
    a1 = params.alpha[1];
    a01 = a0 + a1;

    max = nsamples + 2 * (1 + a01);
    log_gammas = std::vector<double>(max);
    for (i = 1; i < max; i++)
        log_gammas[i] = gsl_sf_lngamma((double)i);
}

void compute_pmf(int nrules, const Params &params)
{
    int i;
    log_lambda_pmf = std::vector<double>(nrules);
    for (i = 0; i < nrules; i++)
    {
        log_lambda_pmf[i] =
            log(gsl_ran_poisson_pdf(i, params.lambda));
    }

    log_eta_pmf = std::vector<double>(1 + MAX_RULE_CARDINALITY);
    eta_pmf = std::vector<double>(1 + MAX_RULE_CARDINALITY);
    for (i = 0; i <= MAX_RULE_CARDINALITY; i++)
    {
        const auto pmf = gsl_ran_poisson_pdf(i, params.eta);
        eta_pmf[i] = pmf;
        log_eta_pmf[i] = log(pmf);
    }

    /*
     * For simplicity, assume that all the cardinalities
     * <= MAX_RULE_CARDINALITY appear in the mined rules
     */
    eta_norm = gsl_cdf_poisson_P(MAX_RULE_CARDINALITY, params.eta) - gsl_ran_poisson_pdf(0, params.eta);
}

void compute_cardinality(std::vector<Rule> &rules, int nrules)
{
    int i;
    for (i = 0; i <= MAX_RULE_CARDINALITY; i++)
        card_count[i] = 0;

    log_card_count.push_back(0);
    for (i = 0; i < nrules; i++)
    {
        card_count[rules[i].cardinality]++;
        if (rules[i].cardinality > maxcard)
            maxcard = rules[i].cardinality;
        log_card_count.push_back(log(i + 1));
    }
}

PredModel
train(Data &train_data, int initialization, int method, const Params &params)
{
    int chain;
    double max_pos, pos_temp, null_bound;
    std::vector<int> default_rule(1, 0);
    Ruleset rs = Ruleset::ruleset_init(train_data.nsamples, default_rule, train_data.rules);

    gsl_rng *RAND_GSL = NULL;
    /* initialize random number generator for some distributions. */
    init_gsl_rand_gen(&RAND_GSL);

    compute_pmf(train_data.nrules, params);
    compute_cardinality(train_data.rules, train_data.nrules);

    compute_log_gammas(train_data.nsamples, params);

    max_pos = compute_log_posterior(rs, train_data.rules,
                                    train_data.nrules, train_data.labels, params, 1, -1, null_bound);

    Permutations rule_permutation(train_data.nrules, RAND_GSL);

    for (chain = 0; chain < params.nchain; chain++)
    {
        auto rs_temp = run_mcmc(params.iters,
                                train_data.nsamples, train_data.nrules,
                                train_data.rules, train_data.labels, params, rule_permutation, max_pos, RAND_GSL);
        pos_temp = compute_log_posterior(rs_temp, train_data.rules,
                                         train_data.nrules, train_data.labels, params, 1, -1,
                                         null_bound);

        if (pos_temp >= max_pos)
        {
            rs = std::move(rs_temp);
            max_pos = pos_temp;
        }
    }
    const auto thetas = get_theta(rs, train_data.rules, train_data.labels, params);
    gsl_rng_free(RAND_GSL);
    return {rs.backup(), thetas, {}};
}

std::vector<double>
get_theta(Ruleset &rs, std::vector<Rule> &rules, std::vector<Rule> &labels, const Params &params)
{
    /* calculate captured 0's and 1's */
    std::vector<double> theta;
    int j;

    for (j = 0; j < rs.length(); j++)
    {
        int n0, n1;

        rule_vand(*g_v0, rs.entries[j].captures,
                  labels[0].truthtable, rs.n_samples, n0);
        n1 = rs.entries[j].ncaptured - n0;
        theta.push_back((n1 + params.alpha[1]) * 1.0 /
                        (n1 + n0 + params.alpha[0] + params.alpha[1]));
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
    /*
     * Construct rulesets with exactly 2 rules -- one drawn from
     * the permutation and the default rule.
     */
    rarray[1] = 0;
    count = 0;
    while (prefix_bound < v_star)
    {
        // TODO Gather some stats on how much we loop in here.
        if (rs.entries.size())
        {
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
    }

    /*
     * The initial ruleset is our best ruleset so far, so keep a
     * list of the rules it contains.
     */
    rs_idarray = rs.backup();
    max_log_posterior = log_post_rs;

    for (i = 0; i < iters; i++)
    {
        propose(rs, rules, labels, nrules, jump_prob,
                log_post_rs, max_log_posterior, nsuccessful_rej,
                jump_prob, params, RAND_GSL, mcmc_accepts);

        if (log_post_rs > max_log_posterior)
        {
            rs_idarray = rs.backup();
            max_log_posterior = log_post_rs;
        }
    }

    /* Regenerate the best rule list */
    rs = Ruleset::ruleset_init(nsamples, rs_idarray, rules);
    return (rs);
}

Ruleset
run_simulated_annealing(int iters, int init_size, int nsamples,
                        int nrules, std::vector<Rule> &rules, std::vector<Rule> &labels, const Params &params, gsl_rng *RAND_GSL)
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

    /* Pre-compute the cooling schedule. */

    tmp[0] = 1;
    for (i = 1; i < 28; i++)
    {
        tmp[i] = tmp[i - 1] + exp(0.25 * (i + 1));
        for (j = (int)tmp[i - 1]; j < (int)tmp[i]; j++)
            T[ntimepoints++] = 1.0 / (i + 1);
    }

    for (k = 0; k < ntimepoints; k++)
    {
        double tk = T[k];
        for (iter = 0; iter < iters_per_step; iter++)
        {
            propose(rs, rules, labels, nrules, jump_prob,
                    log_post_rs, max_log_posterior, dummy, tk,
                    params, RAND_GSL, sa_accepts);

            if (log_post_rs > max_log_posterior)
            {
                rs_idarray = rs.backup();
                max_log_posterior = log_post_rs;
            }
        }
    }
    /* Regenerate the best rule list. */
    return (rs);
}

double
compute_log_posterior(Ruleset &rs, const std::vector<Rule> &rules, const int nrules, std::vector<Rule> &labels,
                      const Params &params, const int ifPrint, const int length4bound, double &prefix_bound)
{

    double log_prior;
    double log_likelihood = 0.0;
    double prefix_prior = 0.0;
    double norm_constant, log_norm_constant;
    int i, j, li;
    int local_cards[1 + MAX_RULE_CARDINALITY];

    for (i = 0; i < (1 + MAX_RULE_CARDINALITY); i++)
        local_cards[i] = card_count[i];

    /* Calculate log_prior. */
    norm_constant = eta_norm;
    log_norm_constant = log(norm_constant);
    log_prior = log_lambda_pmf[rs.length() - 1];

    if (rs.length() - 1 > params.lambda)
        prefix_prior += log_lambda_pmf[rs.length() - 1];
    else
        prefix_prior += log_lambda_pmf[(int)params.lambda];

    // Don't compute the last (default) rule.
    for (i = 0; i < rs.length() - 1; i++)
    {
        li = rules[rs.entries[i].rule_id].cardinality;
        log_prior += log_eta_pmf[li] - log_norm_constant;

        log_prior -= log_card_count[local_cards[li]];
        if (i < length4bound)
        {
            // added for prefix_bound
            prefix_prior += log_eta_pmf[li] -
                            log_norm_constant - log_card_count[local_cards[li]];
        }

        local_cards[li]--;
        if (local_cards[li] == 0)
        {
            norm_constant -= eta_pmf[li];
            log_norm_constant = log(norm_constant);
        }
    }
    /* Calculate log_likelihood */
    double prefix_log_likelihood = 0.0;
    int left0 = labels[0].support, left1 = labels[1].support;

    for (j = 0; j < rs.length(); j++)
    {
        int n0, n1; // Count of 0's; count of 1's

        rule_vand(*g_v0, rs.entries[j].captures,
                  labels[0].truthtable, rs.n_samples, n0);
        n1 = rs.entries[j].ncaptured - n0;
        log_likelihood += log_gammas[n0 + a0] +
                          log_gammas[n1 + a1] -
                          log_gammas[n0 + n1 + a01];
        // Added for prefix_bound.
        left0 -= n0;
        left1 -= n1;
        if (j < length4bound)
        {
            prefix_log_likelihood += log_gammas[n0 + a0] +
                                     log_gammas[n1 + a1] - log_gammas[n0 + n1 + a01];
            if (j == (length4bound - 1))
            {
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
    return (log_prior + log_likelihood);
}

void Ruleset::ruleset_proposal(int nrules,
                               int &ndx1, int &ndx2, Step &stepchar, double &jumpRatio, gsl_rng *RAND_GSL) const
{
    static double MOVEPROBS[15] = {
        0.0, 1.0, 0.0,
        0.0, 0.5, 0.5,
        0.5, 0.0, 0.5,
        1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0,
        1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
    static double JUMPRATIOS[15] = {
        0.0, 0.5, 0.0,
        0.0, 2.0 / 3.0, 2.0,
        1.0, 0.0, 2.0 / 3.0,
        1.0, 1.5, 1.0,
        1.0, 1.0, 1.0};

    double moveProbs[3], jumpRatios[3];
    int offset = 0;
    if (this->length() == 1)
    {
        offset = 0;
    }
    else if (this->length() == 2)
    {
        offset = 3;
    }
    else if (this->length() == nrules - 1)
    {
        offset = 6;
    }
    else if (this->length() == nrules - 2)
    {
        offset = 9;
    }
    else
    {
        offset = 12;
    }
    memcpy(moveProbs, MOVEPROBS + offset, 3 * sizeof(double));
    memcpy(jumpRatios, JUMPRATIOS + offset, 3 * sizeof(double));

    double u = gsl_rng_uniform(RAND_GSL);
    int index1, index2;

    if (u < moveProbs[0])
    {
        // Swap rules: cannot swap with the default rule
        index1 = my_rng(RAND_GSL) % (this->length() - 1);

        // Make sure we do not swap with ourselves
        do
        {
            index2 = my_rng(RAND_GSL) % (this->length() - 1);
        } while (index2 == index1);

        jumpRatio = jumpRatios[0];
        stepchar = Step::Swap;
    }
    else if (u < moveProbs[0] + moveProbs[1])
    {
        /* Add a new rule */
        index1 = this->pick_random_rule(nrules, RAND_GSL);
        index2 = my_rng(RAND_GSL) % this->length();
        jumpRatio = jumpRatios[1] * (nrules - 1 - this->length());
        stepchar = Step::Add;
    }
    else if (u < moveProbs[0] + moveProbs[1] + moveProbs[2])
    {
        /* delete an existing rule */
        index1 = my_rng(RAND_GSL) % (this->length() - 1);
        // cannot delete the default rule
        index2 = 0;
        // index2 doesn 't matter in this case
        jumpRatio = jumpRatios[2] * (nrules - this->length());
        stepchar = Step::Delete;
    }
    else
    {
        // should raise exception here.
        throw std::runtime_error(std::string("unexpected: u = ") + std::to_string(u));
    }
    ndx1 = index1;
    ndx2 = index2;
}

void init_gsl_rand_gen(gsl_rng **p_RAND_GSL)
{
    if (*p_RAND_GSL == NULL)
    {
        gsl_rng_env_setup();
        *p_RAND_GSL = gsl_rng_alloc(gsl_rng_default);
        gsl_rng_set(*p_RAND_GSL, 0);
    }
}

unsigned RANDOM_RANGE(int lo, int hi, gsl_rng *RAND_GSL) { return (unsigned)(lo + (unsigned)((my_rng(RAND_GSL) / (float)RAND_MAX) * (hi - lo + 1))); }

int permute_cmp(const void *v1, const void *v2) { return ((permute_t *)v1)->val - ((permute_t *)v2)->val; }

Permutations::Permutations(int nrules, gsl_rng *RAND_GSL) : ptr{nullptr}, permute_ndx{0}
{
    if (ptr != NULL)
        throw std::runtime_error("Permutations: double initialization");
    if ((ptr = (permute_t *)malloc(sizeof(permute_t) * nrules)) == NULL)
        throw std::runtime_error("Permutations: malloc failed");
    for (int i = 0; i < nrules; i++)
    {
        ptr[i].val = my_rng(RAND_GSL);
        ptr[i].ndx = i;
    }
    qsort(ptr + 1, nrules - 1, sizeof(permute_t), permute_cmp);
    permute_ndx = 1;
}
