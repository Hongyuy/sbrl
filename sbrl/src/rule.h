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

#include <stdlib.h>
#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"
#include "gsl/gsl_cdf.h"
#include "gsl/gsl_sf.h"
#ifdef GMP
#include <gmp.h>
#endif

/*
 * This library implements rule set management for Bayesian rule lists.
 */

/*
 * Rulelist is an ordered collection of rules.
 * A Rule is simply and ID combined with a large binary vector of length N
 * where N is the number of samples and a 1 indicates that the rule captures
 * the sample and a 0 indicates that it does not.
 *
 * Definitions:
 * captures(R, S) -- A rule, R, captures a sample, S, if the rule evaluates
 * true for Sample S.
 * captures(N, S, RS) -- In ruleset RS, the Nth rule captures S.
 */

/*
 * Even though every rule in a given experiment will have the same number of
 * samples (n_ys), we include it in the rule definition. Note that the size of
 * the captures array will be n_ys/sizeof(unsigned).
 *
 * Note that a rule outside a rule set stores captures(R, S) while a rule in
 * a rule set stores captures(N, S, RS).
 */

#include <vector>
#include <string>
/*
 * Define types for bit vectors.
 */
typedef unsigned long v_entry;
#ifdef GMP
typedef mpz_t VECTOR;
#define VECTOR_ASSIGN(dest, src) mpz_init_set(dest, src)
#else
typedef v_entry *VECTOR;
#define VECTOR_ASSIGN(dest, src) dest = src
#endif

int my_rng(gsl_rng *);
#define RANDOM_RANGE(lo, hi) \
(unsigned)(lo + (unsigned)((my_rng(RAND_GSL) / (float)RAND_MAX) * (hi - lo + 1)))

/*
 * We have slightly different structures to represent the original rules 
 * and rulesets. The original structure contains the ascii representation
 * of the rule; the ruleset structure refers to rules by ID and contains
 * captures which is something computed off of the rule's truth table.
 */
enum class Step
{
	Add, Delete, Swap
};

struct Rule {
	std::string features;	/* Representation of the rule. */
	int support;			/* Number of 1's in truth table. */
	int cardinality;
	VECTOR truthtable;		/* Truth table; one bit per sample. */
};

struct RulesetEntry {
	unsigned rule_id;
	int ncaptured;			/* Number of 1's in bit vector. */
	VECTOR captures;		/* Bit vector. */
};

struct Ruleset {
	int n_rules;			/* Number of actual rules. */
	int n_alloc;			/* Spaces allocated for rules. */
	int n_samples;
	std::vector<RulesetEntry> entries;	/* Array of rules. */
	Ruleset(int n=0): entries(n) {}
};

struct Params {
	double lambda;
	double eta;
	double threshold;
	int alpha[2];
	int iters;
	int nchain;
};

struct Data {
	std::vector<Rule> rules;		/* rules in BitVector form in the data */
	std::vector<Rule> labels;	/* labels in BitVector form in the data */
	int nrules;		/* number of rules */
	int nsamples;		/* number of samples in the data. */
};

typedef struct interval {
	double a, b;
} interval_t;

// typedef struct pred_model {
//        Ruleset *rs;          /* best ruleset. */
//        double *theta;
//        interval_t *confIntervals;
// } pred_model_t;
struct PredModel
{
	Ruleset rs;							/* best ruleset. */
	std::vector<double> theta;
	std::vector<interval_t>confIntervals;

	PredModel() {};
};

/*
 * Functions in the library
 */
// size_t getline_portable(char **, size_t *, FILE *);
// char* strsep_portable(char **, const char *);
int ruleset_init(int, int, const std::vector<int> &, std::vector<Rule> &, Ruleset **);
int ruleset_add(std::vector<Rule> &, int, Ruleset **, int, int);
int ruleset_backup(Ruleset *, std::vector<int> &);
int ruleset_copy(Ruleset **, Ruleset *);
void ruleset_delete(std::vector<Rule> &, int, Ruleset *, int);
void ruleset_swap(Ruleset *, int, int, std::vector<Rule> &);
void ruleset_swap_any(Ruleset *, int, int, std::vector<Rule> &);
int pick_random_rule(int, Ruleset *, gsl_rng *);

void ruleset_destroy(Ruleset *);
//void ruleset_print(Ruleset *, Rule *, int);
//void ruleset_entry_print(RulesetEntry *, int, int);
int create_random_ruleset(int, int, int, std::vector<Rule> &, Ruleset **, gsl_rng *);

int rules_init(std::string &, int &, int &, std::vector<Rule> &, int);
void rules_free(std::vector<Rule> &, const int, int);

//void rule_print(Rule *, int, int, int);
//void rule_print_all(Rule *, int, int);
//void rule_vector_print(VECTOR, int);
void rule_copy(VECTOR, VECTOR, int);

int rule_ff1(VECTOR, int, int);
int rule_isset(VECTOR, int);
int rule_vinit(int, VECTOR *);
int rule_vfree(VECTOR *);
int make_default(VECTOR *, int);
void rule_vand(VECTOR, VECTOR, VECTOR, int, int &);
void rule_vandnot(VECTOR, VECTOR, VECTOR, int, int &);
void rule_vor(VECTOR, VECTOR, VECTOR, int, int &);
int count_ones(v_entry);
int count_ones_vector(VECTOR, int);

/* Functions for the Scalable Baysian Rule Lists */
double *predict(PredModel&, std::vector<Rule> &labels, const Params &);
int ruleset_proposal(Ruleset *, int, int *, int *, Step &, double *, gsl_rng *);
Ruleset *run_mcmc(int, int, int, std::vector<Rule> &, std::vector<Rule> &, const Params &, double, gsl_rng *);
Ruleset *run_simulated_annealing(int,
    int, int, int, std::vector<Rule> &, std::vector<Rule> &, const Params &, gsl_rng *);
PredModel train(Data &, int, int, const Params &);
