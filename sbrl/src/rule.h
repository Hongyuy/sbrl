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
#pragma once
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
#include <stdexcept>

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
unsigned RANDOM_RANGE(int lo, int hi, gsl_rng *RAND_GSL);

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

typedef struct _permute {
	int val;
	int ndx;
} permute_t;

struct Permutations {
	permute_t * ptr;
	int permute_ndx;
	Permutations(int nrules, gsl_rng *RAND_GSL);
	~Permutations() { if (ptr) free(ptr); }
	permute_t & operator [](int i) {return ptr[i];}
};

struct BitVec {
	VECTOR vec;
	int rule_ff1(int, int);
	int rule_isset(int);
	int count_ones_vector(int);
	void rule_copy(BitVec &, int);
	int make_default(int);
	BitVec(int n) { rule_vinit(n); };
	BitVec() = delete;
    BitVec(const BitVec &other) = delete;
	BitVec& operator= (const BitVec &other) = delete;
    BitVec(BitVec &&other) {
		vec->_mp_alloc = other.vec->_mp_alloc;
		vec->_mp_size = other.vec->_mp_size;
		vec->_mp_d = other.vec->_mp_d;
		other.vec->_mp_d = nullptr;
	}
	BitVec& operator= (BitVec &&other) {
		if (this == &other) return *this;
		this->vec->_mp_alloc = other.vec->_mp_alloc;
		this->vec->_mp_size = other.vec->_mp_size;
		this->vec->_mp_d = other.vec->_mp_d;
		other.vec->_mp_d = nullptr;
	}
	~BitVec() { rule_vfree(); }
private:
	int rule_vinit(int);
	int rule_vfree();
};

struct Rule {
	std::string features;	/* Representation of the rule. */
	int support;			/* Number of 1's in truth table. */
	int cardinality;
	BitVec truthtable;		/* Truth table; one bit per sample. */
	Rule(const std::string &feat, int supp, int card, int len): features{feat}, support{supp}, cardinality{card}, truthtable{len} {}
	Rule() = delete;
    Rule(const Rule &other) = delete;
	Rule& operator= (const Rule &other) = delete;
    Rule(Rule &&other): features{std::move(other.features)}, support{other.support}, cardinality{other.cardinality}, truthtable{std::move(other.truthtable)} {}
	Rule& operator= (Rule &&other) = delete;
};

struct RulesetEntry {
	unsigned rule_id;
	int ncaptured;			/* Number of 1's in bit vector. */
	BitVec captures;		/* Bit vector. */
	RulesetEntry(unsigned id, int ncap, int len): rule_id{id}, ncaptured{ncap}, captures{len} {}
	RulesetEntry() = delete;
    RulesetEntry(const RulesetEntry &other) = delete;
	RulesetEntry& operator= (const RulesetEntry &other) = delete;
    RulesetEntry(RulesetEntry &&other): rule_id{other.rule_id}, ncaptured{other.ncaptured}, captures{std::move(other.captures)} {
		other.rule_id = -1;
	}
	RulesetEntry& operator= (RulesetEntry &&other) {
		if (this == &other) return *this;
		this->rule_id = other.rule_id;
		this->ncaptured = other.ncaptured;
		this->captures = std::move(other.captures);
		other.rule_id = -1;
	}
};

struct Ruleset {
	int n_samples;
	std::vector<RulesetEntry> entries;	/* Array of rules. */
	Ruleset() = default;
	Ruleset(int nsamp): n_samples{nsamp} {}
    Ruleset(const Ruleset &other) = delete;
	Ruleset& operator= (const Ruleset &other) = delete;
	Ruleset(Ruleset &&other) = default;
	Ruleset& operator= (Ruleset &&other) = default;

	int length() const { return static_cast<int>(entries.size()); }
	std::vector<int> backup() const;
	int pick_random_rule(int, gsl_rng *) const;
	void ruleset_proposal(int, int &, int &, Step &, double &, gsl_rng *) const;
	void ruleset_add(std::vector<Rule> &, int, int, int);
	void ruleset_delete(std::vector<Rule> &, int, int);
	void ruleset_swap(int, int, std::vector<Rule> &);
	void ruleset_swap_any(int, int, std::vector<Rule> &);
	// void ruleset_destroy();
	static Ruleset ruleset_init(int, const std::vector<int> &, std::vector<Rule> &);
	static Ruleset create_random_ruleset(int, int, int, std::vector<Rule> &, gsl_rng *);
	Ruleset ruleset_copy();
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

	PredModel() = default;
    PredModel(const PredModel &other) = delete;
	PredModel& operator= (const PredModel &other) = delete;
	PredModel(PredModel &&other) = default;
	PredModel& operator= (PredModel &&other) = default;
};

/*
 * Functions in the library
 */
// size_t getline_portable(char **, size_t *, FILE *);
// char* strsep_portable(char **, const char *);

//void ruleset_print(Ruleset *, Rule *, int);
//void ruleset_entry_print(RulesetEntry *, int, int);

void rules_init(const std::string &, std::vector<Rule> &, const int, const int, const int);
void rules_free(std::vector<Rule> &, const int, int);

//void rule_print(Rule *, int, int, int);
//void rule_print_all(Rule *, int, int);
//void rule_vector_print(BitVec &, int);

int ascii_to_vector(const char *, size_t, int &, int &, BitVec &);
void rule_vand(BitVec &, BitVec &, BitVec &, int, int &);
void rule_vandnot(BitVec &, BitVec &, BitVec &, int, int &);
void rule_vor(BitVec &, BitVec &, BitVec &, int, int &);
int count_ones(v_entry);

/* Functions for the Scalable Baysian Rule Lists */
// double *predict(PredModel&, std::vector<Rule> &labels, const Params &);
Ruleset run_mcmc(int, int, int, std::vector<Rule> &, std::vector<Rule> &, const Params &, Permutations &, double, gsl_rng *);
Ruleset run_simulated_annealing(int,
    int, int, int, std::vector<Rule> &, std::vector<Rule> &, const Params &, gsl_rng *);
PredModel train(Data &, int, int, const Params &);
