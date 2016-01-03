/*
 * Copyright 2015 President and Fellows of Harvard College.
 * All rights reserved.
 */

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



/*
 * We have slightly different structures to represent the original rules 
 * and rulesets. The original structure contains the ascii representation
 * of the rule; the ruleset structure refers to rules by ID and contains
 * captures which is something computed off of the rule's truth table.
 */
typedef struct rule {
	char *features;			/* Representation of the rule. */
	int support;			/* Number of 1's in truth table. */
    int cardinality;
	VECTOR truthtable;		/* Truth table; one bit per sample. */
} rule_t;

typedef struct ruleset_entry {
	unsigned rule_id;
	int ncaptured;			/* Number of 1's in bit vector. */
	VECTOR captures;		/* Bit vector. */
} ruleset_entry_t;

typedef struct ruleset {
	int n_rules;			/* Number of actual rules. */
	int n_alloc;			/* Spaces allocated for rules. */
	int n_samples;
	ruleset_entry_t rules[];	/* Array of rules. */
} ruleset_t;

typedef struct params {
    double LAMBDA;
    double ETA;
    double ALPHA[2];
    double THRESHOLD;
    int iters;
    int init_size;
    int nchain;
} params_t;

typedef struct data {
    rule_t * rules;			/* rules in BitVector form in the data */
    rule_t * labels;			/* labels in BitVector form in the data */
    int nrules;
    int nsamples;                    /* number of samples in the data. */
} data_t;

typedef struct interval {
    double a, b;
} interval_t;

typedef struct pred_model {
    ruleset_t * rs;         /* best ruleset. */
    char ** rs_str;
    double * theta;
    interval_t * confIntervals;
} pred_model_t;


/*
 * Functions in the library
 */
int ruleset_init(int, int, int *, rule_t *, ruleset_t **);
int ruleset_add(rule_t *, int, ruleset_t *, int, int);
void ruleset_delete(rule_t *, int, ruleset_t *, int);
int ruleset_swap(ruleset_t *, int, int, rule_t *);
int ruleset_swap_any(ruleset_t *, int, int, rule_t *);

void ruleset_print(ruleset_t *, rule_t *);
void ruleset_entry_print(ruleset_entry_t *, int);
void ruleset_print_4test(ruleset_t *);

int rules_init(const char *, int *, int *, rule_t **, int);

void rule_print(rule_t *, int, int);
void rule_print_all(rule_t *, int, int);
void rule_vector_print(VECTOR, int);
void rule_copy(VECTOR, VECTOR, int);

int rule_vinit(int, VECTOR *);
int make_default(VECTOR *, int);
void rule_vand(VECTOR, VECTOR, VECTOR, int, int *);
void rule_vandnot(VECTOR, VECTOR, VECTOR, int, int *);
void rule_vor(VECTOR, VECTOR, VECTOR, int, int *);
int count_ones(v_entry);
int count_ones_vector(VECTOR, int);
int isPositiveInTruthtable(VECTOR, int, int);