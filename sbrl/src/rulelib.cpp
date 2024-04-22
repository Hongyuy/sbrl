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
#ifndef _GNU_SOURCE
	#define _GNU_SOURCE
#endif
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "rule.h"
#include "Rcpp.h"
#include <fstream>

/* Function declarations. */
int ascii_to_vector(const char *, size_t, int *, int *, VECTOR *);
int make_default(VECTOR *, int);
#define RULE_INC 100
#define BITS_PER_ENTRY (sizeof(v_entry) * 8)

/* One-counting tools */
int bit_ones[] = {0, 1, 3, 7, 15, 31, 63, 127};

int byte_ones[] = {
/*   0 */ 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
/*  16 */ 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
/*  32 */ 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
/*  48 */ 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 
/*  64 */ 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,	
/*  80 */ 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 
/*  96 */ 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
/* 112 */ 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
/* 128 */ 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
/* 144 */ 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
/* 160 */ 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
/* 176 */ 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
/* 192 */ 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
/* 208 */ 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
/* 224 */ 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
/* 240 */ 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8 };

#define BYTE_MASK	0xFF


/*
 * Preprocessing step.
 * INPUTS: Using the python from the BRL_code.py: Call get_freqitemsets
 * to generate data files of the form:
 * 	Rule<TAB><bit vector>\n
 *
 * OUTPUTS: an array of Rule's
 */

int
rules_init(std::string &infile, int &nrules,
    int &nsamples, std::vector<Rule> &rules_ret, int add_default_rule)
{
	std::fstream fi(infile.c_str());
	std::string linestr;
	Rule rule;
	int sample_cnt = 0, ones = 0;

	/*
	 * Leave a space for the 0th (default) rule, which we'll add at
	 * the end.
	 */
	if (add_default_rule)
		rules_ret.push_back(rule);
	while (std::getline(fi, linestr) && linestr.size()) {
		/* Get the rule string; line will contain the bits. */
		const auto pos = linestr.find(' ');
		if (pos == std::string::npos)
			goto err;
		rule.features = linestr.substr(0, pos);
		auto truthTable = linestr.data() + pos;
		auto truthTableLen = linestr.size() - pos - 1;
		/*
		 * At this point features is (probably) a line terminated by a
		 * newline at features[len-1]; if it is newline-terminated, then
		 * let's make it NUL-terminated and shorten the line length
		 * by one.
		 */
		if (ascii_to_vector(truthTable, truthTableLen, &sample_cnt, &ones,
		    &rule.truthtable) != 0)
		    	goto err;
		rule.support = ones;

		/* Now compute the number of clauses in the rule. */
		rule.cardinality = 1;
		for (char &c : rule.features)
			rule.cardinality += (c == ',');
		rules_ret.push_back(rule);
	}

	/* Now create the 0'th (default) rule. */
	if (add_default_rule) {
		rules_ret[0].support = sample_cnt;
		rules_ret[0].features = (char*)"default";
		rules_ret[0].cardinality = 0;
		if (make_default(&rules_ret[0].truthtable, sample_cnt) != 0)
		    goto err;
	}

	nsamples = sample_cnt;
	nrules = rules_ret.size();
	return (0);

err:
	return (errno);
}

void
rules_free(std::vector<Rule> &rules, const int nrules, int add_default) {
	int i, start;

	/* Cannot free features for default rule. */
	start = 0;
	if (add_default) {
		rule_vfree(&rules[0].truthtable);
		start = 1;
	}

	for (i = start; i < nrules; i++) {
		rule_vfree(&rules[i].truthtable);
	}
	// free(rules);
}

/* Malloc a vector to contain nsamples bits. */
int
rule_vinit(int len, VECTOR *ret)
{
#ifdef GMP
	mpz_init2(*ret, len);
#else
	int nentries;

	nentries = (len + BITS_PER_ENTRY - 1)/BITS_PER_ENTRY;
	if ((*ret = (v_entry*)calloc(nentries, sizeof(v_entry))) == NULL)
		return(errno);
#endif
	return (0);
}

/* Deallocate a vector. */
int
rule_vfree(VECTOR *v)
{
#ifdef GMP
	mpz_clear(*v);
	/* Clobber the memory. */
	memset(v, 0, sizeof(*v));

#else
	if (*v != NULL) {
		free(*v);
		*v = NULL;
	}
#endif
	return (0);
}

/*
 * Convert an ascii sequence of 0's and 1's to a bit vector.
 * Do this manually if we have the naive implementation; else use
 * GMP functions if we're using the GMP library.
 */
int
ascii_to_vector(const char *line, size_t len, int *nsamples, int *nones, VECTOR *ret)
{
#ifdef GMP
	int retval;
	size_t s;

	if (mpz_init_set_str(*ret, line, 2) != 0) {
		retval = errno;
		mpz_clear(*ret);
		return (retval);
	}
	if ((s = mpz_sizeinbase (*ret, 2)) > (size_t) *nsamples)
		*nsamples = (int) s;
		
	*nones = mpz_popcount(*ret);
	return (0);
#else
	/*
	 * If *nsamples is 0, then we will set it to the number of
	 * 0's and 1's. If it is non-zero, then we'll ensure that
	 * the line is the right length.
	 */

	char *p;
	int i, bufsize, last_i, ones;
	v_entry val;
	v_entry *bufp, *buf;

	/* NOT DONE */
	assert(line != NULL);

	/* Compute bufsize in number of unsigned elements. */
	if (*nsamples == 0)
		bufsize = (len + BITS_PER_ENTRY - 1) / BITS_PER_ENTRY;
	else
		bufsize = (*nsamples + BITS_PER_ENTRY - 1) / BITS_PER_ENTRY;
	if ((buf = (v_entry*)malloc(bufsize * sizeof(v_entry))) == NULL)
		return(errno);
	
	bufp = buf;
	val = 0;
	i = 0;
	last_i = 0;
	ones = 0;


	for(p = line; len-- > 0; p++) {
		switch (*p) {
			case '0':
				val <<= 1;
				i++;
				break;
			case '1':
				val <<= 1;
				val++;
				i++;
				ones++;
				break;
			default:
				break;
		}
		/* If we have filled up val, store it and reset it. */
		if (last_i != i && (i % BITS_PER_ENTRY) == 0) {
			*bufp = val;
			val = 0;
			bufp++;
			last_i = i;
		}
	}

	/* Store val if it contains any bits. */
	if ((i % BITS_PER_ENTRY) != 0)
		*bufp = val;

	if (*nsamples == 0)
		*nsamples = i;
	else if (*nsamples != i) {
//		fprintf(stderr, "Wrong number of samples. Expected %d got %d\n",
//		    *nsamples, i);
		/* free(buf); */
		buf = NULL;
	}
	*nones = ones;
	*ret = buf;
	return (0);
#endif
}

/*
 * Create the truthtable for a default rule -- that is, it captures all samples.
 */
int
make_default(VECTOR *ttp, int len)
{
#ifdef GMP
	mpz_init2(*ttp, len);
	mpz_ui_pow_ui(*ttp, 2, (unsigned long)len);
	mpz_sub_ui (*ttp, *ttp, 1);
	return (0);
#else
	VECTOR tt;
	size_t nventry, nbytes;
	unsigned char *c;
	int m;

	nventry = (len + BITS_PER_ENTRY - 1) / BITS_PER_ENTRY;
	nbytes = nventry * sizeof(v_entry);

	if ((c = (unsigned char *)malloc(nbytes)) == NULL)
		return (errno);

	/* Set all full bytes */
	memset(c, BYTE_MASK, nbytes);
	*ttp = tt = (VECTOR)c;

	/* Fix the last entry so it has 0's for any unused bits. */
	m = len % BITS_PER_ENTRY;
	if (m != 0)
		tt[nventry - 1] = tt[nventry - 1] >> (BITS_PER_ENTRY - m);
    
	return (0);
#endif
}

/* Create a ruleset. */
int
ruleset_init(int nrs_rules,
    int nsamples, int *idarray, std::vector<Rule> &rules, Ruleset **retruleset)
{
	int cnt, i;
	Ruleset *rs;
	VECTOR not_captured;

	/*
	 * Allocate space for the ruleset structure and the ruleset entries.
	 */
	rs = new Ruleset(nrs_rules);
	/*
	 * Allocate the ruleset at the front of the structure and then
	 * the RulesetEntry array at the end.
	 */
	rs->n_rules = 0;
	rs->n_alloc = nrs_rules;
	rs->n_samples = nsamples;
	make_default(&not_captured, nsamples);

	cnt = nsamples;
	for (i = 0; i < nrs_rules; i++) {
		auto cur_rule = &rules[idarray[i]];
		auto cur_re = &rs->entries[i];
		cur_re->rule_id = idarray[i];

		if (rule_vinit(nsamples, &cur_re->captures) != 0)
			goto err1;
		rs->n_rules++;
		rule_vand(cur_re->captures, not_captured,
		    cur_rule->truthtable, nsamples, &cur_re->ncaptured);

		rule_vandnot(not_captured, not_captured,
		    rs->entries[i].captures, nsamples, &cnt);
	}
	assert(cnt==0);

	*retruleset = rs;
	(void)rule_vfree(&not_captured);
	return (0);

err1:
	(void)rule_vfree(&not_captured);
	ruleset_destroy(rs);
	return (ENOMEM);
}

/*
 * Save the idarray for this ruleset incase we need to restore it.
 * We don't know how long the idarray currently is so always call
 * realloc, which will do the right thing.
 */
int
ruleset_backup(Ruleset *rs, int **rs_idarray)
{
	int i, *ids;
	
	ids = *rs_idarray;

	if ((ids = (int*)realloc(ids, (rs->n_rules * sizeof(int)))) == NULL)
		return (errno);

	for (i = 0; i < rs->n_rules; i++)
		ids[i] = rs->entries[i].rule_id;

	*rs_idarray = ids;

	return (0);
}

/*
 * When we copy rulesets, we always allocate new structures; this isn't
 * terribly efficient, but it's simpler. If the allocation and frees become
 * too expensive, we can make this smarter.
 */
int
ruleset_copy(Ruleset **ret_dest, Ruleset *src)
{
	int i;
	Ruleset *dest = new Ruleset(src->n_rules);
	dest->n_alloc = src->n_rules;
	dest->n_rules = src->n_rules;
	dest->n_samples = src->n_samples;
    
	for (i = 0; i < src->n_rules; i++) {
		dest->entries[i].rule_id = src->entries[i].rule_id;
		dest->entries[i].ncaptured = src->entries[i].ncaptured;
		rule_vinit(src->n_samples, &(dest->entries[i].captures));
		rule_copy(dest->entries[i].captures,
		    src->entries[i].captures, src->n_samples);
	}
	*ret_dest = dest;

	return (0);
}

/* Reclaim resources associated with a ruleset. */
void
ruleset_destroy(Ruleset *rs)
{
	int j;
	for (j = 0; j < rs->n_rules; j++)
		rule_vfree(&rs->entries[j].captures);
	// free(rs);
}

/*
 * Add the specified rule to the ruleset at position ndx (shifting
 * all rules after ndx down by one).
 */
int
ruleset_add(std::vector<Rule> &rules, int nrules, Ruleset **rsp, int newrule, int ndx)
{
	int i, cnt;
	Ruleset *rs;
	// RulesetEntry *expand, *cur_re;
	VECTOR not_caught;

	rs = *rsp;
	/* Check for space. */
	if (rs->n_alloc < rs->n_rules + 1) {
		// expand = (RulesetEntry*)realloc(rs->entries, (rs->n_rules + 1) * sizeof(RulesetEntry));
		// if (expand == NULL)
		// 	return (errno);
		// rs->entries = expand;
		rs->entries.push_back({});
		rs->n_alloc = rs->n_rules + 1;
		*rsp = rs;
	}

	/*
	 * Compute all the samples that are caught by rules AFTER the
	 * rule we are inserting. Then we'll recompute all the captures
	 * from the new rule to the end.
	 */
	rule_vinit(rs->n_samples, &not_caught);
	for (i = ndx; i < rs->n_rules; i++)
	    rule_vor(not_caught,
	        not_caught, rs->entries[i].captures, rs->n_samples, &cnt);


	/*
	 * Shift later rules down by 1 if necessary. For GMP, what we're
	 * doing may be a little sketchy -- we're copying the mpz_t's around.
	 */
	if (ndx != rs->n_rules) {
		// memmove(rs->entries + (ndx + 1), rs->entries + ndx,
		//     sizeof(RulesetEntry) * (rs->n_rules - ndx));
		for (int i = rs->n_rules; i > ndx; --i)
			rs->entries[i] = rs->entries[i-1];
	}

	/* Insert and initialize the new rule. */
	rs->n_rules++;
	rs->entries[ndx].rule_id = newrule;
	rule_vinit(rs->n_samples, &rs->entries[ndx].captures);

	/*
	 * Now, recompute all the captures entries for the new rule and
	 * all rules following it.
	 */
    
	for (i = ndx; i < rs->n_rules; i++) {
		auto cur_re = &rs->entries[i];
		/*
		 * Captures for this rule gets anything in not_caught
		 * that is also in the rule's truthtable.
		 */
		rule_vand(cur_re->captures,
		    not_caught, rules[cur_re->rule_id].truthtable,
		    rs->n_samples, &cur_re->ncaptured);

		rule_vandnot(not_caught,
		    not_caught, cur_re->captures, rs->n_samples, &cnt);
	}
	assert(cnt == 0);
	rule_vfree(&not_caught);

	return(0);
}

/*
 * Delete the rule in the ndx-th position in the given ruleset.
 */
void
ruleset_delete(std::vector<Rule> &rules, int nrules, Ruleset *rs, int ndx)
{
	int i, nset;
	VECTOR tmp_vec;
	// RulesetEntry *old_re, *cur_re;

	/* Compute new captures for all rules following the one at ndx.  */
	auto old_re = &rs->entries[ndx];

	if (rule_vinit(rs->n_samples, &tmp_vec) != 0)
		return;
	for (i = ndx + 1; i < rs->n_rules; i++) {
		/*
		 * My new captures is my old captures or'd with anything that
		 * was captured by ndx and is captured by my rule.
		 */
		auto cur_re = &rs->entries[i];
		rule_vand(tmp_vec, rules[cur_re->rule_id].truthtable,
		    old_re->captures, rs->n_samples, &nset);
		rule_vor(cur_re->captures, cur_re->captures,
		    tmp_vec, rs->n_samples, &rs->entries[i].ncaptured);

		/*
		 * Now remove the ones from old_re->captures that just got set
		 * for rule i because they should not be captured later.
		 */
		rule_vandnot(old_re->captures, old_re->captures,
		    cur_re->captures, rs->n_samples, &nset);
	}

	/* Now remove alloc'd data for rule at ndx and for tmp_vec. */
	rule_vfree(&tmp_vec);
	rule_vfree(&rs->entries[ndx].captures);

	/* Shift up cells if necessary. */
	if (ndx != rs->n_rules - 1)
		// memmove(rs->entries + ndx, rs->entries + ndx + 1,
		//     sizeof(RulesetEntry) * (rs->n_rules - 1 - ndx));
		for (int i = ndx; i < rs->n_rules; ++i)
			rs->entries[i] = rs->entries[i+1];

	rs->n_rules--;
	return;
}

/*
 * We create random rulesets for testing and for creating initial proposals
 * in MCMC
 */
int
create_random_ruleset(int size,
    int nsamples, int nrules, std::vector<Rule> &rules, Ruleset **rs, gsl_rng *RAND_GSL)
{
	int i, j, *ids, next, ret;

	ids = (int*)calloc(size, sizeof(int));
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

	ret = ruleset_init(size, nsamples, ids, rules, rs);
	free(ids);
	return (ret);
}

#define MAX_TRIES 10
/*
 * Given a rule set, pick a random rule (not already in the set).
 */
int
pick_random_rule(int nrules, Ruleset *rs, gsl_rng *RAND_GSL)
{
	int cnt, j, new_rule;

	cnt = 0;
pickrule:
	if (cnt < MAX_TRIES)
		new_rule = RANDOM_RANGE(1, (nrules-1));
	else
		new_rule = 1 + (new_rule % (nrules-2));
		
	for (j = 0; j < rs->n_rules; j++) {
		if (rs->entries[j].rule_id == new_rule) {
			cnt++;
			goto pickrule;
		}
	}
	return (new_rule);
}

/* dest must exist */
void
rule_copy(VECTOR dest, VECTOR src, int len)
{
#ifdef GMP
	mpz_set(dest, src);
#else
	int i, nentries;

	assert(dest != NULL);
	nentries = (len + BITS_PER_ENTRY - 1)/BITS_PER_ENTRY;
	for (i = 0; i < nentries; i++)
		dest[i] = src[i];
#endif
}

/*
 * Swap rules i and j such that i + 1 = j.
 *	j.captures = j.captures | (i.captures & j.tt)
 *	i.captures = i.captures & ~j.captures
 * 	then swap positions i and j
 */
void
ruleset_swap(Ruleset *rs, int i, int j, std::vector<Rule> &rules)
{
	int nset;
	VECTOR tmp_vec;
	RulesetEntry re;

	assert(i < (rs->n_rules - 1));
	assert(j < (rs->n_rules - 1));
	assert(i + 1 == j);

	rule_vinit(rs->n_samples, &tmp_vec);

	/* tmp_vec =  i.captures & j.tt */
	rule_vand(tmp_vec, rs->entries[i].captures,
	    rules[rs->entries[j].rule_id].truthtable, rs->n_samples, &nset);
	/* j.captures = j.captures | tmp_vec */
	rule_vor(rs->entries[j].captures, rs->entries[j].captures,
	    tmp_vec, rs->n_samples, &rs->entries[j].ncaptured);

	/* i.captures = i.captures & ~j.captures */
	rule_vandnot(rs->entries[i].captures, rs->entries[i].captures,
	    rs->entries[j].captures, rs->n_samples, &rs->entries[i].ncaptured);

	/* Now swap the two entries */
	re = rs->entries[i];
	rs->entries[i] = rs->entries[j];
	rs->entries[j] = re;

	rule_vfree(&tmp_vec);
}

void
ruleset_swap_any(Ruleset * rs, int i, int j, std::vector<Rule> & rules)
{
	int cnt, cnt_check, k, temp;
	VECTOR caught;

	if (i == j)
		return;

	assert(i < rs->n_rules);
	assert(j < rs->n_rules);

	/* Ensure that i < j. */
	if (i > j) {
		temp = i;
		i = j;
		j = temp;
	}

	/*
	 * The captured arrays before i and after j need not change.
	 * We first compute everything caught between rules i and j
	 * (inclusive) and then compute the captures array from scratch
	 * rules between rule i and rule j, both * inclusive.
	 */
	rule_vinit(rs->n_samples, &caught);

	for (k = i; k <= j; k++)
		rule_vor(caught,
		    caught, rs->entries[k].captures, rs->n_samples, &cnt);

	/* Now swap the rules in the ruleset. */
	temp = rs->entries[i].rule_id;
	rs->entries[i].rule_id = rs->entries[j].rule_id;
	rs->entries[j].rule_id = temp;

	cnt_check = 0;
	for (k = i; k <= j; k++) {
		/*
		 * Compute the items captured by rule k by anding the caught
		 * vector with the truthtable of the kth rule.
		 */
		rule_vand(rs->entries[k].captures, caught,
		    rules[rs->entries[k].rule_id].truthtable,
		    rs->n_samples, &rs->entries[k].ncaptured);
		cnt_check += rs->entries[k].ncaptured;

		/* Now remove the caught items from the caught vector. */
		rule_vandnot(caught,
		    caught, rs->entries[k].captures, rs->n_samples, &temp);
	}
	assert(temp == 0);
	assert(cnt == cnt_check);

	rule_vfree(&caught);
}

/*
 * Dest must have been created.
 */
void
rule_vand(VECTOR dest, VECTOR src1, VECTOR src2, int nsamples, int *cnt)
{
#ifdef GMP
	mpz_and(dest, src1, src2);
	*cnt = 0;
	*cnt = mpz_popcount(dest);
#else
	int i, count, nentries;

	count = 0;
	nentries = (nsamples + BITS_PER_ENTRY - 1)/BITS_PER_ENTRY;
	assert(dest != NULL);
	for (i = 0; i < nentries; i++) {
		dest[i] = src1[i] & src2[i];
		count += count_ones(dest[i]);
	}
	*cnt = count;
	return;
#endif
}

/* Dest must have been created. */
void
rule_vor(VECTOR dest, VECTOR src1, VECTOR src2, int nsamples, int *cnt)
{
#ifdef GMP
	mpz_ior(dest, src1, src2);
	*cnt = 0;
	*cnt = mpz_popcount(dest);
#else
	int i, count, nentries;

	count = 0;
	nentries = (nsamples + BITS_PER_ENTRY - 1)/BITS_PER_ENTRY;

	for (i = 0; i < nentries; i++) {
		dest[i] = src1[i] | src2[i];
		count += count_ones(dest[i]);
	}
	*cnt = count;

	return;
#endif
}

/*
 * Dest must exist.
 * We use this to update existing vectors, so it has to work if the dest and
 * src1 are the same. It's easy in the naive implementation, but trickier in
 * the mpz because you can't do the AND and NOT at the same time. We benchmarked
 * allocating a temporary against doing this in 3 ops without the temporary and
 * the temporary is significantly faster (for large vectors).
 */
void
rule_vandnot(VECTOR dest,
    VECTOR src1, VECTOR src2, int nsamples, int *ret_cnt)
{
#ifdef GMP
	mpz_t tmp;

	rule_vinit(nsamples, &tmp);
	mpz_com(tmp, src2);
	mpz_and(dest, src1, tmp);
	*ret_cnt = 0;
	*ret_cnt = mpz_popcount(dest);
	rule_vfree(&tmp);
#else
	int i, count, nentries;

	nentries = (nsamples + BITS_PER_ENTRY - 1)/BITS_PER_ENTRY;
	count = 0;
	assert(dest != NULL);
	for (i = 0; i < nentries; i++) {
		dest[i] = src1[i] & (~src2[i]);
		count += count_ones(dest[i]);
	}

	*ret_cnt = count;
    return;
#endif
}

int
count_ones_vector(VECTOR v, int len) {
#ifdef GMP
	return mpz_popcount(v);
#else
	int cnt = 0, i;
	for (i = 0; i < (len+BITS_PER_ENTRY-1)/BITS_PER_ENTRY; i++) {
		cnt += count_ones(v[i]);
	}
	return cnt;
#endif
}

int
count_ones(v_entry val)
{
	int count, i;

	count = 0;
	for (i = 0; i < sizeof(v_entry); i++) {
		count += byte_ones[val & BYTE_MASK];
		val >>= 8;
	}
	return (count);
}

/*
 * Find first set bit starting at position start_pos.
 */
int
rule_ff1(VECTOR v, int start_pos, int len)
{
#ifdef GMP
	(void)len;
	return mpz_scan1(v, start_pos);
#else
	int i;
	for (i = start_pos; i < len; i++) {
		if (rule_isset(v, i))
			return i;
	}
	return -1;
#endif
}

//void
//ruleset_print(Ruleset *rs, Rule *rules, int detail)
//{
//	int i, n;
//	int total_support;
//
//	printf("%d rules %d samples\n", rs->n_rules, rs->n_samples);
//	n = (rs->n_samples + BITS_PER_ENTRY - 1) / BITS_PER_ENTRY;
//
//	total_support = 0;
//	for (i = 0; i < rs->n_rules; i++) {
//		rule_print(rules, rs->rules[i].rule_id, n, detail);
//		ruleset_entry_print(rs->rules + i, n, detail);
//		total_support += rs->rules[i].ncaptured;
//	}
//	printf("Total Captured: %d\n", total_support);
//}

//void
//ruleset_entry_print(RulesetEntry *re, int n, int detail)
//{
//	printf("%d captured; \n", re->ncaptured);
//	if (detail)
//		rule_vector_print(re->captures, n);
//}

//void
//rule_print(Rule *rules, int ndx, int n, int detail)
//{
//	Rule *r;
//
//	r = rules + ndx;
//	printf("RULE %d: ( %s ), support=%d, card=%d:",
//	    ndx, r->features, r->support, r->cardinality);
//	if (detail)
//		rule_vector_print(r->truthtable, n);
//}

//void
//rule_vector_print(VECTOR v, int n)
//{
//#ifdef GMP
//	mpz_out_str(stdout, 16, v);
//	printf("\n");
//#else
//	int i;
//	for (i = 0; i < n; i++)
//		printf("0x%lx ", v[i]);
//	printf("\n");
//#endif
//
//}

//void
//rule_print_all(Rule *rules, int nrules, int nsamples)
//{
//	int i, n;
//
//	n = (nsamples + BITS_PER_ENTRY - 1) / BITS_PER_ENTRY;
//	for (i = 0; i < nrules; i++)
//		rule_print(rules, i, n, 1);
//}

/*
 * Return 0 if bit e is not set in vector v; return non-0 otherwise.
 */
int
rule_isset(VECTOR v, int e) {
#ifdef GMP
	return mpz_tstbit(v, e);
#else
	return ((v[e/BITS_PER_ENTRY] & (1 << (e % BITS_PER_ENTRY))) != 0);
#endif
}
/*
size_t getline_portable(char **lineptr, size_t *n, FILE *stream) {
    char *bufptr = NULL;
    char *p = bufptr;
    size_t size, offset;
    int c;

    if (lineptr == NULL) {
    	return -1;
    }
    if (stream == NULL) {
    	return -1;
    }
    if (n == NULL) {
    	return -1;
    }
    bufptr = *lineptr;
    size = *n;

    c = fgetc(stream);
    if (c == EOF) {
    	return -1;
    }
    if (bufptr == NULL) {
		bufptr = (char*)malloc(128);
    	if (bufptr == NULL) {
    		return -1;
    	}
    	size = 128;
    }
    p = bufptr;
    while(c != EOF) {
		offset = p - bufptr;
    	if ((p - bufptr + 1) > size) {
    		size = size + 128;
			bufptr = (char*)realloc(bufptr, size);
    		if (bufptr == NULL) {
    			return -1;
    		}
			p = bufptr + offset;
    	}
    	*p++ = c;
    	if (c == '\n') {
    		break;
    	}
    	c = fgetc(stream);
    }

    *p++ = '\0';
    *lineptr = bufptr;
    *n = size;

    return p - bufptr - 1;
}

char* strsep_portable(char** stringp, const char* delim)
{
  char* start = *stringp;
  char* p;

  p = (start != NULL) ? strpbrk(start, delim) : NULL;

  if (p == NULL)
  {
    *stringp = NULL;
  }
  else
  {
    *p = '\0';
    *stringp = p + 1;
  }

  return start;
}
*/
