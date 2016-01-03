/*
 * Copyright 2015 President and Fellows of Harvard College.
 * All rights reserved.
 */
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "rule.h"

/* Function declarations. */
int ascii_to_vector(char *, size_t, int *, int *, VECTOR *);
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
 * OUTPUTS: an array of rule_t's
 */
int
rules_init(const char *infile, int *nrules, int *nsamples, rule_t **rules_ret, int add_default_rule)
{
	FILE *fi;
	char *line, *rulestr;
	int rule_cnt, sample_cnt, rsize;
	int i, ones, ret;
	rule_t *rules=NULL;
	rule_t default_rule;
	size_t len, rulelen;

	sample_cnt = rsize = 0;

	if ((fi = fopen(infile, "r")) == NULL)
        return (errno);
	/*
	 * Leave a space for the 0th (default) rule, which we'll add at
	 * the end.
	 */
	rule_cnt = (add_default_rule==1);
	while ((line = fgetln(fi, &len)) != NULL) {
		if (rule_cnt >= rsize) {
			rsize += RULE_INC;
			if (rules == NULL)
                rules = malloc(rsize * sizeof(rule_t));
            else
                rules = realloc(rules, rsize * sizeof(rule_t));
			if (rules == NULL)
				goto err;
		}
		/* Get the rule string; line will contain the bits. */
		if ((rulestr = strsep(&line, " ")) == NULL)
			goto err;
		rulelen = strlen(rulestr) + 1;
		len -= rulelen;
		if ((rules[rule_cnt].features = malloc(rulelen)) == NULL)
			goto err;
		(void)strncpy(rules[rule_cnt].features, rulestr, rulelen);
		/*
		 * At this point "len" is a line terminated by a newline
		 * at line[len-1]; let's make it a NUL and shorten the line
		 * length by one.
		 */
		line[len-1] = '\0';
		if (ascii_to_vector(line, len, &sample_cnt, &ones,
		    &rules[rule_cnt].truthtable) != 0)
		    	goto err;
		rules[rule_cnt].support = ones;
        int ncard=1, i=0;
        while (rules[rule_cnt].features[i] != '\0')
            ncard += rules[rule_cnt].features[i++]==',';
        rules[rule_cnt].cardinality = ncard;
		rule_cnt++;
	}
	/* All done! */
	fclose(fi);

	/* Now create the 0'th (default) rule. */
    if (add_default_rule) {
        rules[0].support = sample_cnt;
        rules[0].features = "default";
        rules[0].cardinality = 0;
        if (make_default(&rules[0].truthtable, sample_cnt) != 0)
            goto err;
    }

	*nsamples = sample_cnt;
	*nrules = rule_cnt;
	*rules_ret = rules;

	return (0);

err:
	ret = errno;

	/* Reclaim space. */
	if (rules != NULL) {
		for (i = 1; i < rule_cnt; i++) {
			free(rules[i].features);
#ifdef GMP
			mpz_clear(rules[i].truthtable);
#else
			free(rules[i].truthtable);
#endif
		}	
		free(rules);
	}
	(void)fclose(fi);
	return (ret);
}

/* Malloc a vector to contain nsamples bits. */
int
rule_vinit(int len, VECTOR *ret)
{
#ifdef GMP
	mpz_init(*ret);
#else
	int nentries;

	nentries = (len + BITS_PER_ENTRY - 1)/BITS_PER_ENTRY;
	if ((*ret = calloc(nentries, sizeof(v_entry))) == NULL)
		return(errno);
#endif
	return (0);
}

/*
 * Convert an ascii sequence of 0's and 1's to a bit vector.
 * This is a hand-coded naive implementation; we'll also support
 * the GMP library, switching between the two with a compiler directive.
 */
int
ascii_to_vector(char *line, size_t len, int *nsamples, int *nones, VECTOR *ret)
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
	if ((buf = malloc(bufsize * sizeof(v_entry))) == NULL)
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
		fprintf(stderr, "Wrong number of samples. Expected %d got %d\n",
		    *nsamples, i);
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
make_default(VECTOR *tt, int len)
{
#ifdef GMP
	mpz_t v;
	mpz_init2(v, len);
	mpz_com(*tt, v);
	return (0);
#else
//	int nbytes;

//	nbytes = (len + 7) / 8;
//	unsigned char *c;
//
//	if ((c = malloc(nbytes)) == NULL)
//		return (errno);
//	/* Set all full bytes */
//	memset(c, BYTE_MASK, nbytes);
//	/* Take care of a number of bits not divisible by 8. */
//	if (len % 8 != 0)
//		c[nbytes-1] = bit_ones[len % 8];
//	*tt = (VECTOR)c;
    
    int nentries;
    v_entry *c;
    nentries = (len + BITS_PER_ENTRY -1) / BITS_PER_ENTRY;
    
    if ((c = calloc(nentries, sizeof(v_entry))) == NULL)
        return(errno);
    v_entry MASK = 0;
    for (int i=0; i < sizeof(v_entry); i++)
        MASK = (MASK << 8) + BYTE_MASK;
    for (int i=0; i < nentries-1; i++)
        c[i] = MASK;
    if (len % BITS_PER_ENTRY != 0) {
        for (int i=0; i < len % BITS_PER_ENTRY; i++)
            c[nentries-1] = (c[nentries-1] << 1)+ 1;
    } else if (nentries>0) {
        c[nentries-1] = MASK;
    }
    *tt = (VECTOR)c;
	return (0);
#endif
}

int
ruleset_init(int nrules,
    int nsamples, int *idarray, rule_t *rules, ruleset_t **retruleset)
{
	int i, ret, tmp;
	rule_t *cur_rule;
	ruleset_t *rs;
	ruleset_entry_t *cur_re;
	VECTOR all_captured=NULL;

	/*
	 * Allocate space for the ruleset structure and the ruleset entries.
	 */
	rs = malloc(sizeof(ruleset_t) + nrules * sizeof(ruleset_entry_t));
	if (rs == NULL)
		return (errno);

	/*
	 * Allocate the ruleset at the front of the structure and then
	 * the ruleset_entry_t array at the end.
	 */
    printf("nsamples in ruleset_init = %d\n", nsamples);
	rs->n_rules = nrules;
	rs->n_alloc = nrules;
	rs->n_samples = nsamples;
	if ((ret = rule_vinit(nsamples, &all_captured)) != 0)
		goto err1;
    rule_copy(all_captured, rules[0].truthtable, nsamples); // no problem here
    make_default(&all_captured, nsamples);
//    printf("%p, %p\n", all_captured, rules[0].truthtable);
    int cnt = nsamples;
    for (i = 0; i < nrules; i++) {
        rs->rules[i].rule_id = idarray[i];
        if (rule_vinit(nsamples, &rs->rules[i].captures) != 0)
            goto err1;
        rule_vand(rs->rules[i].captures, all_captured, rules[idarray[i]].truthtable, nsamples, &(rs->rules[i].ncaptured));
        rule_vandnot(all_captured, all_captured, rs->rules[i].captures, nsamples, &cnt);
    }
    assert(cnt==0);
//	for (i = 0; i < nrules; i++) {
//		cur_rule = rules + idarray[i];
//		cur_re = rs->rules + i;
//		cur_re->rule_id = idarray[i];
//
//		if (i == 0) {
//			if (rule_vinit(nsamples, &cur_re->captures) != 0)
//				goto err1;
//			rule_copy(cur_re->captures,
//			    cur_rule->truthtable, nsamples);
//			cur_re->ncaptured = cur_rule->support;
//			rule_copy(all_captured,
//			    cur_rule->truthtable, nsamples);
//		} else {
//			if (rule_vinit(nsamples, &cur_re->captures) != 0)
//				goto err1;
//			rule_vandnot(cur_re->captures, cur_rule->truthtable,
//			    all_captured, nsamples, &cur_re->ncaptured);
//
//			/* Skip this on the last one. */
//			if (i != nrules - 1)
//				rule_vor(all_captured,
//				    all_captured, cur_re->captures, nsamples, &tmp);
//		}
//	}
	*retruleset = rs;
#ifdef GMP
	mpz_clear(all_captured);
#else
	if (all_captured != NULL)
		free(all_captured);
#endif
	return (0);

err1:
#ifdef GMP
	mpz_clear(all_captured);
#else
	if (all_captured != NULL)
		free(all_captured);
#endif
	for (int j = 0; j <= i; j++)
#ifdef GMP
		mpz_clear(rs->rules[i].captures);
#else
		if (rs->rules[i].captures != NULL)
			free (rs->rules[i].captures);
#endif
	return (ENOMEM);
}

/*
 * Add the specified rule to the ruleset at position ndx (shifting
 * all rules after ndx down by one).
 */
int
ruleset_add(rule_t *rules, int nrules, ruleset_t *rs, int newrule, int ndx)
{
	int i, ret, tmp, cnt, cnt2;
	rule_t *expand;
	VECTOR captured, caught;

	/* Check for space. */
	if (rs->n_alloc < rs->n_rules + 1) {
		expand = realloc(rs->rules, 
		    (rs->n_rules + 1) * sizeof(ruleset_entry_t));
		if (expand == NULL)
			return (errno);			
		rs->n_alloc = rs->n_rules + 1;
        rule_vinit(rs->n_samples, &rs->rules[rs->n_rules].captures);
	}
    
    rule_vinit(rs->n_samples, &caught);
    //rule_vand(caught, rules[0].truthtable, rules[0].truthtable, rules[0].support, &cnt2);
//    printf("at this point, cnt2 should be 1761!!!, it is = %d\n", cnt2);

    for (int k=ndx; k<rs->n_rules; k++)
        rule_vor(caught, caught, rs->rules[k].captures, rs->n_samples, &cnt);
    rs->n_rules++;
    for (int k=rs->n_rules-1; k>ndx; k--){
        rs->rules[k].rule_id = rs->rules[k-1].rule_id;
        rs->rules[k].ncaptured = 0;
    }
    rs->rules[ndx].rule_id = newrule;
    
//    printf("caught cnt = %d\nn_alloc=%d\nn_rules=%d\n", cnt, rs->n_alloc, rs->n_rules);
//    ruleset_print_4test(rs);

	/* Shift later rules down by 1. */
//	if (ndx != rs->n_rules)
//		memmove(rs->rules + (ndx + 1), rs->rules + ndx,
//		    sizeof(ruleset_entry_t) * (rs->n_rules - ndx));
    
    
	/*
	 * Insert new rule.
	 * 1. Compute what is already captured by earlier rules.
	 * 2. Add rule into ruleset.
	 * 3. Compute new captures for all rules following the new one.
	 */
//    rule_vinit(rs->n_samples, &captured);
//	if (ndx == 0) {
//		rule_copy(captured, rules[newrule].truthtable, rs->n_samples);
//	} else  {
//		rule_copy(captured,
//		    rules[rs->rules[0].rule_id].truthtable, rs->n_samples);
//
//		for (i = 1; i < ndx; i++)
//			rule_vor(captured, captured,
//			    rs->rules[i].captures, rs->n_samples, &tmp);
//		rule_vandnot(captured, rules[newrule].truthtable,
//		    captured, rs->n_samples, &rs->rules[ndx].ncaptured);
//
//	}
    
//	VECTOR_ASSIGN(rs->rules[ndx].captures, captured);
//    printf("cnt = %d \n", cnt);
    for (int k=ndx; k<rs->n_rules; k++) {
        if (rs->rules[k].captures==NULL) printf("%d, is null\n", k);
        rule_vand(rs->rules[k].captures, caught, rules[rs->rules[k].rule_id].truthtable, rs->n_samples, &(rs->rules[k].ncaptured));
        rule_vandnot(caught, caught, rs->rules[k].captures, rs->n_samples, &cnt);
    }
    assert(cnt == 0);

//	for (i = ndx + 1; i < rs->n_rules; i++)
//		rule_vandnot(rs->rules[i].captures,
//		    rules[rs->rules[i].rule_id].truthtable, rs->rules[i-1].captures,
//		    rs->n_samples, &rs->rules[i].ncaptured);
		
	return(0);
}

void ruleset_print_4test(ruleset_t *rs){
    printf("here-------------print_4test\n");
    int tot_ncaptured=0;
    for (int k=0; k<rs->n_rules; k++){
        printf("rule_id = %6d \t ncaptures = %6d\n", rs->rules[k].rule_id, rs->rules[k].ncaptured);
        tot_ncaptured += rs->rules[k].ncaptured;
    }
    assert(tot_ncaptured == rs->n_samples);
    printf("total ncaptured = %d \t total samples = %d\n", tot_ncaptured, rs->n_samples);
}

/*
 * Delete the rule in the ndx-th position in the given ruleset.
 */
void
ruleset_delete(rule_t *rules, int nrules, ruleset_t *rs, int ndx)
{
	int i, nset;
	VECTOR curvec, oldv, tmp_vec;

	/* Compute new captures for all rules following the one at ndx.  */
	VECTOR_ASSIGN(oldv, rs->rules[ndx].captures);
	if (rule_vinit(rs->n_samples, &tmp_vec) != 0)
		return;
	for (i = ndx + 1; i < rs->n_rules; i++) {
		/*
		 * My new captures is my old captures or'd with anything that
		 * was captured by ndx and is captured by my rule.
		 */
		VECTOR_ASSIGN(curvec, rs->rules[i].captures);
		rule_vand(tmp_vec, rules[rs->rules[i].rule_id].truthtable,
		    oldv, rs->n_samples, &nset);
		rs->rules[i].ncaptured += nset;
		rule_vor(curvec, curvec, tmp_vec, rs->n_samples, &nset);

		/*
		 * Now remove the ones from oldv that just got set for rule
		 * i because they should not be captured later.
		 */
		rule_vandnot(oldv, oldv, curvec, rs->n_samples, &nset);
	}

	/* Now remove alloc'd data for rule at ndx. */
#ifdef GMP
	mpz_clear(tmp_vec);
	mpz_clear(oldv);
#else
	(void)free(tmp_vec);
	(void)free(oldv);
#endif

	/* Shift up cells if necessary. */
	if (ndx != rs->n_rules - 1)
		memmove(rs->rules + ndx, rs->rules + ndx + 1,
		    sizeof(ruleset_entry_t) * (rs->n_rules - ndx));

	rs->n_rules--;
	return;
}

/* dest must exist */
void
rule_copy(VECTOR dest, VECTOR src, int len)
{
#ifdef GMP
	mpz_init_set(dest, src);
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
 * 	newlycaught = (forall k<=i  k.captures) & j.tt
 *	i.captures = i.captures & ~j.captures
 * 	j.captures = j.captures | newlycaught
 * 	then swap positions i and j
 */
int
ruleset_swap(ruleset_t *rs, int i, int j, rule_t *rules)
{
	int ndx, nset, ret;
	VECTOR caught, orig_i, orig_j, tt_j;
	ruleset_entry_t re;

	assert(i <= rs->n_rules);
	assert(j <= rs->n_rules);
	assert(i + 1 == j);

	VECTOR_ASSIGN(orig_i, rs->rules[i].captures);
	VECTOR_ASSIGN(orig_j, rs->rules[j].captures);

	/*
	 * Compute newly caught in two steps: first compute everything
	 * caught in rules 0 to i-1, the compute everything from rule J
	 * that was caught by i and was NOT caught by the previous sum
	 */
	if ((ret = rule_vinit(rs->n_samples, &caught)) != 0)
		return (ret);
	if (i != 0) {
		rule_copy(caught,
		    rules[rs->rules[0].rule_id].truthtable, rs->n_samples);
			return (ret);
		for (ndx = 1; ndx < i; ndx++)
			rule_vor(caught, caught,
			    rs->rules[ndx].captures, rs->n_samples, &nset);
	}

	rule_vandnot(orig_i, orig_i,
	    rules[rs->rules[j].rule_id].truthtable, rs->n_samples, &nset);
	rs->rules[i].ncaptured = nset;

	/*
	 * If we are about to become the first rule, then our captures array
	 * is simply our initial truth table.
	 */
	if (i == 0) {
		/* XXX This is wasteful -- doing an extra allocation. */
		rule_copy(rs->rules[j].captures,
		    rules[rs->rules[j].rule_id].truthtable, rs->n_samples);
		rs->rules[j].ncaptured = rules[rs->rules[j].rule_id].support;
#ifdef GMP
		mpz_clear(orig_j);
#else
		orig_j = NULL;
#endif
	} else {
		rule_vandnot(orig_j, rules[rs->rules[j].rule_id].truthtable,
		    caught, rs->n_samples, &nset);
		rs->rules[j].ncaptured = nset;
	}

	/* Now swap the two entries */
	re = rs->rules[i];
	rs->rules[i] = rs->rules[j];
	rs->rules[j] = re;

#ifdef GMP
	mpz_clear(caught);
#else
	free(caught);
#endif
	return (0);
}

/* Dest must have been created. */
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

/* Dest must exist */
void
rule_vandnot(VECTOR dest,
    VECTOR src1, VECTOR src2, int nsamples, int *ret_cnt)
{
#ifdef GMP
	mpz_com(dest, src2);
	mpz_and(dest, src1, dest);
	*ret_cnt = 0;
	*ret_cnt = mpz_popcount(dest);
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
    int cnt = 0;
    for (int i=0; i < (len+BITS_PER_ENTRY-1)/BITS_PER_ENTRY; i++) {
        cnt += count_ones(v[i]);
    }
    return cnt;
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

void
ruleset_print(ruleset_t *rs, rule_t *rules)
{
	int i, j, n;
	int total_support;
	rule_t *r;

	printf("%d rules %d samples\n", rs->n_rules, rs->n_samples);
	n = (rs->n_samples + BITS_PER_ENTRY - 1) / BITS_PER_ENTRY;

	total_support = 0;
	for (i = 0; i < rs->n_rules; i++) {
		rule_print(rules, rs->rules[i].rule_id, n);
		ruleset_entry_print(rs->rules + i, n);
		total_support += rs->rules[i].ncaptured;
	}
	printf("Total Captured: %d\n", total_support);
}

void
ruleset_entry_print(ruleset_entry_t *re, int n)
{
	printf("%d captured; \n", re->ncaptured);
//	rule_vector_print(re->captures, n);
}

void
rule_print(rule_t *rules, int ndx, int n)
{
	rule_t *r;

	r = rules + ndx;
	printf("RULE %d: ( %s ), support=%d, card=%d: ", ndx, r->features, r->support, r->cardinality);
//	rule_vector_print(r->truthtable, n);
}

void
rule_vector_print(VECTOR v, int n)
{
#ifdef GMP
	mpz_out_str(stdout, 16, v);
	printf("\n");
#else
	int i;

	for (i = 0; i < n; i++)
		printf("oct%lo ", v[i]);
	printf("\n");
#endif

}

void
rule_print_all(rule_t *rules, int nrules, int nsamples)
{
	int i, n;

	n = (nsamples + BITS_PER_ENTRY - 1) / BITS_PER_ENTRY;
	for (i = 0; i < nrules; i++)
		rule_print(rules, i, n);
}

int
isPositiveInTruthtable(VECTOR v, int n, int idx) {
    if (idx<0 || idx>=n) {printf("SOMETHING IS WRONG HERE!\n");return 0;}
    return ( (v[idx/BITS_PER_ENTRY]>>(idx % BITS_PER_ENTRY)) & 1);
}
