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
    /* 240 */ 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};

#define BYTE_MASK 0xFF

/*
 * Preprocessing step.
 * INPUTS: Using the python from the BRL_code.py: Call get_freqitemsets
 * to generate data files of the form:
 * 	Rule<TAB><bit vector>\n
 *
 * OUTPUTS: an array of Rule's
 */

void rules_init(const std::string &infile, std::vector<Rule> &rules_ret,
                const size_t nrules_expected, const size_t nsamples_expected, const int add_default_rule)
{
    std::fstream fi(infile.c_str());
    std::string linestr;
    int sample_cnt = 0, ones = 0;

    /*
     * Leave a space for the 0th (default) rule, which we'll add at
     * the end.
     */
    if (add_default_rule)
        rules_ret.emplace_back("default", 0, 0, nsamples_expected);
    while (std::getline(fi, linestr) && linestr.size())
    {
        /* Get the rule string; line will contain the bits. */
        const auto pos = linestr.find(' ');
        if (pos == std::string::npos)
            // break;
            throw std::runtime_error("failed to parse rule name and truethtable");
        rules_ret.emplace_back(linestr.substr(0, pos), 0, 0, nsamples_expected);
        auto &rule = rules_ret.back();
        auto truthTable = linestr.data() + pos;
        auto truthTableLen = linestr.size() - pos - 1;
        /*
         * At this point features is (probably) a line terminated by a
         * newline at features[len-1]; if it is newline-terminated, then
         * let's make it NUL-terminated and shorten the line length
         * by one.
         */
        rule.truthtable.set_vector_from_ascii(truthTable, truthTableLen, sample_cnt, ones);
        rule.support = ones;

        /* Now compute the number of clauses in the rule. */
        rule.cardinality = 1;
        for (char &c : rule.features)
            rule.cardinality += (c == ',');
    }

    /* Now create the 0'th (default) rule. */
    if (add_default_rule)
    {
        rules_ret[0].support = sample_cnt;
        rules_ret[0].features = (char *)"default";
        rules_ret[0].cardinality = 0;
        rules_ret[0].truthtable.make_default(sample_cnt);
    }
    if (nrules_expected != rules_ret.size())
        throw std::runtime_error("nrules does not match expected");
    if (nsamples_expected != sample_cnt)
        throw std::runtime_error("sample_cnt does not match expected");
}

void rules_free(std::vector<Rule> &rules, const int nrules, int add_default)
{
    int i, start;

    /* Cannot free features for default rule. */
    start = 0;
    if (add_default)
    {
        start = 1;
    }

    for (i = start; i < nrules; i++)
    {
    }
    // free(rules);
}

/* Malloc a vector to contain nsamples bits. */
int BitVec::rule_vinit(size_t len)
{
    if (len <= 0)
        throw std::runtime_error("invalid len");
#ifdef GMP
    mpz_init2(this->vec, len);
#else
    int nentries;

    nentries = (len + BITS_PER_ENTRY - 1) / BITS_PER_ENTRY;
    if ((this->vec = (v_entry *)calloc(nentries, sizeof(v_entry))) == NULL)
        return (errno);
#endif
    return (0);
}

/* Deallocate a vector. */
int BitVec::rule_vfree()
{
#ifdef GMP
    mpz_clear(this->vec);
    /* Clobber the memory. */
    // memset(&this->vec, 0, sizeof(this->vec));

#else
    if (this->vec != NULL)
    {
        free(this->vec);
        this->vec = NULL;
    }
#endif
    return (0);
}

/*
 * Convert an ascii sequence of 0's and 1's to a bit vector.
 * Do this manually if we have the naive implementation; else use
 * GMP functions if we're using the GMP library.
 */
int BitVec::set_vector_from_ascii(const char *line, size_t len, int &nsamples, int &nones)
{
#ifdef GMP
    int retval;
    size_t s;

    if (mpz_set_str(this->vec, line, 2) != 0)
    {
        retval = errno;
        mpz_clear(this->vec);
        return (retval);
    }
    if ((s = mpz_sizeinbase(this->vec, 2)) > (size_t)nsamples)
        nsamples = (int)s;

    nones = mpz_popcount(this->vec);
    return (0);
#else
    /*
     * If nsamples is 0, then we will set it to the number of
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
    if (nsamples == 0)
        bufsize = (len + BITS_PER_ENTRY - 1) / BITS_PER_ENTRY;
    else
        bufsize = (nsamples + BITS_PER_ENTRY - 1) / BITS_PER_ENTRY;
    if ((buf = (v_entry *)malloc(bufsize * sizeof(v_entry))) == NULL)
        return (errno);

    bufp = buf;
    val = 0;
    i = 0;
    last_i = 0;
    ones = 0;

    for (p = line; len-- > 0; p++)
    {
        switch (*p)
        {
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
        if (last_i != i && (i % BITS_PER_ENTRY) == 0)
        {
            *bufp = val;
            val = 0;
            bufp++;
            last_i = i;
        }
    }

    /* Store val if it contains any bits. */
    if ((i % BITS_PER_ENTRY) != 0)
        *bufp = val;

    if (nsamples == 0)
        nsamples = i;
    else if (nsamples != i)
    {
        //		fprintf(stderr, "Wrong number of samples. Expected %d got %d\n",
        //		    *nsamples, i);
        /* free(buf); */
        buf = NULL;
    }
    nones = ones;
    ret = buf;
    return (0);
#endif
}

/*
 * Create the truthtable for a default rule -- that is, it captures all samples.
 */
int BitVec::make_default(int len)
{
#ifdef GMP
    // mpz_init2(this->vec, len);	// must be initialized already
    mpz_ui_pow_ui(this->vec, 2, (unsigned long)len);
    mpz_sub_ui(this->vec, this->vec, 1);
    return (0);
#else
    BitVec tt;
    size_t nventry, nbytes;
    unsigned char *c;
    int m;

    nventry = (len + BITS_PER_ENTRY - 1) / BITS_PER_ENTRY;
    nbytes = nventry * sizeof(v_entry);

    if ((c = (unsigned char *)malloc(nbytes)) == NULL)
        return (errno);

    /* Set all full bytes */
    memset(c, BYTE_MASK, nbytes);
    ttp = tt = (BitVec)c;

    /* Fix the last entry so it has 0's for any unused bits. */
    m = len % BITS_PER_ENTRY;
    if (m != 0)
        tt[nventry - 1] = tt[nventry - 1] >> (BITS_PER_ENTRY - m);

    return (0);
#endif
}

/* Create a ruleset. */
Ruleset
Ruleset::ruleset_init(
    int nsamples, const std::vector<int> &idarray, std::vector<Rule> &rules)
{
    Ruleset rs(nsamples);

    BitVec not_captured(nsamples);
    not_captured.make_default(nsamples);

    int cnt = nsamples;
    for (int i = 0; i < idarray.size(); i++)
    {
        rs.entries.emplace_back(idarray[i], 0, nsamples);
        auto cur_re = &rs.entries.back();
        auto cur_rule = &rules[idarray[i]];

        rule_vand(cur_re->captures, not_captured,
                  cur_rule->truthtable, nsamples, cur_re->ncaptured);

        rule_vandnot(not_captured, not_captured,
                     rs.entries[i].captures, nsamples, cnt);
    }
    assert(cnt == 0);
    return rs;
}

/*
 * Save the idarray for this ruleset incase we need to restore it.
 * We don't know how long the idarray currently is so always call
 * realloc, which will do the right thing.
 */
std::vector<int> Ruleset::backup() const
{
    std::vector<int> ids;
    for (auto &entry : this->entries)
        ids.push_back(entry.rule_id);
    return ids;
}

/*
 * When we copy rulesets, we always allocate new structures; this isn't
 * terribly efficient, but it's simpler. If the allocation and frees become
 * too expensive, we can make this smarter.
 */
Ruleset
Ruleset::ruleset_copy()
{
    Ruleset dest(this->n_samples);
    for (auto &entry : this->entries)
    {
        dest.entries.emplace_back(entry.rule_id, entry.ncaptured, this->n_samples);
        entry.captures.rule_copy(dest.entries.back().captures, this->n_samples);
    }
    return dest;
}

/* Reclaim resources associated with a ruleset. */
/*
void
Ruleset::ruleset_destroy()
{
    int j;
    for (auto &entry : this->entries)
        entry.captures.rule_vfree();
    // free(rs);
}
*/

/*
 * Add the specified rule to the ruleset at position ndx (shifting
 * all rules after ndx down by one).
 */
void Ruleset::ruleset_add(std::vector<Rule> &rules, int nrules, int newrule, int ndx)
{
    int i, cnt;
    // RulesetEntry *expand, *cur_re;
    BitVec not_caught(this->n_samples);

    const auto n_rules = this->length();
    this->entries.emplace_back(0, 0, this->n_samples);
    /*
     * Compute all the samples that are caught by rules AFTER the
     * rule we are inserting. Then we'll recompute all the captures
     * from the new rule to the end.
     */
    for (i = ndx; i < n_rules; i++)
        rule_vor(not_caught,
                 not_caught, this->entries[i].captures, this->n_samples, cnt);

    /*
     * Shift later rules down by 1 if necessary. For GMP, what we're
     * doing may be a little sketchy -- we're copying the mpz_t's around.
     */
    if (ndx != n_rules)
    {
        // memmove(rs->entries + (ndx + 1), rs->entries + ndx,
        //     sizeof(RulesetEntry) * (rs->n_rules - ndx));
        for (int i = n_rules; i > ndx; --i)
            std::swap(this->entries[i], this->entries[i - 1]);
    }

    /* Insert and initialize the new rule. */
    this->entries[ndx].rule_id = newrule;

    /*
     * Now, recompute all the captures entries for the new rule and
     * all rules following it.
     */

    for (i = ndx; i < this->length(); i++)
    {
        auto cur_re = &this->entries[i];
        /*
         * Captures for this rule gets anything in not_caught
         * that is also in the rule's truthtable.
         */
        rule_vand(cur_re->captures,
                  not_caught, rules[cur_re->rule_id].truthtable,
                  this->n_samples, cur_re->ncaptured);

        rule_vandnot(not_caught,
                     not_caught, cur_re->captures, this->n_samples, cnt);
    }
    if (cnt != 0)
        throw std::runtime_error("ruleset_add failed");
}

/*
 * Delete the rule in the ndx-th position in the given ruleset.
 */
void Ruleset::ruleset_delete(std::vector<Rule> &rules, int nrules, int ndx)
{
    int i, nset;
    BitVec tmp_vec(this->n_samples);
    // RulesetEntry *old_re, *cur_re;
    const auto n_rules = this->length();

    /* Compute new captures for all rules following the one at ndx.  */
    auto old_re = &this->entries[ndx];

    for (i = ndx + 1; i < n_rules; i++)
    {
        /*
         * My new captures is my old captures or'd with anything that
         * was captured by ndx and is captured by my rule.
         */
        auto cur_re = &this->entries[i];
        rule_vand(tmp_vec, rules[cur_re->rule_id].truthtable,
                  old_re->captures, this->n_samples, nset);
        rule_vor(cur_re->captures, cur_re->captures,
                 tmp_vec, this->n_samples, this->entries[i].ncaptured);

        /*
         * Now remove the ones from old_re->captures that just got set
         * for rule i because they should not be captured later.
         */
        rule_vandnot(old_re->captures, old_re->captures,
                     cur_re->captures, this->n_samples, nset);
    }

    /* Shift up cells if necessary. */
    if (ndx != n_rules - 1)
        // memmove(rs->entries + ndx, rs->entries + ndx + 1,
        //     sizeof(RulesetEntry) * (rs->n_rules - 1 - ndx));
        for (int i = ndx; i < n_rules - 1; ++i)
            std::swap(this->entries[i], this->entries[i + 1]);

    this->entries.pop_back();
    return;
}

/*
 * We create random rulesets for testing and for creating initial proposals
 * in MCMC
 */
Ruleset
Ruleset::create_random_ruleset(int size,
                               int nsamples, int nrules, std::vector<Rule> &rules, gsl_rng *RAND_GSL)
{
    int i, j, next;
    std::vector<int> ids;

    for (i = 0; i < (size - 1); i++)
    {
        next = RANDOM_RANGE(1, (nrules - 1), RAND_GSL);
        do
        {
            next = RANDOM_RANGE(1, (nrules - 1), RAND_GSL);
            /* Check for duplicates. */
            for (j = 0; j < i; j++)
                if (ids[j] == next)
                    break;
        } while (j != i);
        ids.push_back(next);
    }

    /* Always put rule 0 (the default) as the last rule. */
    ids[i] = 0;

    return Ruleset::ruleset_init(nsamples, ids, rules);
}

#define MAX_TRIES 10
/*
 * Given a rule set, pick a random rule (not already in the set).
 */
int Ruleset::pick_random_rule(int nrules, gsl_rng *RAND_GSL) const
{
    int cnt, new_rule;

    cnt = 0;
pickrule:
    if (cnt < MAX_TRIES)
        new_rule = RANDOM_RANGE(1, (nrules - 1), RAND_GSL);
    else
        new_rule = 1 + (new_rule % (nrules - 2));

    for (auto &entry : this->entries)
    {
        if (entry.rule_id == new_rule)
        {
            cnt++;
            goto pickrule;
        }
    }
    return (new_rule);
}

/* dest must exist */
void BitVec::rule_copy(BitVec &dest, int len)
{
#ifdef GMP
    mpz_set(dest.vec, this->vec);
#else
    int i, nentries;

    assert(dest.vec != NULL);
    nentries = (len + BITS_PER_ENTRY - 1) / BITS_PER_ENTRY;
    for (i = 0; i < nentries; i++)
        dest.vec[i] = this->vec[i];
#endif
}

/*
 * Swap rules i and j such that i + 1 = j.
 *	j.captures = j.captures | (i.captures & j.tt)
 *	i.captures = i.captures & ~j.captures
 * 	then swap positions i and j
 */
void Ruleset::ruleset_swap(int i, int j, std::vector<Rule> &rules)
{
    int nset;
    BitVec tmp_vec(this->n_samples);

    assert(i < (this->length() - 1));
    assert(j < (this->length() - 1));
    assert(i + 1 == j);

    /* tmp_vec =  i.captures & j.tt */
    rule_vand(tmp_vec, this->entries[i].captures,
              rules[this->entries[j].rule_id].truthtable, this->n_samples, nset);
    /* j.captures = j.captures | tmp_vec */
    rule_vor(this->entries[j].captures, this->entries[j].captures,
             tmp_vec, this->n_samples, this->entries[j].ncaptured);

    /* i.captures = i.captures & ~j.captures */
    rule_vandnot(this->entries[i].captures, this->entries[i].captures,
                 this->entries[j].captures, this->n_samples, this->entries[i].ncaptured);

    /* Now swap the two entries */
    std::swap(this->entries[i], this->entries[j]);
}

void Ruleset::ruleset_swap_any(int i, int j, std::vector<Rule> &rules)
{
    int cnt, cnt_check, k, temp;
    BitVec caught(this->n_samples);

    if (i == j)
        return;

    assert(i < this->length());
    assert(j < this->length());

    /* Ensure that i < j. */
    if (i > j)
        std::swap(i, j);

    /*
     * The captured arrays before i and after j need not change.
     * We first compute everything caught between rules i and j
     * (inclusive) and then compute the captures array from scratch
     * rules between rule i and rule j, both * inclusive.
     */

    for (k = i; k <= j; k++)
        rule_vor(caught,
                 caught, this->entries[k].captures, this->n_samples, cnt);

    /* Now swap the rules in the ruleset. */
    std::swap(this->entries[i].rule_id, this->entries[j].rule_id);

    cnt_check = 0;
    for (k = i; k <= j; k++)
    {
        /*
         * Compute the items captured by rule k by anding the caught
         * vector with the truthtable of the kth rule.
         */
        rule_vand(this->entries[k].captures, caught,
                  rules[this->entries[k].rule_id].truthtable,
                  this->n_samples, this->entries[k].ncaptured);
        cnt_check += this->entries[k].ncaptured;

        /* Now remove the caught items from the caught vector. */
        rule_vandnot(caught,
                     caught, this->entries[k].captures, this->n_samples, temp);
    }
    assert(temp == 0);
    assert(cnt == cnt_check);
}

/*
 * Dest must have been created.
 */
void rule_vand(BitVec &dest, BitVec &src1, BitVec &src2, int nsamples, int &cnt)
{
#ifdef GMP
    mpz_and(dest.vec, src1.vec, src2.vec);
    cnt = 0;
    cnt = mpz_popcount(dest.vec);
#else
    int i, count, nentries;

    count = 0;
    nentries = (nsamples + BITS_PER_ENTRY - 1) / BITS_PER_ENTRY;
    assert(dest != NULL);
    for (i = 0; i < nentries; i++)
    {
        dest[i] = src1[i] & src2[i];
        count += count_ones(dest[i]);
    }
    cnt = count;
    return;
#endif
}

/* Dest must have been created. */
void rule_vor(BitVec &dest, BitVec &src1, BitVec &src2, int nsamples, int &cnt)
{
#ifdef GMP
    mpz_ior(dest.vec, src1.vec, src2.vec);
    cnt = 0;
    cnt = mpz_popcount(dest.vec);
#else
    int i, count, nentries;

    count = 0;
    nentries = (nsamples + BITS_PER_ENTRY - 1) / BITS_PER_ENTRY;

    for (i = 0; i < nentries; i++)
    {
        dest[i] = src1[i] | src2[i];
        count += count_ones(dest[i]);
    }
    cnt = count;

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
void rule_vandnot(BitVec &dest, BitVec &src1, BitVec &src2, int nsamples, int &ret_cnt)
{
#ifdef GMP
    BitVec tmp(nsamples);
    mpz_com(tmp.vec, src2.vec);
    mpz_and(dest.vec, src1.vec, tmp.vec);
    ret_cnt = 0;
    ret_cnt = mpz_popcount(dest.vec);
#else
    int i, count, nentries;

    nentries = (nsamples + BITS_PER_ENTRY - 1) / BITS_PER_ENTRY;
    count = 0;
    assert(dest != NULL);
    for (i = 0; i < nentries; i++)
    {
        dest[i] = src1[i] & (~src2[i]);
        count += count_ones(dest[i]);
    }

    ret_cnt = count;
    return;
#endif
}

int BitVec::count_ones_vector(int len)
{
#ifdef GMP
    return mpz_popcount(this->vec);
#else
    int cnt = 0, i;
    for (i = 0; i < (len + BITS_PER_ENTRY - 1) / BITS_PER_ENTRY; i++)
    {
        cnt += count_ones(this->vec[i]);
    }
    return cnt;
#endif
}

int count_ones(v_entry val)
{
    int count, i;

    count = 0;
    for (i = 0; i < sizeof(v_entry); i++)
    {
        count += byte_ones[val & BYTE_MASK];
        val >>= 8;
    }
    return (count);
}

/*
 * Find first set bit starting at position start_pos.
 */
int BitVec::rule_ff1(int start_pos, int len)
{
#ifdef GMP
    (void)len;
    return mpz_scan1(this->vec, start_pos);
#else
    int i;
    for (i = start_pos; i < len; i++)
    {
        if (this->rule_isset(i))
            return i;
    }
    return -1;
#endif
}

// void
// ruleset_print(Ruleset *rs, Rule *rules, int detail)
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
// }

// void
// ruleset_entry_print(RulesetEntry *re, int n, int detail)
//{
//	printf("%d captured; \n", re->ncaptured);
//	if (detail)
//		rule_vector_print(re->captures, n);
// }

// void
// rule_print(Rule *rules, int ndx, int n, int detail)
//{
//	Rule *r;
//
//	r = rules + ndx;
//	printf("RULE %d: ( %s ), support=%d, card=%d:",
//	    ndx, r->features, r->support, r->cardinality);
//	if (detail)
//		rule_vector_print(r->truthtable, n);
// }

// void
// rule_vector_print(BitVec v, int n)
//{
// #ifdef GMP
//	mpz_out_str(stdout, 16, v);
//	printf("\n");
// #else
//	int i;
//	for (i = 0; i < n; i++)
//		printf("0x%lx ", v[i]);
//	printf("\n");
// #endif
//
// }

// void
// rule_print_all(Rule *rules, int nrules, int nsamples)
//{
//	int i, n;
//
//	n = (nsamples + BITS_PER_ENTRY - 1) / BITS_PER_ENTRY;
//	for (i = 0; i < nrules; i++)
//		rule_print(rules, i, n, 1);
// }

/*
 * Return 0 if bit e is not set in vector v; return non-0 otherwise.
 */
int BitVec::rule_isset(int e)
{
#ifdef GMP
    return mpz_tstbit(this->vec, e);
#else
    return ((v[e / BITS_PER_ENTRY] & (1 << (e % BITS_PER_ENTRY))) != 0);
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
