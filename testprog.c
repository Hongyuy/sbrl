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
 * Scalable ruleset test program.
 * 
 * This consists of two types of tests:
 * 1. Low level tests to make sure the basic rule library is working
 * 2. A run of MCMC on known data to make sure the library pieces are all working.
 *
 * Debug levels:
 * > 100 print everything imaginable
 * > 10 Trace general execution
 * > 1  Main parameters and results
 */

#include <assert.h>
#include <errno.h>
#include <float.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include "mytime.h"
#include "rule.h"

/* Convenient macros. */
#define DEFAULT_RULESET_SIZE  4
#define DEFAULT_RULE_CARDINALITY 3
#define NLABELS 2

int load_data(const char *, const char *, int *, int *, rule_t **, rule_t **);
pred_model_t *read_model(const char *, int, rule_t *, int);
void run_experiment(int, int, int, int, rule_t *);
double *test_model(const char *, const char *, pred_model_t *, params_t *);
int write_model(const char *, pred_model_t *);

int debug;

int
usage(void)
{
	(void)fprintf(stderr, "%s %s %s\n",
	    "Usage: testprog [-c chains] [-d debug-level] [-e eta] [-l lambda] ",
	    "[-m model] [-s ruleset-size] [-i iterations] [-t test] [-S seed] ",
	    "train.out [train.label] [test.out] [test.label]");
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
	int iters, nrules, nsamples, tnum;
	char ch, *modelfile;
	data_t train_data;
	double *p;
	pred_model_t *model;
	rule_t *rules, *labels;
	struct timeval tv_acc, tv_start, tv_end;
	params_t params = {9.0, 3.0, 0.5, {1, 1}, 1000, 11};

	debug = 0;
	rules = labels = NULL;
	iters = params.iters;
	tnum = TEST_TEST;
	modelfile = NULL;
	srandom(time(0) + clock());
	while ((ch = getopt(argc, argv, "c:d:e:i:l:m:s:S:t:")) != EOF)
		switch (ch) {
		case 'c':
			params.nchain = atoi(optarg);
			break;
		case 'd':
			debug = atoi(optarg);
			break;
		case 'e':
			params.eta = strtod(optarg, NULL);
			break;
		case 'i':
			iters = atoi(optarg);
			params.iters = iters;
			break;
		case 'l':
			params.lambda = strtod(optarg, NULL);
			break;
		case 'm':
			modelfile = optarg;
			break;
		case 's':
			size = atoi(optarg);
			params.iters = size;
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

	if (argc < 2)
		return (usage());

	/*
	 * We treat the label file as a separate ruleset, since it has
	 * a similar format.
	 */
	INIT_TIME(tv_acc);
	START_TIME(tv_start);
	if ((ret = load_data(argv[0],
	    argv[1], &nsamples, &nrules, &rules, &labels)) != 0)
	    	return (ret);
	END_TIME(tv_start, tv_end, tv_acc);
	REPORT_TIME("Initialize time", "per rule", tv_end, nrules);
    
	if (debug)
		printf("%d rules %d samples\n\n", nrules, nsamples);

	if (debug > 100)
		rule_print_all(rules, nrules, nsamples);
    
	if (debug > 100) {
		printf("Labels for %d samples\n\n", nsamples);
		rule_print_all(labels, nsamples, nsamples);
   	} 
	/*
	 * Testing:
	 * 1. Test basic rule manipulation.
	 * 2. Test training.
	 * 3. Train model and then run test.
	 * 4. Read in previous model and test on it.
	 */
	switch(tnum) {
		case 1:
			run_experiment(iters, size, nsamples, nrules, rules);
			break;
		case 2:
		case 3:
			train_data.rules = rules;
			train_data.labels = labels;
			train_data.nrules = nrules;
			train_data.nsamples = nsamples;
			INIT_TIME(tv_acc);
			START_TIME(tv_start);
			model = train(&train_data, 0, 0, &params);
			END_TIME(tv_start, tv_end, tv_acc);
			REPORT_TIME("Time to train", "", tv_end, 1);

			if (model == NULL) {
				fprintf(stderr, "Error: Train failed\n");
				break;
			}

			printf("\nThe best rulelist for %d MCMC chains is: ",
			    params.nchain); 
			ruleset_print(model->rs, rules, 0);
			for (int i = 0; i < model->rs->n_rules; i++)
				printf("%d, %.8f\n",
				    model->rs->rules[i].rule_id,
				    model->theta[i]);

			printf("Lambda = %.6f\n", params.lambda);
			printf("Eta = %.6f\n", params.eta);
			printf("Alpha[0] = %d\n", params.alpha[0]);
			printf("Alpha[1] = %d\n", params.alpha[1]);
			printf("Number of chains = %d\n", params.nchain);
			printf("Iterations = %d\n", params.iters);

			if (tnum == 3) {
				/* Now test the model */
				if (argc < 4) {
					usage();
					break;
				}
				p = test_model(argv[2],
				    argv[3], model, &params);
			}

			if (modelfile != NULL)
				(void)write_model(modelfile, model);

			ruleset_destroy(model->rs);
			free(model->theta);
			free(model);
			break;
		case 4:	/*
			 * Test model from previous run; requires model
			 * file as well as testdata.
			 */
			if (modelfile == NULL || argc < 2) {
				usage();
				break;
			}
			// Read Modelfile
			model = read_model(modelfile, nrules, rules, nsamples);
			p = predict(model, labels, &params);
			break;
		default:
			usage();
			break;
	}

	rules_free(rules, nrules, 1);
	rules_free(labels, 2, 0);
}

/* ========= Simple test utility routines ======= */

/*
 * Given a rule set, pick a random rule (not already in the set) and
 * add it at the ndx-th position.
 */
int
add_random_rule(rule_t *rules, int nrules, ruleset_t *rs, int ndx)
{
	int r;
	r = pick_random_rule(nrules, rs);
	if (debug > 100)
		printf("Selected %d for new rule\n", r);

	return ruleset_add(rules, nrules, &rs, r, ndx);
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
			ruleset_print(rs, rules, (debug > 10));
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
				ruleset_swap(rs, k - 1, k, rules);
				if (debug)
					ruleset_print(rs, rules, (debug > 100));
			}
		END_TIME(tv_start, tv_end, tv_acc);
		REPORT_TIME("analyze", "per swap", tv_end, ((size-1) * (size-1)));

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
				ruleset_print(rs, rules, (debug > 100));
			add_random_rule(rules, nrules, rs, j);
			if (debug)
				ruleset_print(rs, rules, (debug > 100));
		}
		END_TIME(tv_start, tv_end, tv_acc);
		REPORT_TIME("analyze", "per add/del", tv_end, ((size-1) * 2));
		ruleset_destroy(rs);
	}

}

int
load_data(const char *data_file, const char *label_file,
    int *ret_samples, int *ret_nrules, rule_t **rules, rule_t **labels)
{
	int nlabels, ret, samples_chk;

	/* Load data. */
	if ((ret = rules_init(data_file, ret_nrules, ret_samples, rules, 1)) != 0)
		return (ret);

	/* Load labels. */
	if ((ret = 
	    rules_init(label_file, &nlabels, &samples_chk, labels, 0)) != 0) {
		free (*rules);
		return (ret);
	}

    	assert(nlabels == 2);
	assert(samples_chk == *ret_samples);
	return (0);
}

double *
test_model(const char *data_file,
    const char *label_file, pred_model_t *model, params_t *params)
{
	double *p;
	int *idarray, nsamples, nrules;
	rule_t *rules, *labels;
	ruleset_t *test_rs, *tmp_rs;

	idarray = NULL;
	test_rs = NULL;
	p = NULL;

	/* Make an array of the rules comprising this model. */
	if ((ruleset_backup(model->rs, &idarray)) != 0)
		goto err;

	/* Load test data. */
	if (load_data(data_file,
	    label_file, &nsamples, &nrules, &rules, &labels) != 0)
		goto err;

	/* Create new ruleset with test data. */
	if (ruleset_init(model->rs->n_rules,
	    nsamples, idarray, rules, &test_rs) != 0)
		goto err;

	tmp_rs = model->rs;
	model->rs = test_rs;
	p = predict(model, labels, params);
	model->rs = tmp_rs;

err:
	if (idarray != NULL)
		free (idarray);
	if (test_rs != NULL)
		ruleset_destroy(test_rs);

	return (p);
}

int
write_model(const char *file, pred_model_t *model)
{
	FILE *fi;

	if ((fi = fopen(file, "w")) == NULL) {
		fprintf(stderr, "%s %s: %s\n",
		    "Unable to write model file", file, strerror(errno));
		return (-1);
	} 
	for (int i = 0; i < model->rs->n_rules; i++)
		fprintf(fi, "%d,%.8f\n",
		    model->rs->rules[i].rule_id, model->theta[i]);
	fclose(fi);
	return (0);
}

pred_model_t *
read_model(const char *file, int nrules, rule_t *rules, int nsamples)
{
	double theta, *theta_array;
	FILE *fi;
	int i, id, *idarray, nslots, tmp;
	pred_model_t *model;
	ruleset_t *rs;

	i = nslots = 0;
	model = NULL;
	rs = NULL;
	idarray = NULL;
	theta_array = NULL;

	if ((fi = fopen(file, "r")) == NULL) {
		fprintf(stderr, "%s %s: %s\n",
		    "Unable to read model file", file, strerror(errno));
		return (NULL);
	} 

	while ((tmp = fscanf(fi, "%d,%lf\n", &id, &theta)) == 2) {
		if (debug > 10)
			printf("tmp = %d id = %d theta = %f\n", tmp, id, theta);
		if (i >= nslots) {
			nslots += 50;
			idarray = realloc(idarray, nslots * sizeof(int));
			theta_array =
			    realloc(theta_array, nslots * sizeof(double));
			if (idarray == NULL || theta_array == NULL) {
				fprintf(stderr,
				    "Unable to malloc space: %s\n",
				        strerror(ENOMEM));
				goto err;
			}
		}

		idarray[i] = id;
		theta_array[i++] = theta;
	}

	/* Create the ruleset. */
	if (ruleset_init(i, nsamples, idarray, rules, &rs) != 0)
		goto err;
	/* Create the model. */
	if ((model = malloc(sizeof(pred_model_t))) == NULL)
		goto err;
	model->rs = rs;
	model->theta = theta_array;
	model->confIntervals = 0;
	
	if (0) {
err:
		if (rs != NULL)
			ruleset_destroy(rs);
	}
	if (idarray != NULL)
		free(idarray);
	if (theta_array != NULL)
		free(theta_array);
	fclose(fi);
	return (model);
}
