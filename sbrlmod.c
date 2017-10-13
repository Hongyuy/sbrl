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
#include "sbrlmod.h"

int debug;

int
usage(void)
{
	(void)fprintf(stderr, "%s %s %s\n",
	    "Usage: sbrlmod [-c chains] [-d debug-level] [-e eta] [-l lambda] ",
	    "[-m model] [-s ruleset-size] [-i iterations] [-t test] [-S seed] ",
	    "train.out [train.label] [test.out] [test.label]");
	return (-1);
}

#define	TEST_MCMC	2

int
main (int argc, char *argv[])
{
	extern char *optarg;
	extern int optind, optopt, opterr;
	int ret, size = DEFAULT_RULESET_SIZE;
	int i, iters, nrules, nsamples, tnum;
	char ch, *modelfile;
	data_t train_data;
	double *p;
	pred_model_t *model;
	rule_t *rules, *labels;
	struct timeval tv_acc, tv_start, tv_end;
	params_t params = {9.0, 3.0, 0.5, {1, 1}, 1000, 11};

	debug = 0;
	p = NULL;
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
			for (i = 0; i < model->rs->n_rules; i++)
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
	if (p != NULL)
		free(p);
	return (0);
}
