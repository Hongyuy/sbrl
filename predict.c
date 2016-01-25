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
/* Use scalable bayesian rule list to make predictions. */

#include <assert.h>
#include <errno.h>
#include <float.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include "mytime.h"
#include "rule.h"

extern int debug;

double
get_accuracy(ruleset_t *rs,
    double *theta, rule_t *test_rules, rule_t *test_labels, params_t *params)
{
	int *idarray = NULL;
	int nwrong = 0;
	ruleset_t *rs_test;
	VECTOR v0;

	rule_vinit(rs->n_samples, &v0);
	ruleset_backup(rs, &idarray);
	ruleset_init(rs->n_rules,
	    test_rules[0].support, idarray, test_rules, &rs_test);
	ruleset_print(rs_test, test_rules, 0);

	if (debug > 10) {
		for (int j = 0; j < rs_test->n_rules; j++)
			printf("theta[%d] = %f\n", j, theta[j]);
	}

	rule_vinit(rs->n_samples, &v0);
	for (int j = 0; j < rs_test->n_rules; j++) {
		int n1_correct = 0, n0_correct = 0;

		if (theta[j] >= params->threshold) {
			rule_vand(v0, rs_test->rules[j].captures,
			    test_labels[1].truthtable, rs_test->n_samples,
			    &n1_correct);
			nwrong += abs(n1_correct - rs_test->rules[j].ncaptured);
		} else {
			rule_vand(v0, rs_test->rules[j].captures,
			    test_labels[0].truthtable, rs_test->n_samples,
			    &n0_correct);
			nwrong += abs(n0_correct - rs_test->rules[j].ncaptured);
		}
		if (debug > 10)
			printf("rules[%d] captures %d%s %d, n1=%d,%s %.6f\n",
			    j, rs_test->rules[j].ncaptured,
			    "samples, correct n0=", n0_correct, n1_correct,
			    "test Probability=", (n0_correct + n1_correct) * 1.0
			    / rs_test->rules[j].ncaptured);
	}
	if (debug > 10) {
		printf("ntotal = %d,  n0 = %d,  n1 = %d\n",
		    rs_test->n_samples, test_labels[0].support,
		    test_labels[1].support);
		printf("#wrong predictions = %d,  #total predictions = %d\n",
		    nwrong, rs_test->n_samples);
	}

	free(idarray);
	ruleset_destroy(rs_test);
	return 1.0 - ((float)nwrong / rs_test->n_samples);
}

double *
predict(data_t *data, pred_model_t *pred_model, params_t *params)
{
	double *prob;
	int cnt, rule_id;
    
	prob = calloc(data->nsamples, sizeof(double));
	if (prob == NULL)
		return NULL;

	for (int i = 0; i < data->nsamples; i++)
		prob[i] = 0.0;

	for (int j=0; j < pred_model->rs->n_rules; j++) {
		cnt = 0;
		rule_id = pred_model->rs->rules[j].rule_id;
		for (int i = 0; i < data->nsamples; i++) {
			if (prob[i] < 1e-5 &&
			    rule_isset(data->rules[rule_id].truthtable, i)) {
				prob[i] = pred_model->theta[j];
				cnt++;
			}
		}
		printf("Rule %d captures %d of %d samples\n",
		    rule_id, cnt, data->nsamples);
	}

	if (debug > 10)
		for (int i = 0; i < data->nsamples; i++)
		    printf("%.6f\n", prob[i]);

	if (debug)
		printf("test accuracy = %.6f \n",
		    get_accuracy(pred_model->rs, pred_model->theta,
			data->rules, data->labels, params));

	return prob;
}
