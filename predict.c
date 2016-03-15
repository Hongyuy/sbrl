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
    double *theta, rule_t *test_labels, params_t *params)
{
	int j, nwrong = 0;
	VECTOR v0;

	if (debug > 10) {
		for (j = 0; j < rs->n_rules; j++)
			printf("theta[%d] = %f\n", j, theta[j]);
	}

	rule_vinit(rs->n_samples, &v0);
	for (j = 0; j < rs->n_rules; j++) {
		int n1_correct = 0, n0_correct = 0;

		if (theta[j] >= params->threshold) {
			rule_vand(v0, rs->rules[j].captures,
			    test_labels[1].truthtable, rs->n_samples,
			    &n1_correct);
			nwrong += rs->rules[j].ncaptured - n1_correct;
		} else {
			rule_vand(v0, rs->rules[j].captures,
			    test_labels[0].truthtable, rs->n_samples,
			    &n0_correct);
			nwrong += rs->rules[j].ncaptured - n0_correct;
		}
		if (debug > 10)
			printf("rules[%d] captures %d%s %d, n1=%d,%s %.6f\n",
			    j, rs->rules[j].ncaptured,
			    "samples, correct n0=", n0_correct, n1_correct,
			    "test Probability=", (n0_correct + n1_correct) * 1.0
			    / rs->rules[j].ncaptured);
	}
	if (debug > 10) {
		printf("ntotal = %d,  n0 = %d,  n1 = %d\n",
		    rs->n_samples, test_labels[0].support,
		    test_labels[1].support);
		printf("#wrong predictions = %d,  #total predictions = %d\n",
		    nwrong, rs->n_samples);
	}

	return 1.0 - ((float)nwrong / rs->n_samples);
}

double *
predict(pred_model_t *pred_model, rule_t *labels, params_t *params)
{
	double *prob;
	int i, j, rule_id, sample;
	ruleset_t *rs;
    
	prob = calloc(pred_model->rs->n_samples, sizeof(double));
	if (prob == NULL)
		return NULL;

	for (i = 0; i < pred_model->rs->n_samples; i++)
		prob[i] = 0.0;

	rs = pred_model->rs;
	for (j = 0; j < rs->n_rules; j++) {
		sample = 0;
		rule_id = rs->rules[j].rule_id;
		/*
		 * Iterate over the captured vector finding each sample
		 * captured by this rule; assign it the probability associated
		 * with the rule.
		 */
		for (i = 0; i < rs->rules[j].ncaptured; i++) {
			sample = rule_ff1(rs->rules[j].captures,
			    sample, rs->n_rules);
			prob[sample] = pred_model->theta[j];
			sample++;
		}
		printf("Rule %d captures %d of %d samples\n",
		    rule_id, rs->rules[j].ncaptured, pred_model->rs->n_samples);
	}

	if (debug > 10)
		for (i = 0; i < pred_model->rs->n_samples; i++)
		    printf("%.6f\n", prob[i]);

	if (debug)
		printf("test accuracy = %.6f \n",
		    get_accuracy(pred_model->rs, pred_model->theta,
			labels, params));

	return prob;
}
