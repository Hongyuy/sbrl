#pragma once

#include "mytime.h"
#include "rule.h"

/* Convenient macros. */
#define DEFAULT_RULESET_SIZE  4
#define DEFAULT_RULE_CARDINALITY 3
#define NLABELS 2

#define	TEST_TEST	1

int debug;

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

	return (ruleset_add(rules, nrules, &rs, r, ndx));
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
	int i;

	if ((fi = fopen(file, "w")) == NULL) {
		fprintf(stderr, "%s %s: %s\n",
		    "Unable to write model file", file, strerror(errno));
		return (-1);
	}
	for (i = 0; i < model->rs->n_rules; i++)
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
