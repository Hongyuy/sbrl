#include <stdlib.h>
#include <stdio.h>
#include <Python.h>
#include "sbrlmod.h"

PyObject *pysbrl_run(PyObject *self, PyObject *args, PyObject *keywds)
{
    PyObject *out_train_data;
    char *out_train_fname = NULL;

    PyObject *label_train_data;
    char *label_train_fname = NULL;

    PyObject *out_test_data = NULL;
    char *out_test_fname = NULL;

    PyObject *label_test_data = NULL;
    char *label_test_fname = NULL;

    char *modelfile = NULL;

    params_t params = {9.0, 3.0, 0.5, {1, 1}, 1000, 11};

    int iters = params.iters;
    int size = DEFAULT_RULESET_SIZE;
    int tnum = TEST_TEST;
    int seed = time(0) + clock();

    static char *kwlist[] = {"out_train", "label_train", "out_test", "label_test", "chains", "debug-level", "eta", "lambda", "model_data", "ruleset-size", "iterations", "test", "seed"};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "OO|OOiiddsiii", kwlist, &out_train_data, &label_train_data, &out_test_data, &label_test_data,
                                    &params.nchains, &debug, &params.eta, &params.lambda, &model_data, &iters, &size, &tnum, &seed))
    {
        return NULL;
    }

	int ret;
	int i, nrules, nsamples;
	char ch;
	data_t train_data;
	double *p;
	pred_model_t *model;
	rule_t *train_rules, *train_labels, *test_rules, *test_labels;
	struct timeval tv_acc, tv_start, tv_end;

    params.iters = iters;

    int size_idx = 9;

    PyObject *size_key = PyUnicode_FromString(kwlist[size_idx]);
    if(!size_key) {
        PyErr_SetString(PyExc_Exception, "Could not create ruleset-size key object");
        return NULL;
    }

    int c = PyDict_Contains(keywds, size_key);
    if(c == -1) {
        PyErr_SetString(PyExc_Exception, "Could not check if keywords");
        return NULL;
    }

    if(c || size != DEFAULT_RULESET_SIZE)
        params.iters = size;

    Py_DECREF(size_key);

	int debug = 0;
	p = NULL;
	rules = labels = NULL;

    srandom((unsigned)seed);

	/*
	 * We treat the label file as a separate ruleset, since it has
	 * a similar format.
	 */
	INIT_TIME(tv_acc);
	START_TIME(tv_start);

    char **fnames[] = {&out_train_fname, &label_train_fname, &out_test_fname, &label_test_fname};
    int *num_rules[] = {&train_nrules, &train_nlabels, &test_nrules, &test_nlabels};
    int *num_samples[] = {&train_nsamples, &train_nsamples_chk, &test_nsamples, &test_nsamples_chk};
    rule_t **rule_lists[] = {&train_rules, &train_labels, &test_rules, &test_labels};
    int inc_default[] = {1, 0, 1, 0};
    PyObject *rule_data[] = {out_train_data, label_train_data, out_test_data, label_test_data};

    for(int i = 0; i < 4; i++)
    {
        if(rule_data[i] == NULL)
            continue;

        if(PyBytes_Check(rule_data[i])) {
            if(!(*fnames[i] = strdup(PyBytes_AsString(rule_data[i]))))
                goto err;
        }
        else if(PyUnicode_Check(rule_data[i])) {
            PyObject* bytes = PyUnicode_AsUTF8String(rule_data[i]);
            if(!bytes)
                goto err;

            else if(!(*fnames[i] = strdup(PyBytes_AsString(bytes))))
                goto err;

            Py_DECREF(bytes);
        }
        else if(PyList_Check(rule_data[i])) {
            if(!PyList_Size(rule_data[i])) {
                PyErr_SetString(PyExc_ValueError, "rule lists must be non-empty");
                goto err;
            }
        }
        else {
            PyErr_SetString(PyExc_TypeError, "Rule lists must be file paths or python lists");
            goto err;
        }

        if(*fnames[i]) {
            if(rules_init(*fnames[i], num_rules[i], num_samples[i], rule_lists[i], inc_default[i]) != 0) {
                PyErr_SetString(PyExc_IOError, "could not load file at path '%s'", *fnames[i]);
                goto err;
            }

            free(*fnames[i]);
            *fnames[i] = NULL;
        } else {
            if(load_list(rule_data[i], num_rules[i], num_samples[i], rule_lists[i], inc_default[i]) != 0)
                goto err;
        }

err:
        if(*fnames[i]) {
            free(*fnames[i]);
            *fnames[i] = NULL;
        }

        for(int j = 0; j < i; j++) {
            rules_free(*rule_lists[i], *num_rules[i], inc_defaults[i]);
            *rule_lists[i] = NULL;
        }

        return NULL;
    }

    if(train_nlabels != 2) {
        PyErr_SetString(PyExc_ValueError, "Number of labels must be 2");
        return NULL;
    }
    if(train_nsamples != train_nsamples_chk) {
        PyErr_SetString(PyExc_ValueError, "Number of samples in out and label files must match");
    }

	END_TIME(tv_start, tv_end, tv_acc);
	REPORT_TIME("Initialize time", "per rule", tv_end, train_nrules);

	if (debug)
		printf("%d rules %d samples\n\n", train_nrules, train_nsamples);

	if (debug > 100)
		rule_print_all(train_rules, train_nrules, train_nsamples);

	if (debug > 100) {
		printf("Labels for %d samples\n\n", train_nsamples);
		rule_print_all(train_labels, train_nsamples, train_nsamples);
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
			run_experiment(iters, size, train_nsamples, train_nrules, train_rules);
			break;
		case 2:
		case 3:
			train_data.rules = train_rules;
			train_data.labels = train_labels;
			train_data.nrules = train_nrules;
			train_data.nsamples = train_nsamples;
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
			ruleset_print(model->rs, train_rules, 0);
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



			p = test_model(argv[2],
				   argv[3], model, &params);

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
