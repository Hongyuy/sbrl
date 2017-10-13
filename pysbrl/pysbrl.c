#include <stdlib.h>
#include <stdio.h>

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include <numpy/arrayobject.h>

#include "sbrlmod.h"
#include "utils.h"

#define BUFSZ  512


static PyObject *pysbrl_tofile(PyObject *self, PyObject *args)
{
    PyObject *list;
    const char *fname;

    if(!PyArg_ParseTuple(args, "Os", &list, &fname))
        return NULL;

    if(!PyList_Check(list)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a list");
        return NULL;
    }

    PyObject *tuple, *vector;
    char *features;

    npy_intp list_len = PyList_Size(list);

    FILE *fp;
    if(!(fp = fopen(fname, "w")))
        return NULL;

    for(Py_ssize_t i = 0; i < list_len; i++) {
        if(!(tuple = PyList_GetItem(list, i)))
            goto error;

        if(!PyTuple_Check(tuple)) {
            PyErr_SetString(PyExc_TypeError, "Array members must be tuples");
            goto error;
        }

        if(!PyArg_ParseTuple(tuple, "sO", &features, &vector))
            goto error;

        fprintf(fp, "%s ", features);

        int type = PyArray_TYPE((PyArrayObject*)vector);
        if(PyArray_NDIM((PyArrayObject*)vector) != 1 && (PyTypeNum_ISINTEGER(type) || PyTypeNum_ISBOOL(type))) {
            PyErr_SetString(PyExc_TypeError, "Each rule truthable must be a 1-d array of integers or booleans");
            goto error;
        }

        PyArrayObject *clean = (PyArrayObject*)PyArray_FromAny(vector, PyArray_DescrFromType(NPY_BYTE), 0, 0, NPY_ARRAY_CARRAY | NPY_ARRAY_ENSURECOPY | NPY_ARRAY_FORCECAST, NULL);
        if(clean == NULL) {
            PyErr_SetString(PyExc_Exception, "Could not cast array to byte carray");
            goto error;
        }

        char *data = PyArray_BYTES(clean);
        npy_intp b_len = PyArray_SIZE(clean);

        for(npy_intp j = 0; j < b_len-1; j++) {
            fprintf(fp, "%d ", !!data[j]);
        }

        fprintf(fp, "%d\n", !!data[b_len-1]);

        Py_DECREF(clean);
    }

    fclose(fp);

    Py_INCREF(Py_None);
    return Py_None;

error:
    fclose(fp);
    return NULL;
}

static PyObject *pysbrl_tolist(PyObject *self, PyObject *args)
{
    const char *fname;

    if(!PyArg_ParseTuple(args, "s", &fname))
        return NULL;

    rule_t *rules;
    int nrules, nsamples;

    if(rules_init(fname, &nrules, &nsamples, &rules, 0) != 0) {
        PyErr_SetString(PyExc_ValueError, "Could not load rule file");
        return NULL;
    }

    PyObject *list = PyList_New(nrules);

    PyObject *res = fill_list(list, rules, 0, nrules, nsamples);
    if(!res) {
        Py_XDECREF(list);
        list = NULL;
    }

    rules_free(rules, nrules, 0);

    return list;
}

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

    debug = 0;

    static char *kwlist[] = {"out_train", "label_train", "out_test", "label_test", "chains", "debug_level", "eta", "lambda", "modelfile", "ruleset_size", "iterations", "tnum", "seed"};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "OO|OOiiddsiiii", kwlist, &out_train_data, &label_train_data, &out_test_data, &label_test_data,
                                    &params.nchain, &debug, &params.eta, &params.lambda, &modelfile, &size, &iters, &tnum, &seed))
    {
        return NULL;
    }

    PyObject *final_ret = Py_None;

	int i, train_nrules, train_nlabels, train_nsamples, train_nsamples_chk,
               test_nrules, test_nlabels, test_nsamples, test_nsamples_chk;
	data_t train_data;
	double *p;
	pred_model_t *model;
	rule_t *train_rules = NULL, *train_labels = NULL, *test_rules = NULL, *test_labels = NULL;
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
        PyErr_SetString(PyExc_Exception, "Could not check if keywords contain ruleset-size");
        Py_XDECREF(size_key);
        return NULL;
    }

    if(c || size != DEFAULT_RULESET_SIZE)
        params.iters = size;

    Py_DECREF(size_key);

	p = NULL;

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
    int inc_defaults[] = {1, 0, 1, 0};
    PyObject *rule_data[] = {out_train_data, label_train_data, out_test_data, label_test_data};

    for(int i = 0; i < 4; i++)
    {
        if(rule_data[i] == NULL)
            continue;

        if(PyBytes_Check(rule_data[i])) {
            if(!(*fnames[i] = strdup(PyBytes_AsString(rule_data[i])))) {
                PyErr_SetString(PyExc_ValueError, "Error with file path strings");
                final_ret = NULL;
                goto end;
            }
        }
        else if(PyUnicode_Check(rule_data[i])) {
            PyObject* bytes = PyUnicode_AsUTF8String(rule_data[i]);
            if(!bytes) {
                PyErr_SetString(PyExc_ValueError, "Error with file path strings");
                final_ret = NULL;
                goto end;
            }

            else if(!(*fnames[i] = strdup(PyBytes_AsString(bytes)))) {
                PyErr_SetString(PyExc_ValueError, "Error with file path strings");
                final_ret = NULL;
                goto end;
            }

            Py_DECREF(bytes);
        }
        else if(PyList_Check(rule_data[i])) {
            if(!PyList_Size(rule_data[i])) {
                PyErr_SetString(PyExc_ValueError, "rule lists must be non-empty");
                final_ret = NULL;
                goto end;
            }
        }
        else {
            PyErr_SetString(PyExc_TypeError, "Rule lists must be file paths or python lists");
            final_ret = NULL;
            goto end;
        }

        if(*fnames[i]) {
            if(rules_init(*fnames[i], num_rules[i], num_samples[i], rule_lists[i], inc_defaults[i]) != 0) {
                char buf[BUFSZ];
                sprintf(buf, "could not load file at path '%s'", *fnames[i]);
                PyErr_SetString(PyExc_IOError, buf);
                final_ret = NULL;
                goto end;
            }

            free(*fnames[i]);
            *fnames[i] = NULL;
        } else {
            if(load_list(rule_data[i], num_rules[i], num_samples[i], rule_lists[i], inc_defaults[i]) != 0) {
                final_ret = NULL;
                goto end;
            }
        }
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
			printf("size: %d  nrules: %d\n", size, train_nrules);
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
				PyErr_SetString(PyExc_Exception, "Error: Train failed");
                final_ret = NULL;
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

            if(tnum != 3)
                break;
            else if(test_rules == NULL || test_labels == NULL) {
                PyErr_SetString(PyExc_ValueError, "Test data must be provided to test the generated model");
                final_ret = NULL;
                break;
            }

        	int *idarray = NULL;
        	ruleset_t *test_rs = NULL, *tmp_rs;

        	/* Make an array of the rules comprising this model. */
        	if ((ruleset_backup(model->rs, &idarray)) != 0) {
                if (idarray != NULL)
            		free (idarray);

                PyErr_SetString(PyExc_Exception, "Could not create test rulelist backup");
                final_ret = NULL;
                break;
            }

        	/* Create new ruleset with test data. */
        	if (ruleset_init(model->rs->n_rules, test_nsamples, idarray, test_rules, &test_rs) != 0) {
                if (idarray != NULL)
            		free (idarray);
            	if (test_rs != NULL)
            		ruleset_destroy(test_rs);

                PyErr_SetString(PyExc_Exception, "Could not create test ruleset");
                final_ret = NULL;
                break;
            }

        	tmp_rs = model->rs;
        	model->rs = test_rs;
        	p = predict(model, test_labels, &params);
        	model->rs = tmp_rs;

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
			if (modelfile == NULL || test_rules == NULL || test_labels == NULL) {
				PyErr_SetString(PyExc_ValueError, "A model and test set are required to test a model");
                final_ret = NULL;
                break;
			}
			// Read Modelfile
			model = read_model(modelfile, test_nrules, test_rules, test_nsamples);
			p = predict(model, test_labels, &params);
			break;
		default:
			PyErr_SetString(PyExc_ValueError, "tnum must be between 1 and 4, inclusive");
            final_ret = NULL;
            break;
	}

end:
    if(p)
        free(p);

    for(int i = 0; i < 4; i++) {
        if(*fnames[i]) {
            free(*fnames[i]);
            *fnames[i] = NULL;
        }

        if(*rule_lists[i]) {
            rules_free(*rule_lists[i], *num_rules[i], inc_defaults[i]);
            *rule_lists[i] = NULL;
        }
    }

    if(final_ret == Py_None)
        Py_INCREF(Py_None);

    return final_ret;
}

static PyMethodDef pysbrlMethods[] = {
    {"run", (PyCFunction)pysbrl_run, METH_VARARGS | METH_KEYWORDS },
    {"tolist", (PyCFunction)pysbrl_tolist, METH_VARARGS },
    {"tofile", (PyCFunction)pysbrl_tofile, METH_VARARGS },
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION > 2

static struct PyModuleDef pysbrlModule = {
    PyModuleDef_HEAD_INIT,
    "pysbrl",
    "Python binding to sbrlmod",
    -1,
    pysbrlMethods
};

PyMODINIT_FUNC PyInit_pysbrl(void)
{
    import_array();

    return PyModule_Create(&pysbrlModule);
}

#else

PyMODINIT_FUNC initpysbrl(void)
{
    import_array();

    Py_InitModule("pysbrl", pysbrlMethods);
}

#endif
