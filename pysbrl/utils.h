#pragma once

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <gmp.h>

#define BUFSZ  512

typedef enum gate_en {
    G_AND = 0,
    G_OR,
    G_NOT,
    G_COUNT
} GATE;

typedef struct contains {
    int *data;
    int len;
} contains_t;

const char *gate_names[] = {"&", "|", "&!"};

int generate_rule(rule_t *dest, rule_t src1, rule_t src2, GATE gate, int nsamples, double min_support, int no_repeat, contains_t *dest_contains, contains_t *src1_contains, contains_t *src2_contains)
{
    rule_vinit(nsamples, &dest->truthtable);

    int support;

    switch(gate)
    {
        case G_AND:
            rule_vand(dest->truthtable, src1.truthtable, src2.truthtable, nsamples, &support);
            break;
        case G_OR:
            rule_vor(dest->truthtable, src1.truthtable, src2.truthtable, nsamples, &support);
            break;
        case G_NOT:
            rule_vandnot(dest->truthtable, src1.truthtable, src2.truthtable, nsamples, &support);
            break;
        default:
            break;
    };

    if((double)support / (double)nsamples < min_support) {
        rule_vfree(&dest->truthtable);
        return 1;
    }

    if(no_repeat) {
        dest_contains->len = src1_contains->len + src2_contains->len;
        dest_contains->data = malloc(sizeof(int) * (dest_contains->len));
        memcpy(dest_contains->data, src1_contains->data, sizeof(int) * src1_contains->len);
        memcpy(dest_contains->data + src1_contains->len, src2_contains->data, sizeof(int) * src2_contains->len);
    }

    dest->support = support;

    int len1 = strlen(src1.features);
    int len2 = strlen(src2.features);
    int len3 = strlen(gate_names[gate]);
    dest->features = malloc(len1 + len2 + len3 + 3);

    strcpy(dest->features, src2.features);
    strcat(dest->features, gate_names[gate]);
    strcat(dest->features, "(");
    strcat(dest->features, src1.features);
    strcat(dest->features, ")");

    //printf("%s\n", dest->features);

    return 0;
}

PyObject* fill_list(PyObject *obj, rule_t *rules, int start, int nrules, int nsamples)
{
    PyObject *vector, *tuple;

    if(!obj) {
        PyErr_SetString(PyExc_MemoryError, "List is null");
        return NULL;
    }

    for(int i = 0; i < nrules; i++)
    {
#ifdef GMP
        int leading_zeros = nsamples - mpz_sizeinbase(rules[i].truthtable, 2);
        char* bits = malloc(nsamples + 1);

        mpz_get_str(bits + leading_zeros, 2, rules[i].truthtable);

        for(int j = 0; j < leading_zeros; j++)
            bits[j] = 0;

        for(int j = leading_zeros; j < nsamples; j++)
            bits[j] = bits[j] - '0';

        int num_bits = nsamples;
#else
        char* bits = malloc(nsamples + 1);

        int nentry = (nsamples + BITS_PER_ENTRY - 1) / BITS_PER_ENTRY;
        v_entry mask;
        for(int j = 0; j < nentry; j++) {
            mask = 1;

            for(int k = 0; k < BITS_PER_ENTRY; k++) {
                bits[j * BITS_PER_ENTRY + k] = !!(rules[i].truthtable[j] & mask);
                mask <<= 1;
            }
        }

        bits[nsamples] = '\0';

        int num_bits = nsamples;
#endif

        //rule_print(rules, i, nsamples, 1);
        //printf("%s\n", bits);

        if(!(vector = PyArray_FromString(bits, num_bits, PyArray_DescrFromType(NPY_BOOL), -1, NULL))) {
            PyErr_SetString(PyExc_ValueError, "Could not load bitvector");
            Py_DECREF(obj);
            return NULL;
        }

        free(bits);

        if(!(tuple = Py_BuildValue("sO", rules[i].features, vector))) {
            Py_DECREF(obj);
            return NULL;
        }

        if(PyList_SetItem(obj, start + i, tuple) != 0) {
            Py_XDECREF(tuple);
            PyErr_SetString(PyExc_Exception, "Could not insert tuple into list");
            Py_XDECREF(obj);
            return NULL;
        }
    }

    return obj;
}

int load_list(PyObject *list, int *nrules, int *nsamples, rule_t **rules_ret, int add_default_rule)
{
    rule_t* rules = NULL;

    if(!PyList_Check(list)) {
        PyErr_SetString(PyExc_TypeError, "Data must be a python list");
        goto error;
    }

    Py_ssize_t list_len = PyList_Size(list);

    int ntotal_rules = list_len + (add_default_rule ? 1 : 0);

    rules = malloc(sizeof(rule_t) * ntotal_rules);
    if(!rules) {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate rule array");
        goto error;
    }

    int samples_cnt = 0;

    PyObject* tuple;
    PyObject* vector;
    char* features;

    int rule_idx = ntotal_rules - list_len;
    for(Py_ssize_t i = 0; i < list_len; i++) {
        if(!(tuple = PyList_GetItem(list, i)))
            goto error;

        if(!PyTuple_Check(tuple)) {
            PyErr_SetString(PyExc_TypeError, "Array members must be tuples");
            goto error;
        }

        if(!PyArg_ParseTuple(tuple, "sO", &features, &vector))
            goto error;

        int features_len = strlen(features);
        rules[rule_idx].features = malloc(features_len + 1);
        strcpy(rules[rule_idx].features, features);
        //rule[rule_idx].features[features_len] = '\0';

        rules[rule_idx].cardinality = 1;

        if(!PyArray_Check(vector)) {
            PyErr_SetString(PyExc_TypeError, "The second element of each tuple must be a numpy array");
            goto error;
        }

        int type = PyArray_TYPE((PyArrayObject*)vector);
        if(PyArray_NDIM((PyArrayObject*)vector) != 1 && (PyTypeNum_ISINTEGER(type) || PyTypeNum_ISBOOL(type))) {
            PyErr_SetString(PyExc_TypeError, "Each rule truthable must be a 1-d array of integers or booleans");
            goto error;
        }

        PyArrayObject* clean = (PyArrayObject*)PyArray_FromAny(vector, PyArray_DescrFromType(NPY_BYTE), 0, 0, NPY_ARRAY_CARRAY | NPY_ARRAY_ENSURECOPY | NPY_ARRAY_FORCECAST, NULL);
        if(clean == NULL) {
            PyErr_SetString(PyExc_Exception, "Could not cast array to byte carray");
            goto error;
        }

        char* data = PyArray_BYTES(clean);
        npy_intp b_len = PyArray_SIZE(clean);

        for(npy_intp j = 0; j < b_len; j++)
            data[j] = '0' + !!data[j];

        data[b_len] = '\0';

        if(ascii_to_vector(data, b_len, &samples_cnt, &rules[rule_idx].support, &rules[rule_idx].truthtable) != 0) {
            PyErr_SetString(PyExc_Exception, "Could not load bit vector");
            Py_DECREF(clean);
            goto error;
        }

        Py_DECREF(clean);

        rule_idx++;
    }

    if(add_default_rule) {
        rules[0].support = samples_cnt;
		rules[0].features = "default";
		rules[0].cardinality = 0;
		if (make_default(&rules[0].truthtable, samples_cnt) != 0) {
            PyErr_SetString(PyExc_Exception, "Could not make default rule");
		    goto error;
        }
    }

    *nrules = ntotal_rules;
    *rules_ret = rules;
    *nsamples = samples_cnt;

    return 0;

error:
    if(rules != NULL) {
        for (int i = 1; i < rule_idx; i++) {
            free(rules[i].features);
#ifdef GMP
            mpz_clear(rules[i].truthtable);
#else
            free(rules[i].truthtable);
#endif
        }
        free(rules);
    }
    *rules_ret = NULL;
    *nrules = 0;
    *nsamples = 0;
    return 1;
}
