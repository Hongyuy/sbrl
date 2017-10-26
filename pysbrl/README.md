# Python binding of sbrlmod

## Installation

Run `python setup.py install` or `python3 setup.py install` in this directory
(pysbrl) depending on what version of python you wish to use pysbrl with.

## Usage

First, you have to include the module:
`import pysbrl`

There are currently only three functions in the module: `pysbrl.run`, `pysbrl.tolist`, and `pysbrl.tofile`

### pysbrl.run

#### Parameters

`pysbrl.run` is the most important function, which runs the actual algorithm. Its usage is as follows:

`pysbrl.run(out_train, label_train, [keywords])`

Where out_train and label_train can either be file paths (strings) or a list
of tuples, where each tuple contains two elements, the first being a string
for the rule features and the second being a numpy array of ints or bools
of length nsamples, representing the captured bitvector of each rule.

The optional keywords are:

| Keywords                          | Description 
| ---                               | ---
| out_test (string or list)         | test set rules
| label_test (string or list)       | test set labels
| chains (int)                      | chains option for sbrlmod 
| debug_level (int)                 | verbosity level (higher is more verbose)
| eta (float)                       | eta option for sbrlmod
| lambda (float)                    | lambda option for sbrlmod
| modelfile (string)                | file containing model to test, if that is what is to be done
| ruleset_size (integer)            | size of ruleset
| iterations (integer)              | number of times to run the program
| tnum (int)                        | the type of test to run: 1 is for testing basic rule manipulation, 2 is to test the training set, 3 is to train model and then run test data on it, and 4 is to read in previous model (modelfile keyword) and run test data on it
| seed (integer)                    | if randomization will be used, you can provide this as the initial seed

#### Return value

Currently, pysbrl.run returns none


### pysbrl.tolist

#### Parameters

This function only takes one required argument: a string for the path to
the rule file

#### Return value

returns a list of tuples (in the format detailed above) describing the rules, not containing a default rule

### pysbrl.tofile

#### Parameters

This function only takes two required arguments: the first is a list of tuples describing a list of rules (as specified above), and the second is a path to a file. The function then outputs the list into the file specified by the second argument, the exact opposite of pycorels.tolist.

#### Return value

None
