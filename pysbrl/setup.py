from distutils.core import setup, Extension
import numpy as np

pysbrl = Extension('pysbrl',
                    sources = ['pysbrl.c', '../predict.c', '../train.c', '../rulelib.c'],
		    libraries = ['gmp', 'gsl', 'c', 'gslcblas'],
		    extra_compile_args = ["-DGMP"],
                    include_dirs = [np.get_include(), '../'])

setup (name = 'pysbrl',
       version = '0.1',
       description = 'Python binding of sbrlmod',
       ext_modules = [pysbrl])
