AC_INIT([sbrl], 1.0)

CC=`"${R_HOME}/bin/R" CMD config CC`
CFLAGS=`"${R_HOME}/bin/R" CMD config CFLAGS`
CPPFLAGS=`"${R_HOME}/bin/R" CMD config CPPFLAGS`
CXX=`"${R_HOME}/bin/R" CMD config CXX`
CXXFLAGS=`"${R_HOME}/bin/R" CMD config CXXFLAGS`
LDFLAGS=`"${R_HOME}/bin/R" CMD config LDFLAGS`
MAKE=`"${R_HOME}/bin/R" CMD config MAKE`
AC_PROG_CC
AC_PROG_CPP
AC_PROG_CXX

SBRL_CFLAGS=$CFLAGS
SBRL_CPPFLAGS=$CPPFLAGS
SBRL_CXXFLAGS=$CXXFLAGS
SBRL_LDFLAGS=$LDFLAGS

## Use gsl-config to find arguments for compiler and linker flags
##
## Check for non-standard programs: gsl-config(1)
AC_PATH_PROG([GSL_CONFIG], [gsl-config])
## If gsl-config was found, let's use it
if test "${GSL_CONFIG}" != ""; then
    # Use gsl-config for header and linker arguments
    GSL_CFLAGS=`${GSL_CONFIG} --cflags`
    GSL_LIBS=`${GSL_CONFIG} --libs`
else
    AC_MSG_ERROR([gsl-config not found, is GSL1 installed?])
fi
 
# Now substitute these variables in src/Makevars.in to create src/Makevars
AC_SUBST(GSL_CFLAGS)
AC_SUBST(GSL_LIBS)
AC_SUBST(SBRL_CFLAGS)
AC_SUBST(SBRL_CPPFLAGS)
AC_SUBST(SBRL_CXXFLAGS)
AC_SUBST(SBRL_LDFLAGS)

AC_CHECK_HEADER([gmp.h], , 
   [AC_MSG_ERROR([gmp.h not found, is GMP installed?])])
AC_CHECK_LIB(gmp, __gmpz_init, ,
   [AC_MSG_ERROR([gmp library not found, is GMP installed?])])

# seems GMP only needs -DGMP(from sbrl) and -lgmp
#AC_SUBST(GMP_CFLAGS, ${CFLAGS})
#AC_SUBST(GMP_LIBS, ${LIBS})

# AC_OUTPUT(src/Makevars)
AC_CONFIG_FILES(src/Makevars)
AC_CONFIG_COMMANDS()
AC_OUTPUT

