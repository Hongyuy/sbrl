AUTOMAKE_OPTIONS = foreign
MAJOR_VERSION=1
MINOR_VERSION=0
VERSION=$(MAJOR_VERSION).$(MINOR_VERSION)
LIBTOOL_VERSION=$(MAJOR_VERSION):$(MINOR_VERSION)

ACLOCAL_AMFLAGS= -I m4
AM_CPPFLAGS = -DGMP -Wall -I. -I/usr/local/include -I/opt/local/include $(INCLUDES)
AM_LDFLAGS = -L/opt/local/lib -L/usr/local/lib -lgsl -lgmp -lm -lc

# Support both static and dynamic builds
bin_PROGRAMS = sbrlmod
# ADD THIS TO PREVIOUS LINE FOR STATIC BUILD: sbrlmod_static
include_HEADERS = rule.h
dist_pkgdata_DATA = ttt_test.out ttt_test.label ttt_train.out ttt_train.label

# Static build -- not built by default, but it works
sbrlmod_static_SOURCES = sbrlmod.c mytime.h
sbrlmod_static_LDADD = libsbrlmod.la
sbrlmod_static_LDFLAGS = -static

# Dynamic build
sbrlmod_SOURCES = sbrlmod.c mytime.h
sbrlmod_LDADD = libsbrlmod.la

lib_LTLIBRARIES = libsbrlmod.la
libsbrlmod_la_SOURCES = rulelib.c train.c predict.c
libsbrlmod_la_LDFLAGS = -version-info $(LIBTOOL_VERSION)


