AUTOMAKE_OPTIONS = foreign
VERSION=1.0

AM_CPPFLAGS = -DGMP -Wall -I. -I/usr/local/include -I/opt/local/include $(INCLUDES)
AM_LDFLAGS = -L/opt/local/lib -L/usr/local/lib -lgsl -lgmp -lm -lc

bin_PROGRAMS = testprog
testprog_SOURCES = predict.c rulelib.c testprog.c train.c mytime.h
include_HEADERS = rule.h
dist_pkgdata_DATA = ttt_test.out ttt_test.label ttt_train.out ttt_train.label


