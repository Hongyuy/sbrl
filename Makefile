# Makefile should be able to build three things:
# Testprog: test program used to verify that the rule library, training, and prediction
# 	are working correctly
# Lib: a library containing the training and prediction code for use in C programs
# RLib: An R package containing the C code; prediction to be done in R.

TARGET = analyze
OBJECTS = rulelib.o analyze.o
LOBJECTS = rulelib.o train.o
TOBJECTS = $(LOBJECTS) testprog.o predict.o
EXTRA = makedata.pyc
INCLUDES = -I. -I/opt/local/include 

# Put this here so we can specify something like -DGMP to switch between
# representations.
CC = cc
CFLAGS = -g $(INCLUDES)
#-DGMP
LIBS = -L/opt/local/lib  -lc -lgsl -lgmp

rulelib-gmp.o: rulelib.c
	$(CC) $(CFLAGS) -c -DGMP -o rulelib-gmp.o rulelib.c

analyze-gmp.o: analyze.c
	$(CC) $(CFLAGS) -c -DGMP -o analyze-gmp.o analyze.c

analyze-gmp: analyze-gmp.o rulelib-gmp.o
	$(CC) -g -o analyze-gmp analyze-gmp.o rulelib-gmp.o $(LIBS)

$(TARGET) : $(OBJECTS)
	$(CC) -o $(TARGET) $(OBJECTS) $(LIBS)

%.o : %.c
	$(CC) $(CFLAGS) -c $<

clean:
	/bin/rm -rf $(TARGET) $(OBJECTS) $(EXTRA)
