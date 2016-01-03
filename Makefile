TARGET = analyze
OBJECTS = rulelib.o analyze.o
EXTRA = makedata.pyc
INCLUDES = -I. -I/opt/local/include 

# Put this here so we can specify something like -DGMP to switch between
# representations.
CC = cc
CFLAGS = -g $(INCLUDES)
#-DGMP
LIBS = -L/opt/local/lib  -lc -lgsl 
#Version = -mmacosx-version-min=10.5 

$(TARGET) : $(OBJECTS)
	$(CC) -o $(TARGET) $(OBJECTS) $(LIBS) $(Version)

%.o : %.c
	$(CC) $(CFLAGS) -c $<

clean:
	/bin/rm $(TARGET) $(OBJECTS) $(EXTRA)
