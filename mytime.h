/*
 * Copyright (c) 2016 Hongyu Yang, Cynthia Rudin, Margo Seltzer, and
 * The President and Fellows of Harvard College
 * 
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include <sys/time.h>
#define	INIT_TIME(TV) {		\
	TV.tv_sec = 0;		\
	TV.tv_usec = 0;		\
}

/* Add TV1 into TV2 */
#define ADD_TIME(TV1, TV2) {		\
	TV2.tv_sec += TV1.tv_sec;	\
	TV2.tv_usec += TV1.tv_usec;	\
}

#define START_TIME(TV) gettimeofday(&TV, NULL)
#define	END_TIME(TV1, TV2, ACC_TV) {			\
	gettimeofday(&TV2, NULL);			\
	TV2.tv_sec  -= TV1.tv_sec;			\
	if (TV2.tv_usec > TV1.tv_usec)			\
		TV2.tv_usec -= TV1.tv_usec;		\
	else {						\
		TV2.tv_sec--;				\
		TV2.tv_usec += 1000000 - TV1.tv_usec;	\
	}						\
	ADD_TIME(TV2, ACC_TV);				\
}

#define REPORT_TIME(S, T, TV, N) {						\
	float _tmp;							\
	TV.tv_sec += (TV.tv_usec / 1000000);				\
	TV.tv_usec %= 1000000;						\
	_tmp = TV.tv_sec + (float)TV.tv_usec / 1000000;			\
	printf("%s: Elapsed time %7.4f\t%7.4f %s\n", S, _tmp, _tmp / N, T);	\
}
