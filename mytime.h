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
	TV2.tv_sec  -= TV1.tv_sec;				\
	if (TV2.tv_usec > TV1.tv_usec)			\
		TV2.tv_usec -= TV1.tv_usec;			\
	else {						\
		TV2.tv_sec++;				\
		TV2.tv_sec += 1000000 - TV1.tv_usec;		\
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
