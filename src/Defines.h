#ifndef DEFINES_H_
#define DEFINES_H_

#include <vector_types.h>

#define VERSION "0.6.2"
#define ALPHA 4
#define STAGE_2_PADDING '?'
#define STAGE_2_PADDING_LAST_ROWS '#'

#define TRACEBACK_LAST_SHORT 1
#define TRACEBACK_LAST_LONG 2
#define TRACEBACK_SHORT_LONG 3

typedef struct {
    int score;
    int row;
    int column;
} alignmentScore;

typedef struct {
	int *diagonal;
	int2 *left0;
	int2 *left1;
	int2 *left2;
	int2 *left3;
} VerticalBuffer;

typedef struct {
	int2 *up;
} HorizontalBuffer;

typedef struct {
	VerticalBuffer vBuffer;
	HorizontalBuffer hBuffer;
} GlobalBuffer;

#endif
