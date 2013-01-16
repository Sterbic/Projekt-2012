/*
 * Defines.h
 *
 *  Created on: Dec 4, 2012
 *      Author: Luka
 */

#ifndef DEFINES_H_
#define DEFINES_H_

#include <vector_types.h>

#define VERSION "0.5.4"
#define ALPHA 4
#define STAGE_2_PADDING '?'
#define STAGE_2_PADDING_LAST_ROWS '#'

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

#endif /* DEFINES_H_ */
