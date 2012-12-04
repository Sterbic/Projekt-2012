/*
 * Buffers.h
 *
 *  Created on: Dec 3, 2012
 *      Author: Luka
 */

#ifndef BUFFERS_H_
#define BUFFERS_H_

#include "Defines.h"

typedef struct {

} VerticalBuffer;

typedef struct {

} HorizontalBuffer;

__device__ void getRowBuffer(int i, char *sequence, char *seqBuffer) {
	for(int a = 0; a < ALPHA; a++)
		seqBuffer[a] = sequence[i + a];
}

__device__ void getColumnBuffer(int j, char *sequence, char *seqBuffer) {

}


#endif /* BUFFERS_H_ */
