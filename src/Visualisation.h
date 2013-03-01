#ifndef VISUALISATION_H_
#define VISUALISATION_H_

# include <cuda.h>

# define PRINT_ROW_LENGTH 50


void visualOutput(char *firstSequence, char *secondSequence, FILE *firstGapList,
                  FILE *secondGapList, int2 startPoint, int2 endPoint);



#endif /* VISUALISATION_H_ */
