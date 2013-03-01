# include <cuda.h>
# include <iostream>
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <vector>

# include "Visualisation.h"

void visualOutput(char *firstSequence, char *secondSequence, const char *firstFilename,
                  const char *secondFilename, int2 startPoint, int2 endPoint) {
    std::vector<int2> firstSeqGaps;
    std::vector<int2> secondSeqGaps;
    int2 gapInfo;
    int gapBegin;
    int gapLength;

    printf("\tCTRL: in visual output\n");

    FILE *firstGapList = fopen(firstFilename, "rb");
    FILE *secondGapList = fopen(secondFilename, "rb");

    rewind(firstGapList);
    rewind(secondGapList);

    while (fread(&gapBegin, sizeof(int), 1, firstGapList) == 1 &&
           fread(&gapLength, sizeof(int), 1, firstGapList) == 1) {

        //printf("\tsuccessfully read from first file: %d %d\n", gapBegin, gapLength);
        gapInfo.x = gapBegin;
        gapInfo.y = gapLength;
        firstSeqGaps.push_back(gapInfo);
    }

    while (fread(&gapBegin, sizeof(int), 1, secondGapList) == 1 &&
           fread(&gapLength, sizeof(int), 1, secondGapList) == 1) {

        //printf("\tsuccessfully read from second file: %d %d\n", gapBegin, gapLength);
        gapInfo.x = gapBegin;
        gapInfo.y = gapLength;
        secondSeqGaps.push_back(gapInfo);
    }

    fclose(firstGapList);
    fclose(secondGapList);

    int firstOffset, secondOffset;
    firstOffset = startPoint.x;
    secondOffset = startPoint.y;

    printf("\n\tVisual representation of the alignment:\n\n");

    int firstInGap = 0;
    int secondInGap = 0;
    int done = 0;
    std::vector<int2>::iterator firstIt = firstSeqGaps.begin();
    std::vector<int2>::iterator secondIt = secondSeqGaps.begin();
    char visual[3][PRINT_ROW_LENGTH + 5];


    while(true) {
        memset(visual, 0, 3 * (PRINT_ROW_LENGTH + 5) * sizeof(char));

        for (int i = 0; i < PRINT_ROW_LENGTH; ++i) {
            if(firstInGap) {
                visual[0][i] = '-';
                firstInGap--;
            }
            else if ((*firstIt).x == firstOffset) {
                firstInGap = (*firstIt).y;
                --i;
                ++firstIt;
            }
            else {
                visual[0][i] = *(firstSequence + firstOffset);
                ++firstOffset;
            }

            if(firstOffset > endPoint.x) {
                ++done;
                break;
            }
        }

        for (int i = 0; i < PRINT_ROW_LENGTH; ++i) {
            if(secondInGap) {
                visual[2][i] = '-';
                secondInGap--;
            }
            else if ((*secondIt).x == secondOffset) {
                secondInGap = (*secondIt).y;
                --i;
                ++secondIt;
            }
            else {
                visual[2][i] = *(secondSequence + secondOffset);
                ++secondOffset;
            }

            if(secondOffset > endPoint.y) {
                ++done;
                break;
            }
        }

        for (int i = 0; i < PRINT_ROW_LENGTH; ++i) {
            if (visual[0][i] == 0)
                visual[1][i] = 0;
            else if (visual[0][i] == visual[2][i])
                visual[1][i] = '|';
            else
                visual[1][i] = ' ';
        }

        for(int i = 0; i < 3; ++i) {
            printf("\t\t");
            for (int j = 0; j < PRINT_ROW_LENGTH && (visual[i][j] != 0); ++j) {
                printf("%c", visual[i][j]);
            }
            printf("\n");
        }
        printf("\n");

        if (done != 0)
            break;
    }

}
/*
int main() {
    FILE *file1;
    FILE *bin1;
    FILE *file2;
    FILE *bin2;

    int gapBegin, gapLength;

    file1 = fopen("gap1.txt", "r");
    bin1 = fopen("gap1.bin", "wb");
    while(!feof(file1)) {
        fscanf(file1, "%d %d", &gapBegin, &gapLength);
        printf("%d %d\t", gapBegin, gapLength);
        fwrite((void *)&gapBegin, sizeof(int), 1, bin1);
        fwrite((void *)&gapLength, sizeof(int), 1, bin1);
    }
    fclose(file1);
    fclose(bin1);

    printf("\nCTRL: made gap1.bin\n");

    file2 = fopen("gap2.txt", "r");
    bin2 = fopen("gap2.bin", "wb");
    while(!feof(file2)) {
        fscanf(file2, "%d %d", &gapBegin, &gapLength);
        printf("%d %d\t", gapBegin, gapLength);
        fwrite((void *)&gapBegin, sizeof(int), 1, bin2);
        fwrite((void *)&gapLength, sizeof(int), 1, bin2);
    }
    fclose(file2);
    fclose(bin2);

    printf("\nCTRL: made gap2.bin\n");

    char firstSequence[100] = {0};
    char secondSequence[100] = {0};

    file1 = fopen("seq1.txt", "r");

    int i = 0;
    while(!feof(file1))
        fscanf(file1, " %c", &firstSequence[i++]);
    fclose(file1);

    std::cout << "CTRL: first\t" << firstSequence << std::endl;

    file2 = fopen("seq2.txt", "r");
    i = 0;
    while(!feof(file2))
        fscanf(file2, " %c", &secondSequence[i++]);
    fclose(file2);

    std::cout << "CTRL: second\t" << secondSequence << std::endl;

    int2 startPoint, endPoint;

    startPoint.x = startPoint.y = 0;
    endPoint.x = 48;
    endPoint.y = 45;

    printf("CTRL: entering visualOutput...\n");
    visualOutput(firstSequence, secondSequence, "gap1.bin", "gap2.bin", startPoint, endPoint);

    return 0;
}
*/
