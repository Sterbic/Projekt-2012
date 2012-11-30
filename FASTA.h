/*
 * FASTA.h
 *
 *  Created on: Nov 7, 2012
 *      Author: Luka
 */

#ifndef FASTA_H_
#define FASTA_H_

class FASTAsequence {
private:
	char *fileName;
	char *sequenceName;
	char *sequence;
	long length;

	bool doPadding(int padTo, char withSymb);

public:
	FASTAsequence(char *fileName);

	~FASTAsequence();

	char *getSequenceName();

	char *getSequence();

	int getLength();

	bool load();

	bool doPaddingForColumns(int BLOCKS_PER_GRID);

	bool doPaddingForRows(int ALPHA);
};

#endif /* FASTA_H_ */
