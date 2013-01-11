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
	char *reverseSequence;
	int length;
	int paddedLength;

	void createReverse(int maxIndex);
	bool doPadding(int padTo, char withSymb);

public:
	FASTAsequence(char *fileName);

	~FASTAsequence();

	char *getSequenceName();

	char *getSequence();

	int getLength();

	int getPaddedLength();

	bool load();

	char *getReversedSequence(int maxIndex);

	bool doPaddingForColumns(int blocks);

	bool doPaddingForRows();
};

#endif /* FASTA_H_ */
