/*
 * FASTA.cpp
 *
 *  Created on: Nov 7, 2012
 *      Author: Luka
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Defines.h"
#include "FASTA.h"

FASTAsequence::FASTAsequence(char *fileName) {
	this->fileName = fileName;
	length = 0;
	sequence = NULL;
	sequenceName = NULL;
}

FASTAsequence::~FASTAsequence() {
	if(sequenceName != NULL) free(sequenceName);
	if(sequence != NULL) free(sequence);
}

char *FASTAsequence::getSequenceName() {
	return sequenceName;
}

char *FASTAsequence::getSequence() {
	return sequence;
}

int FASTAsequence::getLength() {
	return length;
}

int FASTAsequence::getPaddedLength() {
	return paddedLength;
}

bool FASTAsequence::load() {
	FILE *f = fopen(fileName, "r");

	if(f == NULL) return false;

	char c[300 + 1];
	memset(c, 0, 301);
	fgets(c, 200, f);

	if(c[0] != '>') return false;

	int size = strlen(c) - 1;
	sequenceName = (char *)malloc(size);
	memset(sequenceName, 0, size);
	if(sequenceName == NULL) return false;
	for(int i = 0; i < size - 1; i++) {
		sequenceName[i] = c[i + 1];
	}

	long current = size + 1;
	fseek(f, 0L, 2);
	long seqLength = ftell(f) - current;

	char *seq = (char *)malloc(seqLength);
	if(seq == NULL) return false;
	memset(seq, 0, seqLength);

	fseek(f, 0L, 0);
	fgets(c, 300, f);

	int i = 0;
	char base = 0;
	while(fscanf(f, "%c", &base) != EOF) {
		if(base == '\n' || base == '\r') continue;
		seq[i++] = base;
	}

	length = i;
	paddedLength = length;

	fclose(f);

	seq = (char *)realloc(seq, length + 1);
	if(seq == NULL) return false;
	sequence = seq;

	return true;
}

bool FASTAsequence::doPadding(int padTo, char withSymb) {
	int C = (length + padTo - 1) / padTo;
	if(length % C == 0)
		return true;

	int padding = C - length % C;

	sequence = (char *)realloc(sequence, length + padding * sizeof(char) + 1);
	if(sequence == NULL)
		return false;

	for(int i = 0; i < padding; i++) {
		sequence[length + i] = withSymb;
	}
	sequence[length + padding] = '\0';

	paddedLength += padding;

	return true;
}

bool FASTAsequence::doPaddingForColumns(int blocks) {
	return doPadding(blocks, '#');
}

bool FASTAsequence::doPaddingForRows() {
	return doPadding(ALPHA, '$');
}
