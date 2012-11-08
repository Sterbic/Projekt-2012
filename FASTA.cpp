/*
 * FASTA.cpp
 *
 *  Created on: Nov 7, 2012
 *      Author: Luka
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

bool FASTAsequence::load() {
	FILE *f = fopen(fileName, "r");

	if(f == NULL) return false;

	char c[300 + 1];
	memset(c, 0, 301);
	fgets(c, 300, f);

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
	memset(seq, 0, seqLength);
	if(seq == NULL) return false;

	fseek(f, current, 1);

	int i = 0;
	char base = 0;
	while(fscanf(f, "%c", &base) != EOF) {
		if(base == '\n' || base == '\r') continue;
		seq[i++] = base;
	}

	fclose(f);

	length = i;

	seq = (char *)realloc(seq, length + 1);
	if(seq == NULL) return false;
	sequence = seq;

	return true;
}

