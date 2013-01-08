/*
 * RowBuilder.h
 *
 *  Created on: Jan 7, 2013
 *      Author: Luka
 */

#ifndef ROWBUILDER_H_
#define ROWBUILDER_H_

#include "SWutils.h"

class RowBuilder {
private:
	int firstLen;
	int secondLen;
	int dumpRows;
	LaunchConfig *config;
	int2 *hb;

public:
	RowBuilder(int firstLen, int secondLen, LaunchConfig *config);

	~RowBuilder();

	void dumpShort(int2 *devHBuffer, int dk);

	void dumpLong(int2 *devHBuffer, int dk);

	int getRowHeight();
};

#endif /* ROWBUILDER_H_ */
