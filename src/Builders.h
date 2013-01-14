#ifndef BUILDERS_H_
#define BUILDERS_H_

#include "SWutils.h"

class RowBuilder {
private:
	int firstLen;
	int secondLen;
	int rowHeight;
	LaunchConfig *config;
	int2 *hb;

public:
	RowBuilder(int firstLen, int secondLen, LaunchConfig *config);

	~RowBuilder();

	void dumpShort(int2 *devHBuffer, int dk);

	void dumpLong(int2 *devHBuffer, int dk);

	int getRowHeight();
};

class ColumnBuilder {

};

#endif /* BUILDERS_H_ */
