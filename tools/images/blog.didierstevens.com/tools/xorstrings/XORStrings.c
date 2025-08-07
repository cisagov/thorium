/*
	2013/03/08
	XORStrings V0.0.1, look for XOR, ROL or SHIFT encoded strings in a file
	Source code put in public domain by Didier Stevens, no Copyright
	https://DidierStevens.com
	Use at your own risk

	Shortcommings, or todo's ;-)
	- file must fit in memory
	- unicode support

	History:
		2013/02/26: start, fork from XORSearch V1.8
		2013/03/01: added l and t options
		2013/03/02: added c option
		2013/03/03: added ParseHexArg
		2013/03/08: added options o and k; refactoring
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>

#define MAXPATH 1024

#define OPR_XOR 0
#define OPR_ROL 1
#define OPR_SHIFT 2

char *apszOperations[3] = {"XOR", "ROL", "SHIFT"};

struct RESULT
{
	int iOperation;
	int iKey;
	int iCountStrings;
	int iCountCharacters;
	int iMaxStringLength;
	long lIndexMaxString;
};

struct RESULT asResults[512];

long ParseNumericArg(const char *szArg)
{
	char *szError;
	long lResult;

	lResult = strtol(szArg, &szError, 0);
	if (*szError != '\0' || lResult == LONG_MIN || lResult == LONG_MAX)
		return -1;
	else
		return lResult;
}

long ParseHexArg(const char *szArg)
{
	long lResult = 0;

	while (*szArg != '\0')
	{
		if (*szArg >= '0' && *szArg <= '9')
			lResult = lResult * 0x10 + *szArg - '0';
		else if (*szArg >= 'a' && *szArg <= 'f')
			lResult = lResult * 0x10 + *szArg - 'a' + 10;
		else if (*szArg >= 'A' && *szArg <= 'F')
			lResult = lResult * 0x10 + *szArg - 'A' + 10;
		else
			return -1;
		if (lResult == LONG_MIN || lResult == LONG_MAX)
			return -1;
		szArg++;
	}
	return lResult;
}

int ParseArgs(int argc, char **argv, char **ppcFile, int *piSave, int *piUnicode, int *piDump, int *piMaxSort, int *piMinimumStringLength, int *piTerminator, int *piCSV, int *piOperation, int *piKey)
{
	int iIterArgv;
	int iCountParameters;
	char *pcFlags;
	int iFlagMinimumStringLength;
	int iFlagTerminator;
	int iFlagOperation;
	int iFlagKey;

	iCountParameters = 0;
	iFlagMinimumStringLength = 0;
	iFlagTerminator = 0;
	iFlagOperation = 0;
	iFlagKey = 0;
	*piSave = 0;
	*piUnicode = 0;
	*piDump = 0;
	*piMaxSort = 0;
	*piMinimumStringLength = -1;
	*piTerminator = -1;
	*piCSV = 0;
	*piOperation = -1;
	*piKey = -1;
	for (iIterArgv = 1; iIterArgv < argc; iIterArgv++)
	{
		if (argv[iIterArgv][0] == '-')
		{
			if (iFlagMinimumStringLength || iFlagTerminator || iFlagOperation || iFlagKey)
				return 1;
			pcFlags = argv[iIterArgv] + 1;
			while (*pcFlags)
				switch (*pcFlags++)
				{
					case 's':
						*piSave = 1;
						break;
					case 'u':
						*piUnicode = 1;
						break;
					case 'd':
						*piDump = 1;
						break;
					case 'm':
						*piMaxSort = 1;
						break;
					case 'l':
						iFlagMinimumStringLength = 1;
						break;
					case 't':
						iFlagTerminator = 1;
						break;
					case 'c':
						*piCSV = 1;
						break;
					case 'o':
						iFlagOperation = 1;
						break;
					case 'k':
						iFlagKey = 1;
						break;
					default:
						return 1;
				}
		}
		else if (iFlagMinimumStringLength)
		{
			*piMinimumStringLength = ParseNumericArg(argv[iIterArgv]);
			if (*piMinimumStringLength < 1)
				return 1;
			iFlagMinimumStringLength = 0;
		}
		else if (iFlagTerminator)
		{
			if (!strncmp(argv[iIterArgv], "0x", 2))
				*piTerminator = ParseHexArg(argv[iIterArgv] + 2);
			else
				*piTerminator = ParseNumericArg(argv[iIterArgv]);
			if (*piTerminator == -1)
				return 1;
			iFlagTerminator = 0;
		}
		else if (iFlagOperation)
		{
			int iIter;

			for (iIter = 0; iIter <= OPR_SHIFT; iIter++)
				if (!strcmp(argv[iIterArgv], apszOperations[iIter]))
					*piOperation = iIter;
			if (*piOperation == -1)
				return 1;
			iFlagOperation = 0;
		}
		else if (iFlagKey)
		{
			if (!strncmp(argv[iIterArgv], "0x", 2))
				*piKey = ParseHexArg(argv[iIterArgv] + 2);
			else
				*piKey = ParseNumericArg(argv[iIterArgv]);
			if (*piKey == -1)
				return 1;
			iFlagKey = 0;
		}
		else if (iCountParameters == 0)
		{
			*ppcFile = argv[iIterArgv];
			iCountParameters++;
		}
		else
			iCountParameters++;
	}
	if (iCountParameters != 1)
		return 1;
	else
		return 0;
}

void XOR(unsigned char *pcBuffer, long lSize, unsigned char cXOR)
{
	unsigned char *pcBufferEnd;

	pcBufferEnd = pcBuffer + lSize;
	while (pcBuffer < pcBufferEnd)
		*pcBuffer++ ^= cXOR;
}

void ROL(unsigned char *pcBuffer, long lSize)
{
	unsigned char *pcBufferEnd;

	pcBufferEnd = pcBuffer + lSize;
	while (pcBuffer < pcBufferEnd)
	{
		*pcBuffer = *pcBuffer << 1 | *pcBuffer >> 7;
		pcBuffer++;
	}
}

void SHIFTL(unsigned char *pcBuffer, long lSize)
{
	unsigned char *pcBufferEnd;
	unsigned char ucFirstBit;

	pcBufferEnd = pcBuffer + lSize;
	ucFirstBit = *pcBuffer >> 7;
	while (pcBuffer < pcBufferEnd - 1)
	{
		*pcBuffer = *pcBuffer << 1 | *(pcBuffer + 1) >> 7;
		pcBuffer++;
	}
	*(pcBufferEnd - 1) = *(pcBufferEnd - 1) << 1 | ucFirstBit;
}

void SHIFTR(unsigned char *pcBuffer, long lSize)
{
	unsigned char *pcBufferIter;
	unsigned char ucLastBit;

	pcBufferIter = pcBuffer + lSize - 1;
	ucLastBit = *pcBufferIter & 0x01;
	while (pcBuffer < pcBufferIter)
	{
		*pcBufferIter = *pcBufferIter >> 1 | *(pcBufferIter - 1) << 7;
		pcBufferIter--;
	}
	*pcBuffer = *pcBuffer >> 1 | ucLastBit << 7;
}

void SaveFile(const char *pcFile, const char *sOperation, unsigned char ucXOR, void *pBuffer, long lSize)
{
	char szFileNameSave[MAXPATH];
	FILE *fOut;

	snprintf(szFileNameSave, MAXPATH, "%s.%s.%02X", pcFile, sOperation, ucXOR);
	if ((fOut = fopen(szFileNameSave, "wb")) == NULL)
		fprintf(stderr, "error opening file %s\n", szFileNameSave);
	else
	{
		if (fwrite(pBuffer, lSize, 1, fOut) != 1)
			fprintf(stderr, "error writing file %s\n", szFileNameSave);
		fclose (fOut);
	}
}

int IsPrintable(unsigned char ucByte)
{
	return ucByte >= 0x20 && ucByte <= 0x7E;
}

void StringsAnalysis(const unsigned char *pucBuffer, long lSize, struct RESULT *psResult, int iMinimumStringLength, unsigned char ucTerminator)
{
	long lIter;
	long lStringStart = -1;
	int iCountStrings = 0;
	int iCountCharacters = 0;
	int iMaxStringLength = 0;
	long lIndexMaxString= -1;

	for (lIter = 0; lIter < lSize; lIter++)
	{
		if (pucBuffer[lIter] != ucTerminator && IsPrintable(pucBuffer[lIter]))
		{
			if (lStringStart == -1)
				lStringStart = lIter;
		}
		else if (lStringStart != -1)
		{
			if (pucBuffer[lIter] == ucTerminator && lIter - lStringStart >= iMinimumStringLength)
			{
				iCountStrings++;
				if (iMaxStringLength < lIter - lStringStart)
				{
					iMaxStringLength = lIter - lStringStart;
					lIndexMaxString = lStringStart;
				}
				iCountCharacters += lIter - lStringStart;
			}
			lStringStart = -1;
		}
	}
	psResult->iCountStrings = iCountStrings;
	psResult->iCountCharacters = iCountCharacters;
	psResult->iMaxStringLength = iMaxStringLength;
	psResult->lIndexMaxString = lIndexMaxString;
}

char *OPRString(int iOperation)
{
	if (iOperation <= OPR_SHIFT)
		return apszOperations[iOperation];
	else
		return "ERROR";
}

int CompareStringCounts(const void *a, const void *b)
{
	const struct RESULT *resulta = (struct RESULT *) a;
	const struct RESULT *resultb = (struct RESULT *) b;

	return (resulta->iCountStrings > resultb->iCountStrings) - (resulta->iCountStrings < resultb->iCountStrings);
}

int CompareStringMaxLengths(const void *a, const void *b)
{
	const struct RESULT *resulta = (struct RESULT *) a;
	const struct RESULT *resultb = (struct RESULT *) b;

	return (resulta->iMaxStringLength > resultb->iMaxStringLength) - (resulta->iMaxStringLength < resultb->iMaxStringLength);
}

void StringsPrint(const unsigned char *pucBuffer, long lSize, int iMinimumStringLength, unsigned char ucTerminator)
{
	long lIter;
	long lStringStart = -1;

	for (lIter = 0; lIter < lSize; lIter++)
	{
		if (pucBuffer[lIter] != ucTerminator && IsPrintable(pucBuffer[lIter]))
		{
			if (lStringStart == -1)
				lStringStart = lIter;
		}
		else if (lStringStart != -1)
		{
			if (pucBuffer[lIter] == ucTerminator && lIter - lStringStart >= iMinimumStringLength)
				printf("%.*s\n", (int)(lIter - lStringStart), pucBuffer + lStringStart);
			lStringStart = -1;
		}
	}
}

void DoStringsAnalysis(void *pBuffer, long lSize, const char *pcArgFile, int iFlagSave, int iFlagDump, int iFlagMaxSort, int iMinimumStringLength, int iTerminator, int iFlagCSV)
{
	unsigned char ucOPRIter;
	int iCountResults = 0;
	int iIter;

	ucOPRIter = 0;

	do
	{
		XOR((unsigned char *) pBuffer, lSize, ucOPRIter);

		asResults[iCountResults].iOperation = OPR_XOR;
		asResults[iCountResults].iKey = ucOPRIter;
		StringsAnalysis(pBuffer, lSize, &asResults[iCountResults++], iMinimumStringLength, (unsigned char) iTerminator);
		if (iFlagSave && asResults[iCountResults - 1].iCountStrings > 0)
			SaveFile(pcArgFile, OPRString(OPR_XOR), ucOPRIter, pBuffer, lSize);

		XOR((unsigned char *) pBuffer, lSize, ucOPRIter);
	} while (++ucOPRIter);

	for (ucOPRIter = 1; ucOPRIter < 8; ucOPRIter++)
	{
		ROL((unsigned char *) pBuffer, lSize);

		asResults[iCountResults].iOperation = OPR_ROL;
		asResults[iCountResults].iKey = ucOPRIter;
		StringsAnalysis(pBuffer, lSize, &asResults[iCountResults++], iMinimumStringLength, (unsigned char) iTerminator);
		if (iFlagSave && asResults[iCountResults - 1].iCountStrings > 0)
			SaveFile(pcArgFile, OPRString(OPR_ROL), ucOPRIter, pBuffer, lSize);
	}
	ROL((unsigned char *) pBuffer, lSize);

	for (ucOPRIter = 1; ucOPRIter < 8; ucOPRIter++)
	{
		SHIFTL((unsigned char *) pBuffer, lSize);

		asResults[iCountResults].iOperation = OPR_SHIFT;
		asResults[iCountResults].iKey = ucOPRIter;
		StringsAnalysis(pBuffer, lSize, &asResults[iCountResults++], iMinimumStringLength, (unsigned char) iTerminator);
		if (iFlagSave && asResults[iCountResults - 1].iCountStrings > 0)
			SaveFile(pcArgFile, OPRString(OPR_SHIFT), ucOPRIter, pBuffer, lSize);
	}
	for (ucOPRIter = 1; ucOPRIter < 8; ucOPRIter++)
		SHIFTR((unsigned char *) pBuffer, lSize);

	if (iFlagMaxSort)
		qsort(asResults, iCountResults, sizeof(struct RESULT), CompareStringMaxLengths);
	else
		qsort(asResults, iCountResults, sizeof(struct RESULT), CompareStringCounts);

	puts(iFlagCSV ? "Opr,Key,Count,Avg,Max" : "Opr   Key  Count Avg   Max");
	for (iIter = 0; iIter < iCountResults; iIter++)
	{
		if (asResults[iIter].iCountStrings != 0)
		{
			printf(iFlagCSV ? "%s,0x%02x,%d,%.1f,%d" : "%5s 0x%02x %5d %5.1f %5d", OPRString(asResults[iIter].iOperation), asResults[iIter].iKey, asResults[iIter].iCountStrings, asResults[iIter].iCountCharacters * 1.0 / asResults[iIter].iCountStrings, asResults[iIter].iMaxStringLength);
			if (iFlagDump)
				switch (asResults[iIter].iOperation)
				{
					case OPR_XOR:
						XOR((unsigned char *) pBuffer, lSize, asResults[iIter].iKey);
						printf(iFlagCSV ? ",\"%.*s\"\n" : " %.*s\n", asResults[iIter].iMaxStringLength, (char *)pBuffer + asResults[iIter].lIndexMaxString);
						XOR((unsigned char *) pBuffer, lSize, asResults[iIter].iKey);
						break;

					case OPR_ROL:
						for (ucOPRIter = 0; ucOPRIter < asResults[iIter].iKey; ucOPRIter++)
							ROL((unsigned char *) pBuffer, lSize);
						printf(iFlagCSV ? ",\"%.*s\"\n" : " %.*s\n", asResults[iIter].iMaxStringLength, (char *)pBuffer + asResults[iIter].lIndexMaxString);
						for (ucOPRIter = 0; ucOPRIter < (8 - asResults[iIter].iKey); ucOPRIter++)
							ROL((unsigned char *) pBuffer, lSize);
						break;

					case OPR_SHIFT:
						for (ucOPRIter = 0; ucOPRIter < asResults[iIter].iKey; ucOPRIter++)
							SHIFTL((unsigned char *) pBuffer, lSize);
						printf(iFlagCSV ? ",\"%.*s\"\n" : " %.*s\n", asResults[iIter].iMaxStringLength, (char *)pBuffer + asResults[iIter].lIndexMaxString);
						for (ucOPRIter = 0; ucOPRIter < asResults[iIter].iKey; ucOPRIter++)
							SHIFTR((unsigned char *) pBuffer, lSize);
						break;
				}
			else
				puts("");
		}
	}
}

void DoStringsPrint(void *pBuffer, long lSize, int iOperation, int iKey, int iMinimumStringLength, int iTerminator, const char *pcArgFile, int iFlagSave)
{
	int iIter = iKey;

	switch (iOperation)
	{
		case OPR_XOR:
			XOR((unsigned char *) pBuffer, lSize, iKey);
			break;

		case OPR_ROL:
			while (iIter--)
				ROL((unsigned char *) pBuffer, lSize);
			break;

		case OPR_SHIFT:
			while (iIter--)
				SHIFTL((unsigned char *) pBuffer, lSize);
			break;
	}
	StringsPrint(pBuffer, lSize, iMinimumStringLength, (unsigned char) iTerminator);
	if (iFlagSave)
		SaveFile(pcArgFile, OPRString(iOperation), iKey, pBuffer, lSize);
}

int CheckArgumentsAndOptions(const char *pcArgFile, int iFlagDump, int iFlagMaxSort, int iTerminator, int iFlagCSV, int iOperation, int iKey)
{
	if (strlen(pcArgFile) >= MAXPATH - 1)
	{
		fprintf(stderr, "Error: filename is too long\n");
		return -1;
	}
	if (iTerminator < 0x00 || iTerminator > 0xFF)
	{
		fprintf(stderr, "Error: terminator out of range\n");
		return -1;
	}
	if (iOperation == -1 && iKey != -1 || iOperation != -1 && iKey == -1)
	{
		fprintf(stderr, "Error: flags -o and -k must be used together\n");
		return -1;
	}
	if (iOperation != -1 && iKey != -1 && (iFlagDump || iFlagMaxSort || iFlagCSV))
	{
		fprintf(stderr, "Error: flags -o and -k can't be used with flags -d, -m or -c\n");
		return -1;
	}

	return 0;
}

int StoreFileContentInMemory(const char *pcArgFile, void **ppBuffer, long *plSize)
{
	FILE *fIn;
	struct stat statFile;

	if (stat(pcArgFile, &statFile) != 0)
	{
		fprintf(stderr, "error opening file %s\n", pcArgFile);
		return -1;
	}

	if ((*ppBuffer = malloc(statFile.st_size)) == NULL)
	{
		fprintf (stderr, "file %s is too large %ld\n", pcArgFile, (unsigned long)statFile.st_size);
		return -1;
	}

	if ((fIn = fopen(pcArgFile, "rb")) == NULL)
	{
		fprintf(stderr, "error opening file %s\n", pcArgFile);
		free (*ppBuffer);
		return -1;
	}

	if (fread(*ppBuffer, statFile.st_size, 1, fIn) != 1)
	{
		fprintf(stderr, "error reading file %s\n", pcArgFile);
		fclose (fIn);
		free (*ppBuffer);
		return -1;
	}

	fclose (fIn);

	*plSize = statFile.st_size;

	return 0;
}

main(int argc, char **argv)
{
	void *pBuffer;
	char *pcArgFile;
	int iFlagSave;
	int iFlagUnicode;
	int iFlagDump;
	int iFlagMaxSort;
	int iMinimumStringLength;
	int iTerminator;
	int iFlagCSV;
	int iOperation;
	int iKey;
	long lSize;

	if (ParseArgs(argc, argv, &pcArgFile, &iFlagSave, &iFlagUnicode, &iFlagDump, &iFlagMaxSort, &iMinimumStringLength, &iTerminator, &iFlagCSV, &iOperation, &iKey))
	{
		fprintf(stderr, "Usage: XORStrings [options] file\n"
										"XORStrings V0.0.1, look for XOR, ROL or SHIFT encoded strings in a file\n"
										"Use -s to save the XOR, ROL or SHIFT encoded file\n"
//										"Use -u to search for Unicode strings (limited support)\n"
										"Use -d to dump the longest string\n"
										"Use -m sort by maximum string length\n"
										"Use -l to set the minimum string length (default 5)\n"
										"Use -t to set the string terminator character, accepts integer or hex number (default 0)\n"
										"Use -c to output CSV\n"
										"Use -o to select the operation (XOR, ROL or SHIFT) to perform (to be used together with -k)\n"
										"Use -k to select the key for the operation to perform (to be used together with -o)\n"
										"Source code put in the public domain by Didier Stevens, no Copyright\n"
										"Use at your own risk\n"
										"https://DidierStevens.com\n");
		return -1;
	}

	// Set defaults
	if (iMinimumStringLength == -1)
		iMinimumStringLength = 5;
	if (iTerminator == -1)
		iTerminator = 0;

	if (-1 == CheckArgumentsAndOptions(pcArgFile, iFlagDump, iFlagMaxSort, iTerminator, iFlagCSV, iOperation, iKey))
		return -1;

	if (-1 == StoreFileContentInMemory(pcArgFile, &pBuffer, &lSize))
		return -1;

	if (iOperation == -1 && iKey == -1)
		DoStringsAnalysis(pBuffer, lSize, pcArgFile, iFlagSave, iFlagDump, iFlagMaxSort, iMinimumStringLength, iTerminator, iFlagCSV);
	else
		DoStringsPrint(pBuffer, lSize, iOperation, iKey, iMinimumStringLength, iTerminator, pcArgFile, iFlagSave);

	free(pBuffer);

	return 0;
}
