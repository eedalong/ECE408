

#ifndef __WB_ARG_H__
#define __WB_ARG_H__


struct st_wbArg_t {
    int inputCount;
    char ** inputFiles;
    char * outputFile;
    char * expectedOutput;
    char * type;
};

#define wbArg_getInputCount(wa)					((wa).inputCount)
#define wbArg_getInputFiles(wa)					((wa).inputFiles)
#define wbArg_getInputFile(wa, ii)				(wbArg_getInputFiles(wa)[ii])
#define wbArg_getOutputFile(wa)					((wa).outputFile)
#define wbArg_getExpectedOutputFile(wa)			((wa).expectedOutput)
#define wbArg_getType(wa)						((wa).type)

#define wbArg_setInputCount(wa, val)			(wbArg_getInputCount(wa) = val)
#define wbArg_setInputFiles(wa, val)			(wbArg_getInputFiles(wa) = val)
#define wbArg_setInputFile(wa, ii, val)			(wbArg_getInputFile(wa, ii) = val)
#define wbArg_setOutputFile(wa, val)			(wbArg_getOutputFile(wa) = val)
#define wbArg_setExpectedOutputFile(wa, val)	(wbArg_getExpectedOutputFile(wa) = val)
#define wbArg_setType(wa, val)					(wbArg_getType(wa) = val)


wbArg_t wbArg_new(void);
void wbArg_delete(wbArg_t arg);
wbArg_t wbArg_read(int argc, char ** argv);


#endif /* __WB_ARG_H__ */


