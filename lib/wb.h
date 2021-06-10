

#ifndef __WB_H__
#define __WB_H__

/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/

#ifdef _MSC_VER
#define __func__					__FUNCTION__
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS		1
#endif /* _CRT_SECURE_NO_WARNINGS */
#define _CRT_SECURE_NO_DEPRECATE	1
#define _CRT_NONSTDC_NO_DEPRECATE	1

// there are problems with MSVC not using lazy load. These
// mostly occur because code for non-lazy loads doesn't
// deal well with lines that don't have a newline, and many of the
// supplied datasets done.
#define LAZY_FILE_LOAD

#include    <windows.h>
#include    <direct.h>
#include    <io.h>

#endif /* _MSC_VER */

#include    <stdio.h>
#include    <stdlib.h>

#define wbStmt(stmt)        stmt

/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/

#define wbLine                      __LINE__
#define wbFile                      __FILE__
#define wbFunction                  __func__

#define wbExit()					wbAssert(0); exit(1)

#ifdef WB_USE_COURSERA
#define wbLogger_printOnExit		1
#else /* WB_USE_COURSERA */
#define wbLogger_printOnLog			1
#endif /* WB_USE_COURSERA */


/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/

extern char * solutionJSON;

/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/

#include    <wbTypes.h>
#include    <wbAssert.h>
#include    <wbMalloc.h>
#include    <wbString.h>
#include    <wbTimer.h>
#include    <wbLogger.h>
#include    <wbComparator.h>
#include    <wbFile.h>
#include    <wbImport.h>
#include    <wbExport.h>
#include    <wbCast.h>
#include    <wbImage.h>
#include    <wbArg.h>
#include    <wbSolution.h>
#include    <wbExit.h>
#include    <wbInit.h>
#include    <wbCUDA.h>

/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/
/***********************************************************/


#endif /* __WB_H__ */

