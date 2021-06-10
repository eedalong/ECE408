

#ifndef __WB_ASSERT_H__
#define __WB_ASSERT_H__

#include	<assert.h>

#ifdef WB_DEBUG
#define wbAssert(cond)              assert(cond)
#define wbAssertMessage(msg, cond)  do {                        \
                                        if (!(cond)) {          \
                                            wbPrint(msg);       \
                                            wbAssert(cond);     \
                                        }                       \
                                    } while (0)
#else /* WB_DEBUG */
#define wbAssert(...)
#define wbAssertMessage(...)
#endif /* WB_DEBUG */

#endif /* __WB_ASSERT_H__ */


