

#ifndef __WB_TIMER_H__
#define __WB_TIMER_H__


#include    <time.h>
#include    <stdint.h>
#include    <sys/types.h>

#ifdef __APPLE__
#include    <mach/mach_time.h>
#endif /* __APPLE__ */


#ifdef _WIN32
extern uint64_t _hrtime_frequency;
#endif /* _WIN32 */

extern wbTimer_t _timer;

typedef enum en_wbTimerKind_t {
    wbTimerKind_Generic,
    wbTimerKind_IO,
    wbTimerKind_GPU,
    wbTimerKind_Copy,
    wbTimerKind_Driver,
    wbTimerKind_CopyAsync,
    wbTimerKind_Compute,
    wbTimerKind_CPUGPUOverlap,
} wbTimerKind_t;

struct st_wbTimerNode_t {
    int id;
    int level;
    wbBool stoppedQ;
    wbTimerKind_t kind;
    uint64_t startTime;
    uint64_t endTime;
    uint64_t elapsedTime;
    int startLine;
    int endLine;
    const char * startFunction;
    const char * endFunction;
    const char * startFile;
    const char * endFile;
    wbTimerNode_t next;
    wbTimerNode_t prev;
    wbTimerNode_t parent;
    char * msg;
};

struct st_wbTimer_t {
    size_t length;
    wbTimerNode_t head;
    wbTimerNode_t tail;
    uint64_t startTime;
    uint64_t endTime;
    uint64_t elapsedTime;
};

#define wbTimerNode_getId(node)                 ((node)->id)
#define wbTimerNode_getLevel(node)              ((node)->level)
#define wbTimerNode_getStoppedQ(node)           ((node)->stoppedQ)
#define wbTimerNode_getKind(node)               ((node)->kind)
#define wbTimerNode_getStartTime(node)          ((node)->startTime)
#define wbTimerNode_getEndTime(node)            ((node)->endTime)
#define wbTimerNode_getElapsedTime(node)        ((node)->elapsedTime)
#define wbTimerNode_getStartLine(node)          ((node)->startLine)
#define wbTimerNode_getEndLine(node)            ((node)->endLine)
#define wbTimerNode_getStartFunction(node)      ((node)->startFunction)
#define wbTimerNode_getEndFunction(node)        ((node)->endFunction)
#define wbTimerNode_getStartFile(node)          ((node)->startFile)
#define wbTimerNode_getEndFile(node)            ((node)->endFile)
#define wbTimerNode_getNext(node)               ((node)->next)
#define wbTimerNode_getPrevious(node)           ((node)->prev)
#define wbTimerNode_getParent(node)             ((node)->parent)
#define wbTimerNode_getMessage(node)            ((node)->msg)

#define wbTimerNode_setId(node, val)            (wbTimerNode_getId(node) = val)
#define wbTimerNode_setLevel(node, val)         (wbTimerNode_getLevel(node) = val)
#define wbTimerNode_setStoppedQ(node, val)      (wbTimerNode_getStoppedQ(node) = val)
#define wbTimerNode_setKind(node, val)          (wbTimerNode_getKind(node) = val)
#define wbTimerNode_setStartTime(node, val)     (wbTimerNode_getStartTime(node) = val)
#define wbTimerNode_setEndTime(node, val)       (wbTimerNode_getEndTime(node) = val)
#define wbTimerNode_setElapsedTime(node, val)   (wbTimerNode_getElapsedTime(node) = val)
#define wbTimerNode_setStartLine(node, val)     (wbTimerNode_getStartLine(node) = val)
#define wbTimerNode_setEndLine(node, val)       (wbTimerNode_getEndLine(node) = val)
#define wbTimerNode_setStartFunction(node, val) (wbTimerNode_getStartFunction(node) = val)
#define wbTimerNode_setEndFunction(node, val)   (wbTimerNode_getEndFunction(node) = val)
#define wbTimerNode_setStartFile(node, val)     (wbTimerNode_getStartFile(node) = val)
#define wbTimerNode_setEndFile(node, val)       (wbTimerNode_getEndFile(node) = val)
#define wbTimerNode_setNext(node, val)          (wbTimerNode_getNext(node) = val)
#define wbTimerNode_setPrevious(node, val)      (wbTimerNode_getPrevious(node) = val)
#define wbTimerNode_setParent(node, val)        (wbTimerNode_getParent(node) = val)
#define wbTimerNode_setMessage(node, val)       (wbTimerNode_getMessage(node) = val)

#define wbTimerNode_stoppedQ(node)              (wbTimerNode_getStoppedQ(node) == wbTrue)
#define wbTimerNode_hasNext(node)               (wbTimerNode_getNext(node) != NULL)
#define wbTimerNode_hasPrevious(node)           (wbTimerNode_getPrevious(node) != NULL)
#define wbTimerNode_hasParent(node)             (wbTimerNode_getParent(node) != NULL)

uint64_t _hrtime(void);

wbTimer_t wbTimer_new(void);
void wbTimer_delete(wbTimer_t timer);

string wbTimer_toJSON(wbTimer_t timer);
string wbTimer_toJSON();

string wbTimer_toXML(wbTimer_t timer);
string wbTimer_toXML();

wbTimerNode_t wbTimer_start(wbTimerKind_t kind, const char * file, const char * fun, int line);
wbTimerNode_t wbTimer_start(wbTimerKind_t kind, string msg, const char * file, const char * fun, int line);
void wbTimer_stop(wbTimerKind_t kind, string msg, const char * file, const char * fun, int line);
void wbTimer_stop(wbTimerKind_t kind, const char * file, const char * fun, int line);

#define wbTime_start(kind, ...)				wbTimer_start(wbTimerKind_##kind, wbString(__VA_ARGS__), wbFile, wbFunction, wbLine)
#define wbTime_stop(kind, ...)              wbTimer_stop(wbTimerKind_##kind, wbString(__VA_ARGS__), wbFile, wbFunction, wbLine)


#endif /* __WB_TIMER_H__ */

