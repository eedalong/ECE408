
#ifndef __WB_LOGGER_H__
#define __WB_LOGGER_H__

typedef enum en_wbLogLevel_t {
    wbLogLevel_unknown = -1,
    wbLogLevel_OFF = 0,
    wbLogLevel_FATAL,
    wbLogLevel_ERROR,
    wbLogLevel_WARN,
    wbLogLevel_INFO,
    wbLogLevel_DEBUG,
    wbLogLevel_TRACE
} wbLogLevel_t;

struct st_wbLogEntry_t {
    int line;
    char * msg;
    uint64_t time;
    const char * fun;
    const char * file;
    wbLogLevel_t level;
    wbLogEntry_t next;
};

struct st_wbLogger_t {
    int length;
    wbLogEntry_t head;
    wbLogLevel_t level;
};


#define wbLogEntry_getMessage(elem)         ((elem)->msg)
#define wbLogEntry_getTime(elem)            ((elem)->time)
#define wbLogEntry_getLevel(elem)           ((elem)->level)
#define wbLogEntry_getNext(elem)            ((elem)->next)
#define wbLogEntry_getLine(elem)            ((elem)->line)
#define wbLogEntry_getFunction(elem)        ((elem)->fun)
#define wbLogEntry_getFile(elem)            ((elem)->file)

#define wbLogEntry_setMessage(elem, val)    (wbLogEntry_getMessage(elem) = val)
#define wbLogEntry_setTime(elem, val)       (wbLogEntry_getTime(elem) = val)
#define wbLogEntry_setLevel(elem, val)      (wbLogEntry_getLevel(elem) = val)
#define wbLogEntry_setNext(elem, val)       (wbLogEntry_getNext(elem) = val)
#define wbLogEntry_setLine(elem, val)       (wbLogEntry_getLine(elem) = val)
#define wbLogEntry_setFunction(elem, val)   (wbLogEntry_getFunction(elem) = val)
#define wbLogEntry_setFile(elem, val)       (wbLogEntry_getFile(elem) = val)

#define wbLogger_getLength(log)             ((log)->length)
#define wbLogger_getHead(log)               ((log)->head)
#define wbLogger_getLevel(log)              ((log)->level)

#define wbLogger_setLength(log, val)        (wbLogger_getLength(log) = val)
#define wbLogger_setHead(log, val)          (wbLogger_getHead(log) = val)

#define wbLogger_incrementLength(log)       (wbLogger_getLength(log)++)
#define wbLogger_decrementLength(log)       (wbLogger_getLength(log)--)

#define wbLog(level, ...)                   wbLogger_append(wbLogLevel_##level, wbString(__VA_ARGS__), wbFile, wbFunction, wbLine)


extern wbLogger_t _logger;

wbLogger_t wbLogger_new();

void wbLogger_clear(wbLogger_t logger);

void wbLogger_delete(wbLogger_t logger);

void wbLogger_append(wbLogLevel_t level, string msg, const char * file, const char * fun, int line);

string wbLogger_toXML(wbLogger_t logger);
string wbLogger_toXML();

string wbLogger_toJSON(wbLogger_t logger);
string wbLogger_toJSON();

#endif /* __WB_LOGGER_H__ */


