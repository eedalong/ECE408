
#include	<wb.h>


wbLogger_t _logger = NULL;

static inline wbBool wbLogEntry_hasNext(wbLogEntry_t elem) {
    return wbLogEntry_getNext(elem) != NULL;
}

static inline wbLogEntry_t wbLogEntry_new() {
    wbLogEntry_t elem;

    elem = wbNew(struct st_wbLogEntry_t);

    wbLogEntry_setMessage(elem, NULL);
    wbLogEntry_setTime(elem, _hrtime());
#ifndef NDEBUG
    wbLogEntry_setLevel(elem, wbLogLevel_TRACE);
#else
    wbLogEntry_setLevel(elem, wbLogLevel_OFF);
#endif
    wbLogEntry_setNext(elem, NULL);

    wbLogEntry_setLine(elem, -1);
    wbLogEntry_setFile(elem, NULL);
    wbLogEntry_setFunction(elem, NULL);

    return elem;
}

static inline wbLogEntry_t wbLogEntry_initialize(wbLogLevel_t level, string msg, const char * file,
        const char * fun, int line) {
    wbLogEntry_t elem;

    elem = wbLogEntry_new();

    wbLogEntry_setLevel(elem, level);

    wbLogEntry_setMessage(elem, wbString_duplicate(msg));

    wbLogEntry_setLine(elem, line);
    wbLogEntry_setFile(elem, file);
    wbLogEntry_setFunction(elem, fun);

    return elem;
}

static inline void wbLogEntry_delete(wbLogEntry_t elem) {
    if (elem != NULL) {
        if (wbLogEntry_getMessage(elem) != NULL) {
            wbFree(wbLogEntry_getMessage(elem));
        }
        wbDelete(elem);
    }
    return ;
}

static inline const char * getLevelName(wbLogLevel_t level) {
    switch (level) {
    case wbLogLevel_unknown:
        return "Unknown";
    case wbLogLevel_OFF:
        return "Off";
    case wbLogLevel_FATAL:
        return "Fatal";
    case wbLogLevel_ERROR:
        return "Error";
    case wbLogLevel_WARN:
        return "Warn";
    case wbLogLevel_INFO:
        return "Info";
    case wbLogLevel_DEBUG:
        return "Debug";
    case wbLogLevel_TRACE:
        return "Trace";
    }
    return NULL;
}

static inline string wbLogEntry_toJSON(wbLogEntry_t elem) {
    if (elem != NULL) {
        stringstream ss;

        ss << "{\n";
        ss << wbString_quote("level") << ":" << wbString_quote(getLevelName(wbLogEntry_getLevel(elem))) << ",\n";
        ss << wbString_quote("message") << ":" << wbString_quote(wbLogEntry_getMessage(elem)) << ",\n";
        ss << wbString_quote("file") << ":" << wbString_quote(wbLogEntry_getFile(elem)) << ",\n";
        ss << wbString_quote("function") << ":" << wbString_quote(wbLogEntry_getFunction(elem)) << ",\n";
        ss << wbString_quote("line") << ":" << wbLogEntry_getLine(elem) << ",\n";
        ss << wbString_quote("time") << ":" << wbLogEntry_getTime(elem) << "\n";
        ss << "}";

        return ss.str();
    }
    return "";
}

static inline string wbLogEntry_toXML(wbLogEntry_t elem) {
    if (elem != NULL) {
        stringstream ss;

        ss << "<entry>\n";
        ss << "<type>" << "LoggerElement" << "</type>\n";
        ss << "<level>" << wbLogEntry_getLevel(elem) << "</level>\n";
        ss << "<message>" << wbLogEntry_getMessage(elem) << "</message>\n";
        ss << "<file>" << wbLogEntry_getFile(elem) << "</file>\n";
        ss << "<function>" << wbLogEntry_getFunction(elem) << "</function>\n";
        ss << "<line>" << wbLogEntry_getLine(elem) << "</line>\n";
        ss << "<time>" << wbLogEntry_getTime(elem) << "</time>\n";
        ss << "</entry>\n";

        return ss.str();
    }
    return "";
}

wbLogger_t wbLogger_new() {
    wbLogger_t logger;

    logger = wbNew(struct st_wbLogger_t);

    wbLogger_setLength(logger, 0);
    wbLogger_setHead(logger, NULL);
#ifndef NDEBUG
    wbLogger_getLevel(logger) = wbLogLevel_TRACE;
#else
    wbLogger_getLevel(logger) = wbLogLevel_OFF;
#endif

    return logger;
}


static inline void _wbLogger_setLevel(wbLogger_t logger, wbLogLevel_t level) {
    wbLogger_getLevel(logger) = level;
}

static inline void _wbLogger_setLevel(wbLogLevel_t level) {
    wb_init();
    _wbLogger_setLevel(_logger, level);
}

#define wbLogger_setLevel(level)				_wbLogger_setLevel(wbLogLevel_##level)

void wbLogger_clear(wbLogger_t logger) {
    if (logger != NULL) {
        wbLogEntry_t tmp;
        wbLogEntry_t iter;

        iter = wbLogger_getHead(logger);
        while (iter != NULL) {
            tmp = wbLogEntry_getNext(iter);
            wbLogEntry_delete(iter);
            iter = tmp;
        }

        wbLogger_setLength(logger, 0);
        wbLogger_setHead(logger, NULL);
    }
}

void wbLogger_delete(wbLogger_t logger) {
    if (logger != NULL) {
        wbLogger_clear(logger);
        wbDelete(logger);
    }
    return ;
}

void wbLogger_append(wbLogLevel_t level, string msg, const char * file,
                     const char * fun, int line) {
    wbLogEntry_t elem;
    wbLogger_t logger;

    wb_init();

    logger = _logger;

    if (wbLogger_getLevel(logger) < level) {
        return ;
    }

    elem = wbLogEntry_initialize(level, msg, file, fun, line);

    if (wbLogger_getHead(logger) == NULL) {
        wbLogger_setHead(logger, elem);
    } else {
        wbLogEntry_t prev = wbLogger_getHead(logger);

        while (wbLogEntry_hasNext(prev)) {
            prev = wbLogEntry_getNext(prev);
        }
        wbLogEntry_setNext(prev, elem);
    }

#if 0
    if (level <= wbLogger_getLevel(logger) && elem) {
        const char * levelName = getLevelName(level);

        fprintf(stderr,
                "= LOG: %s: %s (In %s:%s on line %d). =\n",
                levelName,
                wbLogEntry_getMessage(elem),
                wbLogEntry_getFile(elem),
                wbLogEntry_getFunction(elem),
                wbLogEntry_getLine(elem));
    }
#endif

    wbLogger_incrementLength(logger);

    return ;
}

string wbLogger_toJSON() {
    return wbLogger_toJSON(_logger);
}

string wbLogger_toJSON(wbLogger_t logger) {
    if (logger != NULL) {
        wbLogEntry_t iter;
        stringstream ss;

        ss << "{\n";
        ss << wbString_quote("elements") << ":[\n";
        for (iter = wbLogger_getHead(logger); iter != NULL; iter = wbLogEntry_getNext(iter)) {
            ss << wbLogEntry_toJSON(iter);
            if (wbLogEntry_getNext(iter) != NULL) {
                ss << ",\n";
            }
        }
        ss << "]\n";
        ss << "}\n";

        return ss.str();
    }
    return "";
}


string wbLogger_toXML() {
    return wbLogger_toXML(_logger);
}

string wbLogger_toXML(wbLogger_t logger) {
    if (logger != NULL) {
        wbLogEntry_t iter;
        stringstream ss;

        ss << "<logger>\n";
        ss << "<type>" << "Logger" << "</type>\n";
        ss << "<elements>\n";
        for (iter = wbLogger_getHead(logger); iter != NULL; iter = wbLogEntry_getNext(iter)) {
            ss << wbLogEntry_toXML(iter);
        }
        ss << "</elements>\n";
        ss << "</logger>\n";

        return ss.str();
    }
    return "";
}


