

#ifndef __WB_STRING_H__
#define __WB_STRING_H__


#include	<wb.h>
#include	<vector>
#include	<string>
#include	<ostream>
#include	<iostream>
#include	<sstream>
#include	<cstring>

using namespace std;

template <typename T> static inline string wbString(const T & x);

static inline void wbString_replace(string & value, string const & search, string const & replace) {
    for (string::size_type next = value.find(search); next != string::npos; next = value.find(search, next)) {
        value.replace(next, search.length(), replace);
        next += replace.length();
    }
}

static inline string wbString_quote(string str) {
    string s = str;
    wbString_replace(s, "\\", "\\\\");
    s = "\"" + s + "\"";
    return s;
}


static inline char * wbString_duplicate(const char * str) {
    if (str == NULL) {
        return NULL;
    } else {
        char * newstr;
        size_t len = strlen(str);
        newstr = wbNewArray(char, len+1);
        memcpy(newstr, str, len * sizeof(char));
        newstr[len] = '\0';
        return newstr;
    }
}

static inline char * wbString_duplicate(std::string str) {
    return wbString_duplicate(str.c_str());
}

static inline string wbString(void) {
    string s = "";
    return s;
}

template <typename T>
static inline string wbString(const T & x) {
    stringstream ss;
    ss << x;
    return ss.str();
}

template <>
inline string wbString(const bool & x) {
    return x ? "True" : "False";
}

template <>
inline string wbString(const vector<string> & x) {
    stringstream ss;
    ss << "{";
    for (size_t ii = 0; ii < x.size(); ii++) {
        ss << wbString_quote(x[ii]);
        if (ii != x.size() - 1) {
            ss << ", ";
        }
    }
    ss << "}";

    return ss.str();
}

template <>
inline string wbString(const vector<int> & x) {
    stringstream ss;
    ss << "{";
    for (size_t ii = 0; ii < x.size(); ii++) {
        ss << x[ii];
        if (ii != x.size() - 1) {
            ss << ", ";
        }
    }
    ss << "}";

    return ss.str();
}

template <>
inline string wbString(const vector<double> & x) {
    stringstream ss;
    ss << "{";
    for (size_t ii = 0; ii < x.size(); ii++) {
        ss << x[ii];
        if (ii != x.size() - 1) {
            ss << ", ";
        }
    }
    ss << "}";

    return ss.str();
}


template <typename T0, typename T1>
static inline string wbString(const T0 & x0, const T1 & x1) {
    stringstream ss;
    ss << wbString(x0) << wbString(x1);

    return ss.str();
}

template <typename T0, typename T1, typename T2>
static inline string wbString(const T0 & x0, const T1 & x1, const T2 & x2) {
    stringstream ss;
    ss << wbString(x0) << wbString(x1) << wbString(x2);
    return ss.str();
}


template <typename T0, typename T1, typename T2, typename T3>
static inline string wbString(const T0 & x0, const T1 & x1, const T2 & x2, const T3 & x3) {
    stringstream ss;
    ss << wbString(x0) << wbString(x1) << wbString(x2) << wbString(x3);

    return ss.str();
}

template <typename T0, typename T1, typename T2, typename T3, typename T4>
static inline string wbString(const T0 & x0, const T1 & x1, const T2 & x2, const T3 & x3, const T4 & x4) {
    stringstream ss;
    ss << wbString(x0) << wbString(x1) << wbString(x2) << wbString(x3) << wbString(x4);

    return ss.str();
}

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
static inline string wbString(const T0 & x0, const T1 & x1, const T2 & x2, const T3 & x3, const T4 & x4,
                              const T5 & x5) {
    stringstream ss;
    ss << wbString(x0) << wbString(x1) << wbString(x2) << wbString(x3) << wbString(x4)
       << wbString(x5);

    return ss.str();
}

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
         typename T6>
static inline string wbString(const T0 & x0, const T1 & x1, const T2 & x2, const T3 & x3, const T4 & x4,
                              const T5 & x5, const T6 & x6) {
    stringstream ss;
    ss << wbString(x0) << wbString(x1) << wbString(x2) << wbString(x3) << wbString(x4)
       << wbString(x5) << wbString(x6);

    return ss.str();
}

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
         typename T6, typename T7>
static inline string wbString(const T0 & x0, const T1 & x1, const T2 & x2, const T3 & x3, const T4 & x4,
                              const T5 & x5, const T6 & x6, const T7 & x7) {
    stringstream ss;
    ss << wbString(x0) << wbString(x1) << wbString(x2) << wbString(x3) << wbString(x4)
       << wbString(x5) << wbString(x6) << wbString(x7);

    return ss.str();
}

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
         typename T6, typename T7, typename T8>
static inline string wbString(const T0 & x0, const T1 & x1, const T2 & x2, const T3 & x3, const T4 & x4,
                              const T5 & x5, const T6 & x6, const T7 & x7, const T8 & x8) {
    stringstream ss;
    ss << wbString(x0) << wbString(x1) << wbString(x2) << wbString(x3) << wbString(x4)
       << wbString(x5) << wbString(x6) << wbString(x7) << wbString(x8);

    return ss.str();
}

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
         typename T6, typename T7, typename T8, typename T9>
static inline string wbString(const T0 & x0, const T1 & x1, const T2 & x2, const T3 & x3, const T4 & x4,
                              const T5 & x5, const T6 & x6, const T7 & x7, const T8 & x8, const T9 & x9) {
    stringstream ss;
    ss << wbString(x0) << wbString(x1) << wbString(x2) << wbString(x3) << wbString(x4)
       << wbString(x5) << wbString(x6) << wbString(x7) << wbString(x8) << wbString(x9);

    return ss.str();
}

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
         typename T6, typename T7, typename T8, typename T9, typename T10>
static inline string wbString(const T0 & x0, const T1 & x1, const T2 & x2, const T3 & x3, const T4 & x4,
                              const T5 & x5, const T6 & x6, const T7 & x7, const T8 & x8, const T9 & x9,
                              const T10 & x10) {
    stringstream ss;
    ss << wbString(x0) << wbString(x1) << wbString(x2) << wbString(x3) << wbString(x4)
       << wbString(x5) << wbString(x6) << wbString(x7) << wbString(x8) << wbString(x9)
       << wbString(x1);

    return ss.str();
}

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
         typename T6, typename T7, typename T8, typename T9, typename T10, typename T11>
static inline string wbString(const T0 & x0, const T1 & x1, const T2 & x2, const T3 & x3, const T4 & x4,
                              const T5 & x5, const T6 & x6, const T7 & x7, const T8 & x8, const T9 & x9,
                              const T10 & x10, const T11 & x11) {
    stringstream ss;
    ss << wbString(x0) << wbString(x1) << wbString(x2) << wbString(x3) << wbString(x4)
       << wbString(x5) << wbString(x6) << wbString(x7) << wbString(x8) << wbString(x9)
       << wbString(x10) << wbString(x11);

    return ss.str();
}

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
         typename T6, typename T7, typename T8, typename T9, typename T10, typename T11,
         typename T12>
static inline string wbString(const T0 & x0, const T1 & x1, const T2 & x2, const T3 & x3, const T4 & x4,
                              const T5 & x5, const T6 & x6, const T7 & x7, const T8 & x8, const T9 & x9,
                              const T10 & x10, const T11 & x11, const T12 & x12) {
    stringstream ss;
    ss << wbString(x0) << wbString(x1) << wbString(x2) << wbString(x3) << wbString(x4)
       << wbString(x5) << wbString(x6) << wbString(x7) << wbString(x8) << wbString(x9)
       << wbString(x10) << wbString(x11) << wbString(x12);
    return ss.str();
}

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
         typename T6, typename T7, typename T8, typename T9, typename T10, typename T11,
         typename T12, typename T13>
static inline string wbString(const T0 & x0, const T1 & x1, const T2 & x2, const T3 & x3, const T4 & x4,
                              const T5 & x5, const T6 & x6, const T7 & x7, const T8 & x8, const T9 & x9,
                              const T10 & x10, const T11 & x11, const T12 & x12, const T13 & x13) {
    stringstream ss;
    ss << wbString(x0) << wbString(x1) << wbString(x2) << wbString(x3) << wbString(x4)
       << wbString(x5) << wbString(x6) << wbString(x7) << wbString(x8) << wbString(x9)
       << wbString(x10) << wbString(x11) << wbString(x12) << wbString(x13);
    return ss.str();
}


template <typename X, typename Y>
static inline wbBool wbString_sameQ(const X & x, const Y & y) {
    string xs = wbString(x);
    string ys = wbString(y);
    return strcmp(xs.c_str(), ys.c_str()) == 0;
}

static inline wbBool wbString_sameQ(const string & x, const string & y) {
    return x.compare(y) == 0;
}


static inline char * wbString_toLower(const char * str) {
    if (str == NULL) {
        return NULL;
    } else {
        char * res, * iter;

        res = iter = wbString_duplicate(str);
        while (*iter != '\0') {
            *iter++ = tolower(*str++);
        }
        return res;
    }
}

static inline wbBool wbString_startsWith(const char * str, const char * prefix) {
    while (*prefix != '\0') {
        if (*str == '\0' || *str != *prefix) {
            return wbFalse;
        }
        str++;
        prefix++;
    }
    return wbTrue;
}

#endif /* __WB_STRING_H__ */
