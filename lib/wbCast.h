

#ifndef __WB_CAST_H__
#define __WB_CAST_H__

template <typename X, typename Y>
static inline void wbCast(X & x, const Y & y, size_t len) {
    size_t ii;

    for (ii = 0; ii < len; ii++) {
        x[ii] = (X) y[ii];
    }

    return ;
}

template <typename X, typename Y>
static inline X * wbCast(const Y & y, size_t len) {
    size_t ii;
    X * x = wbNewArray(X, len);

    for (ii = 0; ii < len; ii++) {
        x[ii] = (X) y[ii];
    }

    return x;
}

#endif /* __WB_CAST_H__ */

