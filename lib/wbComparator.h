

#ifndef __WB_COMPARATOR_H__
#define __WB_COMPARATOR_H__

#include    <wb.h>

template <typename T>
static inline T _abs(const T & a) {
    return a < 0 ? -a : a;
}

static inline wbBool _almostEqual(double A, double B, double eps) {
    if (A == 0) {
        return _abs(B) < eps;
    } else if (B == 0) {
        return _abs(A) < eps;
    } else {
#if 0
        double d = max(_abs(A), _abs(B));
        double g = (_abs(A - B)/d);
#else
        double g = _abs(A - B);
#endif
        if (g <= eps) {
            return wbTrue;
        } else {
            return wbFalse;
        }
    }
}

static inline wbBool _almostEqual(float A, float B, float eps) {
    if (A == 0) {
        return _abs(B) < eps;
    } else if (B == 0) {
        return _abs(A) < eps;
    } else {
#if 0
        float d = max(_abs(A), _abs(B));
        float g = (_abs(A - B)/d);
#else
        float g = _abs(A - B);
#endif
        if (g <= eps) {
            return wbTrue;
        } else {
            return wbFalse;
        }
    }
}

static inline wbBool _almostEqual(double A, double B) {
    return _almostEqual(A, B, 0.1);
}

static inline wbBool _almostEqual(float A, float B) {
    return _almostEqual(A, B, 0.1f);
}

static inline wbBool _almostEqual2sComplement(float A, float B, int maxUlps) {
    // Make sure maxUlps is non-negative and small enough that the
    // default NAN won't compare as equal to anything.

    int aInt, bInt, intDiff;

    wbAssert(maxUlps > 0 && maxUlps < 4 * 1024 * 1024);

    int * tmp = reinterpret_cast<int *>(&A);
    aInt = *tmp;

    // Make aInt lexicographically ordered as a twos-complement int
    if (aInt < 0) {
        aInt = 0x80000000 - aInt;
    }
    // Make bInt lexicographically ordered as a twos-complement int
    tmp = reinterpret_cast<int *>(&B);
    bInt = *tmp;
    if (bInt < 0) {
        bInt = 0x80000000 - bInt;
    }
    intDiff = _abs(aInt - bInt);
    if (intDiff <= maxUlps) {
        return wbTrue;
    }
    return wbFalse;
}

static inline wbBool _almostEqual2sComplement(float A, float B) {
    return _almostEqual2sComplement(A, B, 4);
}

static inline wbBool _almostEqual2sComplement(double A, double B, int maxUlps) {
    // Make sure maxUlps is non-negative and small enough that the
    // default NAN won't compare as equal to anything.

    int64_t aInt, bInt, intDiff;

    wbAssert(maxUlps > 0 && maxUlps < 4 * 1024 * 1024);

    int64_t * tmp = reinterpret_cast<int64_t *>(&A);
    aInt = *tmp;

    // Make aInt lexicographically ordered as a twos-complement int
    if (aInt < 0) {
        aInt = 0x80000000 - aInt;
    }
    // Make bInt lexicographically ordered as a twos-complement int
    tmp = reinterpret_cast<int64_t *>(&B);
    bInt = *tmp;
    if (bInt < 0) {
        bInt = 0x80000000 - bInt;
    }
    intDiff = _abs(aInt - bInt);
    if (intDiff <= maxUlps) {
        return wbTrue;
    }
    return wbFalse;
}

static inline wbBool _almostEqual2sComplement(double A, double B) {
    return _almostEqual2sComplement(A, B, 4);
}

template <typename T>
static inline int wbCompare(const T & a, const T & b) {
    if (a == b) {
        return 0;
    } else if (a < b) {
        return -1;
    } else {
        return 1;
    }
}

template <>
inline int wbCompare(const double & a, const double & b) {
    if (_almostEqual(a, b)) {
        return 0;
    } else if (a < b) {
        return -1;
    } else {
        return 1;
    }
}

template <>
inline int wbCompare(const float & a, const float & b) {
    if (_almostEqual(a, b)) {
        return 0;
    } else if (a < b) {
        return -1;
    } else {
        return 1;
    }
}

template <typename T>
static inline wbBool wbEqualQ(const T & a, const T & b) {
    return wbCompare(a, b) == 0;
}

template <typename T>
static inline wbBool wbUnequalQ(const T & a, const T & b) {
    return wbCompare(a, b) != 0;
}

template <typename T>
static inline wbBool wbEqualQ(const T * a, const T * b, size_t n) {
    size_t ii;

    for (ii = 0; ii < n; ii++) {
        if (wbUnequalQ(a[ii], b[ii])) {
            return wbFalse;
        }
    }
    return wbTrue;
}

template <typename T>
static inline wbBool wbUnequalQ(const T * a, const T * b, size_t n) {
    return !wbEqualQ(a, b, n);
}


#endif /* __WB_COMPARATOR_H__ */

