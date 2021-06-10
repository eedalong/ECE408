

#ifndef __WB_MALLOC_H__
#define __WB_MALLOC_H__

#ifdef WB_USE_CUSTOM_MALLOC
#ifdef __linux__
#define THROW __THROW
#else
#define THROW
#endif

static inline void * _malloc(size_t size) THROW {
    if (size == 0) {
        return NULL;
    } else {
        int err;
        void * res = memmgr_alloc((ulong) size, &err);
        if (err) {
            fprintf(stderr, "<<MEMORY>>:: Memory allocation failed\n");
            exit(1);
        } else {
            size_t ii = 0;
            unsigned char *p = (unsigned char *) res;
            while (ii++ < size) {
                *p++ = 0;
            }
            return res;
        }
    }
}

static inline void _free(void *ptr) THROW {
    if (ptr != NULL) {
        memmgr_free(ptr);
    }
}

static inline void *_calloc(size_t nmemb, size_t size) THROW {
    return _malloc(nmemb * size);
}

static inline void * _realloc(void *ptr, size_t size) THROW {
    if (size == 0) {
        free(ptr);
        return NULL;
    } else if (ptr == NULL) {
        return malloc(size);
    } else {
        void *buf;
        unsigned char *dst;
        unsigned char *src;
        size_t alloc_size, to_copy, i = 0;

        // Allocate new buffer
        buf = malloc(size);

        if (buf != 0) {
            // Find original allocation size
            alloc_size = (size_t) memmgr_get_block_size(ptr);
            to_copy = alloc_size;
            if (to_copy > size) {
                to_copy = size;
            }

            // Copy data to new buffer
            dst = (unsigned char *) buf;
            src = (unsigned char *) ptr;
            while (i++ < to_copy) {
                *dst++ = *src++;
            }

            // Free the old buffer
            free(ptr);
        }

        return buf;
    }
}

#define wbNew(type)                 ((type *) _malloc(sizeof(type)))
#define wbNewArray(type, len)       ((type * ) _malloc((len) * sizeof(type)))
#define wbMalloc(sz)				_malloc(sz)
#define wbDelete(var)               _free(var); var = NULL
#define wbFree(var)                 _free(var); var = NULL
#define wbRealloc(var, newSize)     _realloc(var, newSize)
#define wbReallocArray(t, m, n)     ((t*) _realloc(m, n*sizeof(t)))

#define free _free
#define malloc _malloc
#define calloc _calloc
#define realloc _realloc

#else /* WB_USE_CUSTOM_MALLOC */


static inline void * xMalloc(size_t sz) {
    void * mem = NULL;
    if (sz != 0) {
        mem = malloc(sz);
    }
    return mem;
}

static inline void xFree(void * mem) {
    if (mem != NULL) {
        free(mem);
    }
    return ;
}

static inline void * xRealloc(void * mem, size_t sz) {
    if (mem == NULL) {
        return NULL;
    } else if (sz == 0) {
        xFree(mem);
        return NULL;
    } else {
        void * res = realloc(mem, sz);
        wbAssert(res != NULL);
        return res;
    }
}


#define wbNew(type)                 ((type *) wbMalloc(sizeof(type)))
#define wbNewArray(type, len)       ((type * ) wbMalloc((len) * sizeof(type)))
#define wbMalloc(sz)				xMalloc(sz)
#define wbDelete(var)               wbFree(var)
#define wbFree(var)                 xFree(var); var = NULL
#define wbRealloc(var, newSize)     xRealloc(var, newSize)
#define wbReallocArray(t, m, n)     ((t*) xRealloc(m, n*sizeof(t)))


#endif /* WB_USE_CUSTOM_MALLOC */

#endif /* __WB_MALLOC_H__ */


