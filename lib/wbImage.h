

#ifndef __IMAGE_H__
#define __IMAGE_H__

typedef struct st_wbImage_t {
    int width;
    int height;
    int channels;
    int pitch;
    float * data;
} * wbImage_t;

#define wbImage_channels                3

#define wbImage_getWidth(img)			((img)->width)
#define wbImage_getHeight(img)			((img)->height)
#define wbImage_getChannels(img)		((img)->channels)
#define wbImage_getPitch(img)			((img)->pitch)
#define wbImage_getData(img)			((img)->data)

#define wbImage_setWidth(img, val)		(wbImage_getWidth(img) = val)
#define wbImage_setHeight(img, val)		(wbImage_getHeight(img) = val)
#define wbImage_setChannels(img, val)	(wbImage_getChannels(img) = val)
#define wbImage_setPitch(img, val)		(wbImage_getPitch(img) = val)
#define wbImage_setData(img, val)		(wbImage_getData(img) = val)


typedef void (*wbImage_onSameFunction_t)(string str);
wbImage_t wbImage_new(int width, int height, int channels);
wbImage_t wbImage_new(int width, int height);
void wbImage_delete(wbImage_t img);
wbBool wbImage_sameQ(wbImage_t a, wbImage_t b, wbImage_onSameFunction_t onUnSame);
wbBool wbImage_sameQ(wbImage_t a, wbImage_t b);



#endif /* __IMAGE_H__ */
