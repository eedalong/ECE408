

#include <wb.h>

static inline float _min(float x, float y) { return x < y ? x : y; }

static inline float _max(float x, float y) { return x > y ? x : y; }

static inline float _clamp(float x, float start, float end) {
  return _min(_max(x, start), end);
}

wbImage_t wbImage_new(int width, int height, int channels) {
  float *data;
  wbImage_t img;

  img = wbNew(struct st_wbImage_t);

  wbImage_setWidth(img, width);
  wbImage_setHeight(img, height);
  wbImage_setChannels(img, channels);
  wbImage_setPitch(img, width * channels);

  data = wbNewArray(float, width *height *channels);

  wbImage_setData(img, data);
  return img;
}

wbImage_t wbImage_new(int width, int height) {
  return wbImage_new(width, height, wbImage_channels);
}

void wbImage_delete(wbImage_t img) {
  if (img != NULL) {
    if (wbImage_getData(img) != NULL) {
      wbDelete(wbImage_getData(img));
    }
    wbDelete(img);
  }
}

static inline void wbImage_setPixel(wbImage_t img, int x, int y, int c,
                                    float val) {
  float *data = wbImage_getData(img);
  int channels = wbImage_getChannels(img);
  int pitch = wbImage_getPitch(img);

  data[y * pitch + x * channels + c] = val;

  return;
}

static inline float wbImage_getPixel(wbImage_t img, int x, int y, int c) {
  float *data = wbImage_getData(img);
  int channels = wbImage_getChannels(img);
  int pitch = wbImage_getPitch(img);

  return data[y * pitch + x * channels + c];
}

wbBool wbImage_sameQ(wbImage_t a, wbImage_t b,
                     wbImage_onSameFunction_t onUnSame) {
  
  std::cout<<"check solution image"<<std::endl;
  for(int row = 0; row < 5; row++){
      for(int col = 0; col < 5; col++){
          std::cout<<wbImage_getPixel(a, col, row, 0)<<", ";
      }
      std::cout<<std::endl;
  }

  std::cout<<"check expected image"<<std::endl;
  for(int row = 0; row < 5; row++){
      for(int col = 0; col < 5; col++){
          std::cout<<wbImage_getPixel(b, col, row, 0)<<", ";
      }
      std::cout<<std::endl;
  }

  if (a == NULL || b == NULL) {
    wbLog(ERROR, "Comparing null images.");
    return wbFalse;
  } else if (a == b) {
    return wbTrue;
  } else if (wbImage_getWidth(a) != wbImage_getWidth(b)) {
    wbLog(ERROR, "Image widths do not match.");
    return wbFalse;
  } else if (wbImage_getHeight(a) != wbImage_getHeight(b)) {
    wbLog(ERROR, "Image heights do not match.");
    return wbFalse;
  } else if (wbImage_getChannels(a) != wbImage_getChannels(b)) {
    wbLog(ERROR, "Image channels do not match.");
    return wbFalse;
  } else {
    float *aData, *bData;
    int width, height, channels;
    int ii, jj, kk;

    aData = wbImage_getData(a);
    bData = wbImage_getData(b);

    wbAssert(aData != NULL);
    wbAssert(bData != NULL);

    width = wbImage_getWidth(a);
    height = wbImage_getHeight(a);
    channels = wbImage_getChannels(a);

    for (ii = 0; ii < height; ii++) {
      for (jj = 0; jj < width; jj++) {
        for (kk = 0; kk < channels; kk++) {
          float x, y;
          if (channels <= 3) {
            x = _clamp(*aData++, 0, 1);
            y = _clamp(*bData++, 0, 1);
          } else {
            x = *aData++;
            y = *bData++;
          }
          if (wbUnequalQ(x, y)) {
            
            if (onUnSame != NULL) {
              string str = wbString("Image pixels do not match at position (",
                                    wbString(ii, ", ", jj, ", ", kk, "). [ "),
                                    wbString(x, ", ", y, "]"));
              onUnSame(str);
            }
            return wbFalse;
          }
        }
      }
    }
    return wbTrue;
  }
}

static void wbImage_onUnsameFunction(string str) { wbLog(ERROR, str); }

wbBool wbImage_sameQ(wbImage_t a, wbImage_t b) {
  return wbImage_sameQ(a, b, wbImage_onUnsameFunction);
}
