

#include <wb.h>

#define wbFile_maxCount 256

static wbFile_t wbFile_handles[wbFile_maxCount];

static int wbFile_nextIndex(void) {
  int ii;
  for (ii = 0; ii < wbFile_maxCount; ii++) {
    if (wbFile_handles[ii] == NULL) {
      return ii;
    }
  }
  wbLog(ERROR, "Ran out of file handles.");
  wbExit();
  return -1;
}

wbFile_t wbFile_new(void) {
  int idx = wbFile_nextIndex();
  wbFile_t file = wbNew(struct st_wbFile_t);

  wbAssert(idx >= 0);

  wbFile_setIndex(file, idx);
  wbFile_setFileName(file, NULL);
  wbFile_setMode(file, NULL);
  wbFile_setFileHandle(file, NULL);
  wbFile_setData(file, NULL);

  wbFile_handles[idx] = file;

  return file;
}

void wbFile_delete(wbFile_t file) {
  if (file != NULL) {
    int idx = wbFile_getIndex(file);
    if (wbFile_getFileName(file) != NULL) {
      wbDelete(wbFile_getFileName(file));
    }
    if (wbFile_getMode(file) != NULL) {
      wbDelete(wbFile_getMode(file));
    }
    if (wbFile_getFileHandle(file) != NULL) {
      fclose(wbFile_getFileHandle(file));
    }
    if (idx >= 0) {
      wbAssert(wbFile_handles[idx] == file);
      wbFile_handles[idx] = NULL;
    }
    if (wbFile_getData(file) != NULL) {
      wbDelete(wbFile_getData(file));
    }
    wbDelete(file);
  }
}

void wbFile_init(void) {
  int ii;

  for (ii = 0; ii < wbFile_maxCount; ii++) {
    wbFile_handles[ii] = NULL;
  }
}

void wbFile_atExit(void) {
  int ii;

  for (ii = 0; ii < wbFile_maxCount; ii++) {
    if (wbFile_handles[ii] != NULL) {
      wbFile_delete(wbFile_handles[ii]);
    }
  }
}

int wbFile_count(void) {
  int ii, count = 0;

  for (ii = 0; ii < wbFile_maxCount; ii++) {
    if (wbFile_handles[ii] != NULL) {
      count++;
    }
  }
  return count;
}

wbFile_t wbFile_open(const char *fileName, const char *mode) {
  FILE *handle;
  wbFile_t file;

  if (fileName == NULL) {
    return NULL;
  }

  handle = fopen(fileName, mode);
  if (handle == NULL) {
    wbLog(ERROR, "Failed to open ", file, " in mode ", mode);
    return NULL;
  }

  file = wbFile_new();
  wbFile_setFileName(file, wbString_duplicate(fileName));
  wbFile_setMode(file, wbString_duplicate(mode));
  wbFile_setFileHandle(file, handle);

  return file;
}

wbFile_t wbFile_open(const char *fileName) {
  return wbFile_open(fileName, "r");
}

void wbFile_close(wbFile_t file) { wbFile_delete(file); }

char *wbFile_read(wbFile_t file, size_t size, size_t count) {
  size_t res;
  char *buffer;
  size_t bufferLen;
  FILE *handle;

  if (file == NULL) {
    return NULL;
  }
#ifndef LAZY_FILE_LOAD
  if (wbFile_getData(file) != NULL) {
    char *data = wbFile_getData(file) + wbFile_getDataOffset(file);
    wbFile_setDataOffset(file, wbFile_getDataOffset(file) + size * count);
    return data;
  }
#endif /* LAZY_FILE_LOAD */

  handle = wbFile_getFileHandle(file);
  bufferLen = size * count + 1;
  buffer = wbNewArray(char, bufferLen);

  res = fread(buffer, size, count, handle);
  // make valid C string
  buffer[size * res] = '\0';

  return buffer;
}

char *wbFile_read(wbFile_t file, size_t len) {
  char *buffer = wbFile_read(file, sizeof(char), len);
  return buffer;
}

void wbFile_rewind(wbFile_t file) {

  if (file == NULL) {
    return;
  }

  if (wbFile_getData(file) == NULL) {
    FILE *handle;
    handle = wbFile_getFileHandle(file);
    wbAssert(handle != NULL);
    rewind(handle);
  }
#ifndef LAZY_FILE_LOAD
  else {
    wbFile_setDataOffset(file, 0);
  }
#endif

  return;
}

size_t wbFile_size(wbFile_t file) {
  size_t len;
  FILE *handle;

  if (file == NULL) {
    return 0;
  }
#ifndef LAZY_FILE_LOAD
  if (wbFile_getData(file) != NULL) {
    if (wbFile_getLength(file) == 0) {
      wbFile_setLength(file, strlen(wbFile_getData(file)));
    }
    return wbFile_getLength(file);
  }
#endif /* LAZY_FILE_LOAD */

  handle = wbFile_getFileHandle(file);

  fseek(handle, 0, SEEK_END);
  len = ftell(handle);
  rewind(handle);

  return len;
}

char *wbFile_read(wbFile_t file) {
  size_t len;

  if (file == NULL) {
    return NULL;
  }

  len = wbFile_size(file);

  if (len == 0) {
    return NULL;
  }

  wbFile_setLength(file, len);

  return wbFile_read(file, len);
}

#define MAX_CHARS_PER_LINE (1 << 17)

static char buffer[MAX_CHARS_PER_LINE];

char *wbFile_readLine(wbFile_t file) {

  if (file == NULL) {
    return NULL;
  }
#ifdef LAZY_FILE_LOAD
  FILE *handle;
  memset(buffer, 0, MAX_CHARS_PER_LINE);

  handle = wbFile_getFileHandle(file);

  if (fgets(buffer, MAX_CHARS_PER_LINE - 1, handle)) {
    return buffer;
  } else {
    // wbLog(ERROR, "Was not able to read line from ",
    // wbFile_getFileName(file));
    return NULL;
  }
#else
  size_t newOffset;
  size_t lenToNewLine = 0;
  const char *tmp;

  if (wbFile_getData(file) == NULL) {
    wbFile_setData(file, wbFile_read(file));
    fclose(wbFile_getFileHandle(file));
    wbFile_setFileHandle(file, NULL);
    wbFile_setDataOffset(file, 0);
    wbFile_setLength(file, strlen(wbFile_getData(file)));
  }

  memset(buffer, 0, MAX_CHARS_PER_LINE);

  if (wbFile_getDataOffset(file) >= wbFile_getLength(file)) {
    return NULL;
  }

  newOffset = wbFile_getDataOffset(file);
  tmp = wbFile_getData(file) + wbFile_getDataOffset(file);
  while (newOffset < wbFile_getLength(file) && *tmp != '\n') {
    tmp++;
    lenToNewLine++;
    newOffset++;
  }

  memcpy(buffer, wbFile_getData(file) + wbFile_getDataOffset(file),
         lenToNewLine);
  wbFile_setDataOffset(file, newOffset + 1);

  return buffer;
#endif
}

void wbFile_write(wbFile_t file, const void *buffer, size_t size,
                  size_t count) {
  size_t res;
  FILE *handle;

  if (file == NULL) {
    return;
  }

  handle = wbFile_getFileHandle(file);

  res = fwrite(buffer, size, count, handle);
  if (res != count) {
    wbLog(ERROR, "Failed to write data to ", wbFile_getFileName(file));
  }

  return;
}

void wbFile_write(wbFile_t file, const void *buffer, size_t len) {
  wbFile_write(file, buffer, sizeof(char), len);
  return;
}

void wbFile_write(wbFile_t file, const char *buffer) {
  size_t len;

  len = strlen(buffer);
  wbFile_write(file, buffer, len);

  return;
}

void wbFile_writeLine(wbFile_t file, const char *buffer0) {
  string buffer = wbString(buffer0, "\n");
  wbFile_write(file, buffer.c_str());
}

void wbFile_write(wbFile_t file, string buffer) {
  wbFile_write(file, buffer.c_str());
}

void wbFile_writeLine(wbFile_t file, string buffer0) {
  string buffer = buffer0 + "\n";
  wbFile_write(file, buffer.c_str());
}

wbBool wbFile_existsQ(const char *path) {
  if (path == NULL) {
    return wbFalse;
  } else {
    FILE *file = fopen(path, "r");
    if (file != NULL) {
      fclose(file);
      return wbTrue;
    }
    return wbFalse;
  }
}

char *wbFile_extension(const char *file) {
  char *extension;
  char *extensionLower;
  char *end;
  size_t len;

  len = strlen(file);
  end = (char *)&file[len - 1];
  while (*end != '.') {
    end--;
  }
  if (*end == '.') {
    end++;
  }

  extension = wbString_duplicate(end);
  extensionLower = wbString_toLower(extension);
  wbDelete(extension);

  return extensionLower;
}
