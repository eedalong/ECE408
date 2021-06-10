

#ifndef __wbPPM_H__
#define __wbPPM_H__

wbImage_t wbPPM_import(const char * filename);
void wbPPM_export(const char * filename, wbImage_t img);

#endif /* __wbPPM_H__ */

