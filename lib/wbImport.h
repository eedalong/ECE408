
#ifndef __WB_IMPORT_H__
#define __WB_IMPORT_H__

#include	<wbImage.h>

typedef enum en_wbImportKind_t {
    wbImportKind_unknown = -1,
    wbImportKind_raw = 0x1000,
    wbImportKind_csv,
    wbImportKind_tsv,
    wbImportKind_ppm
} wbImportKind_t;

typedef enum en_wbImportType_t {
    wbImportType_char = 0,
    wbImportType_int,
    wbImportType_float,
    wbImportType_double
} wbImportType_t;

typedef struct st_wbImportCSV_t {
    int rows;
    int columns;
    void * data;
    wbFile_t file;
    char seperator;
} * wbImportCSV_t;

#define wbImportCSV_getRowCount(csv)			((csv)->rows)
#define wbImportCSV_getColumnCount(csv)			((csv)->columns)
#define wbImportCSV_getData(csv)				((csv)->data)
#define wbImportCSV_getFile(csv)				((csv)->file)
#define wbImportCSV_getSeperator(csv)			((csv)->seperator)

#define wbImportCSV_setRowCount(csv, val)		(wbImportCSV_getRowCount(csv) = val)
#define wbImportCSV_setColumnCount(csv, val)	(wbImportCSV_getColumnCount(csv) = val)
#define wbImportCSV_setData(csv, val)			(wbImportCSV_getData(csv) = val)
#define wbImportCSV_setSeperator(csv, val)		(wbImportCSV_getSeperator(csv) = val)


typedef struct st_wbImportRaw_t {
    int rows;
    int columns;
    void * data;
    wbFile_t file;
} * wbImportRaw_t;

#define wbImportRaw_getRowCount(raw)			((raw)->rows)
#define wbImportRaw_getColumnCount(raw)			((raw)->columns)
#define wbImportRaw_getData(raw)				((raw)->data)
#define wbImportRaw_getFile(raw)				((raw)->file)

#define wbImportRaw_setRowCount(raw, val)		(wbImportRaw_getRowCount(raw) = val)
#define wbImportRaw_setColumnCount(raw, val)	(wbImportRaw_getColumnCount(raw) = val)
#define wbImportRaw_setData(raw, val)			(wbImportRaw_getData(raw) = val)

typedef struct st_wbImport_t {
    wbImportKind_t kind;
    union {
        wbImportRaw_t raw;
        wbImportCSV_t csv;
        wbImage_t img;
    } container;
} wbImport_t;

#define wbImport_getKind(imp)					((imp).kind)
#define wbImport_getContainer(imp)				((imp).container)
#define wbImport_getRaw(imp)					(wbImport_getContainer(imp).raw)
#define wbImport_getCSV(imp)					(wbImport_getContainer(imp).csv)
#define wbImport_getImage(imp)					(wbImport_getContainer(imp).img)

#define wbImport_setKind(imp, val)				(wbImport_getKind(imp) = val)
#define wbImport_setRaw(imp, val)				(wbImport_getRaw(imp) = val)
#define wbImport_setCSV(imp, val)				(wbImport_getCSV(imp) = val)
#define wbImport_setImage(imp, val)				(wbImport_getImage(imp) = val)


void * wbImport(const char * file, int * rows, int * columns);
void * wbImport(const char * file, int * rows);
wbImage_t wbImport(const char * file);

#endif /* __WB_IMPORT_H__ */


