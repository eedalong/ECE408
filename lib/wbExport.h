

#ifndef __WB_EXPORT_H__
#define __WB_EXPORT_H__

#include	<wb.h>
#include	<wbFile.h>
#include	<wbPPM.h>

typedef enum en_wbExportKind_t {
    wbExportKind_unknown = -1,
    wbExportKind_raw = 0x1000,
    wbExportKind_csv,
    wbExportKind_tsv,
    wbExportKind_ppm,
} wbExportKind_t;

typedef struct st_wbExportRaw_t {
    int rows;
    int columns;
    wbFile_t file;
} * wbExportRaw_t;

#define wbExportRaw_getColumnCount(raw)			((raw)->columns)
#define wbExportRaw_getRowCount(raw)			((raw)->rows)
#define wbExportRaw_getFile(raw)				((raw)->file)

#define wbExportRaw_setRowCount(raw, val)		(wbExportRaw_getRowCount(raw) = val)
#define wbExportRaw_setColumnCount(raw, val)	(wbExportRaw_getColumnCount(raw) = val)


typedef struct st_wbExportCVS_t {
    int rows;
    int columns;
    wbFile_t file;
    char seperator;
} * wbExportCSV_t;

#define wbExportCSV_getRowCount(csv)			((csv)->rows)
#define wbExportCSV_getColumnCount(csv)			((csv)->columns)
#define wbExportCSV_getFile(csv)				((csv)->file)
#define wbExportCSV_getSeperator(csv)			((csv)->seperator)

#define wbExportCSV_setRowCount(csv, val)		(wbExportCSV_getRowCount(csv) = val)
#define wbExportCSV_setColumnCount(csv, val)	(wbExportCSV_getColumnCount(csv) = val)
#define wbExportCSV_setSeperator(csv, val)		(wbExportCSV_getSeperator(csv) = val)

typedef struct st_wbExport_t {
    wbExportKind_t kind;
    union {
        wbExportRaw_t raw;
        wbExportCSV_t csv;
        wbImage_t img;
    } container;
    char * file;
} wbExport_t;

#define wbExport_getKind(exprt)					((exprt).kind)
#define wbExport_getContainer(exprt)			((exprt).container)
#define wbExport_getRaw(exprt)					(wbExport_getContainer(exprt).raw)
#define wbExport_getCSV(exprt)					(wbExport_getContainer(exprt).csv)
#define wbExport_getImage(exprt)				(wbExport_getContainer(exprt).img)
#define wbExport_getFile(exprt)					((exprt).file)

#define wbExport_setKind(exprt, val)			(wbExport_getKind(exprt) = val)
#define wbExport_setRaw(exprt, val)				(wbExport_getRaw(exprt) = val)
#define wbExport_setCSV(exprt, val)				(wbExport_getCSV(exprt) = val)
#define wbExport_setImage(exprt, val)			(wbExport_getImage(exprt) = val)
#define wbExport_setFile(exprt, val)			(wbExport_getFile(exprt) = val)


void wbExport(const char * file, int * data, int rows, int columns);
void wbExport(const char * file, int * data, int rows);
void wbExport(const char * file, wbReal_t * data, int rows, int columns);
void wbExport(const char * file, wbReal_t * data, int rows);
void wbExport(const char * file, wbImage_t img);


#endif /* __WB_EXPORT_H__ */


