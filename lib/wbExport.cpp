
#include	<wb.h>


static inline void wbExportRaw_setFile(wbExportRaw_t raw, const char * path) {
    if (raw != NULL) {
        if (wbExportRaw_getFile(raw) != NULL) {
            wbFile_delete(wbExportRaw_getFile(raw));
        }
        if (path != NULL) {
            wbExportRaw_getFile(raw) = wbFile_open(path, "w");
        } else {
            wbExportRaw_getFile(raw) = NULL;
        }
    }

    return ;
}

static inline wbExportRaw_t wbExportRaw_new(void) {
    wbExportRaw_t raw;

    raw = wbNew(struct st_wbExportRaw_t);

    wbExportRaw_getFile(raw) = NULL;
    wbExportRaw_setRowCount(raw, -1);
    wbExportRaw_setColumnCount(raw, -1);

    return raw;
}

static inline void wbExportRaw_delete(wbExportRaw_t raw) {
    if (raw != NULL) {
        wbExportRaw_setFile(raw, NULL);
        wbDelete(raw);
    }
    return ;
}

static inline void wbExportRaw_write(wbExportRaw_t raw, void * data, int rows, int columns, wbBool asIntegerQ) {
    int ii, jj;
    FILE * handle;
    wbFile_t file;

    if (raw == NULL || wbExportRaw_getFile(raw) == NULL) {
        return ;
    }

    file = wbExportRaw_getFile(raw);

    handle = wbFile_getFileHandle(file);

    if (handle == NULL) {
        return ;
    }

    if (columns == 1) {
        fprintf(handle, "%d\n", rows);
    } else {
        fprintf(handle, "%d %d\n", rows, columns);
    }

    for (ii = 0; ii < rows; ii++) {
        for (jj = 0; jj < columns; jj++) {
            if (asIntegerQ) {
                int elem = ((int *) data)[ii*columns + jj];
                fprintf(handle, "%d", elem);
            } else {
                wbReal_t elem = ((wbReal_t *) data)[ii*columns + jj];
                fprintf(handle, "%f", elem);
            }
            if (jj == columns - 1) {
                fprintf(handle, "\n");
            } else {
                fprintf(handle, " ");
            }
        }
    }

    return ;

}

static inline void wbExportRaw_writeAsInteger(wbExportRaw_t raw, int * data, int rows, int columns) {
    wbExportRaw_write(raw, data, rows, columns, wbTrue);
    return ;
}

static inline void wbExportRaw_writeAsReal(wbExportRaw_t raw, wbReal_t * data, int rows, int columns) {
    wbExportRaw_write(raw, data, rows, columns, wbFalse);
    return ;
}

static inline void wbExportCSV_setFile(wbExportCSV_t csv, const char * path) {
    if (csv != NULL) {
        if (wbExportCSV_getFile(csv) != NULL) {
            wbFile_delete(wbExportCSV_getFile(csv));
        }
        if (path != NULL) {
            wbExportCSV_getFile(csv) = wbFile_open(path, "w+");
        } else {
            wbExportCSV_getFile(csv) = NULL;
        }
    }

    return ;
}

static inline wbExportCSV_t wbExportCSV_new(void) {
    wbExportCSV_t csv;

    csv = wbNew(struct st_wbExportCVS_t);

    wbExportCSV_getFile(csv) = NULL;
    wbExportCSV_setColumnCount(csv, -1);
    wbExportCSV_setRowCount(csv, -1);
    wbExportCSV_setSeperator(csv, '\0');

    return csv;
}

static inline void wbExportCSV_delete(wbExportCSV_t csv) {
    if (csv != NULL) {
        wbExportCSV_setFile(csv, NULL);
        wbDelete(csv);
    }
}

static inline void wbExportCSV_write(wbExportCSV_t csv, void * data, int rows, int columns, char sep, wbBool asIntegerQ) {
    int ii, jj;
    wbFile_t file;
    FILE * handle;
    char seperator[2];

    if (csv == NULL || wbExportCSV_getFile(csv) == NULL) {
        return ;
    }

    file = wbExportCSV_getFile(csv);

    handle = wbFile_getFileHandle(file);

    if (handle == NULL) {
        return ;
    }

    if (sep == '\0') {
        seperator[0] = ',';
    } else {
        seperator[0] = sep;
    }
    seperator[1] = '\0';

    for (ii = 0; ii < rows; ii++) {
        for (jj = 0; jj < columns; jj++) {
            if (asIntegerQ) {
                int elem = ((int *) data)[ii*columns + jj];
                fprintf(handle, "%d", elem);
            } else {
                wbReal_t elem = ((wbReal_t *) data)[ii*columns + jj];
                fprintf(handle, "%f", elem);
            }
            if (jj == columns - 1) {
                fprintf(handle, "\n");
            } else {
                fprintf(handle, "%s", seperator);
            }
        }
    }

    return ;

}

static inline void wbExportCSV_writeAsInteger(wbExportCSV_t csv, int * data, int rows, int columns) {
    char seperator;

    if (csv == NULL) {
        return ;
    }

    seperator = wbExportCSV_getSeperator(csv);

    wbExportCSV_write(csv, data, rows, columns, seperator, wbTrue);

    return ;
}

static inline void wbExportCSV_writeAsReal(wbExportCSV_t csv, wbReal_t * data, int rows, int columns) {
    char seperator;

    if (csv == NULL) {
        return ;
    }

    seperator = wbExportCSV_getSeperator(csv);

    wbExportCSV_write(csv, data, rows, columns, seperator, wbFalse);

    return ;
}


static inline wbExport_t wbExport_open(const char * file, wbExportKind_t kind) {
    wbExport_t exprt;

    if (file == NULL) {
        wbLog(ERROR, "Go NULL for file value.");
        wbExit();
    }

    wbExport_setFile(exprt, NULL);
    wbExport_setKind(exprt, kind);

    if (kind == wbExportKind_raw) {
        wbExportRaw_t raw = wbExportRaw_new();
        wbExportRaw_setFile(raw, file);
        wbExport_setRaw(exprt, raw);
    } else if (kind == wbExportKind_tsv || kind == wbExportKind_csv) {
        wbExportCSV_t csv = wbExportCSV_new();
        if (kind == wbExportKind_csv) {
            wbExportCSV_setSeperator(csv, ',');
        } else {
            wbExportCSV_setSeperator(csv, '\t');
        }
        wbExportCSV_setFile(csv, file);
        wbExport_setCSV(exprt, csv);
    } else if (kind == wbExportKind_ppm) {
        wbExport_setFile(exprt, wbString_duplicate(file));
    } else {
        wbLog(ERROR, "Invalid export type.");
        wbExit();
    }

    return exprt;
}

static inline wbExport_t wbExport_open(const char * file, const char * type0) {
    wbExport_t exprt;
    wbExportKind_t kind;
    char * type;

    type = wbString_toLower(type0);

    if (wbString_sameQ(type, "cvs")) {
        kind = wbExportKind_csv;
    } else if (wbString_sameQ(type, "tsv")) {
        kind = wbExportKind_tsv;
    } else if (wbString_sameQ(type, "raw") || wbString_sameQ(type, "dat")) {
        kind = wbExportKind_raw;
    } else if (wbString_sameQ(type, "ppm")) {
        kind = wbExportKind_ppm;
    } else {
        wbLog(ERROR, "Invalid export type ", type0);
        wbExit();
    }

    exprt = wbExport_open(file, kind);

    wbDelete(type);

    return exprt;
}

static inline void wbExport_close(wbExport_t exprt) {
    wbExportKind_t kind;

    kind = wbExport_getKind(exprt);

    if (wbExport_getFile(exprt)) {
        wbDelete(wbExport_getFile(exprt));
    }

    if (kind == wbExportKind_tsv || kind == wbExportKind_csv) {
        wbExportCSV_t csv = wbExport_getCSV(exprt);
        wbExportCSV_delete(csv);
        wbExport_setCSV(exprt, NULL);
    } else if (kind == wbExportKind_raw) {
        wbExportRaw_t raw = wbExport_getRaw(exprt);
        wbExportRaw_delete(raw);
        wbExport_setRaw(exprt, NULL);
    } else if (kind == wbExportKind_ppm) {
    } else {
        wbLog(ERROR, "Invalid export type.");
        wbExit();
    }
    return ;
}

static inline void wbExport_writeAsImage(wbExport_t exprt, wbImage_t img) {
    wbExportKind_t kind;

    kind = wbExport_getKind(exprt);

    wbAssert(kind == wbExportKind_ppm);

    wbPPM_export(wbExport_getFile(exprt), img);

    return ;
}

static inline void wbExport_write(wbExport_t exprt, void * data, int rows, int columns, char sep, wbBool asIntegerQ) {
    wbExportKind_t kind;

    kind = wbExport_getKind(exprt);
    if (kind == wbExportKind_tsv || kind == wbExportKind_csv) {
        wbExportCSV_t csv = wbExport_getCSV(exprt);
        wbExportCSV_write(csv, data, rows, columns, sep, asIntegerQ);
    } else if (kind == wbExportKind_raw) {
        wbExportRaw_t raw = wbExport_getRaw(exprt);
        wbExportRaw_write(raw, data, rows, columns, asIntegerQ);
    } else {
        wbLog(ERROR, "Invalid export type.");
        wbExit();
    }
    return ;
}

static inline void wbExport_write(wbExport_t exprt, void * data, int rows, int columns, wbBool asIntegerQ) {
    wbExport_write(exprt, data, rows, columns, ',', asIntegerQ);
}

static inline void wbExport_writeAsInteger(wbExport_t exprt, int * data, int rows, int columns) {
    wbExport_write(exprt, data, rows, columns, wbTrue);
    return ;
}

static inline void wbExport_writeAsReal(wbExport_t exprt, wbReal_t * data, int rows, int columns) {
    wbExport_write(exprt, data, rows, columns, wbFalse);
    return ;
}

static wbExportKind_t _parseExportExtension(const char * file) {
    char * extension;
    wbExportKind_t kind;

    extension = wbFile_extension(file);

    if (wbString_sameQ(extension, "csv")) {
        kind = wbExportKind_csv;
    } else if (wbString_sameQ(extension, "tsv")) {
        kind = wbExportKind_tsv;
    } else if (wbString_sameQ(extension, "raw") ||
               wbString_sameQ(extension, "dat")) {
        kind = wbExportKind_raw;
    } else if (wbString_sameQ(extension, "ppm")) {
        kind = wbExportKind_ppm;
    } else {
        kind = wbExportKind_unknown;
        wbLog(ERROR, "File ", file, " does not have a compatible extension.");
    }

    wbDelete(extension);

    return kind;
}

void wbExport(const char * file, int * data, int rows) {
    wbExport(file, data, rows, 1);
    return ;
}

void wbExport(const char * file, int * data, int rows, int columns) {
    wbExportKind_t kind;
    wbExport_t exprt;


    if (file == NULL) {
        fprintf(stderr, "Failed to import file.\n");
        wbExit();
    }


    kind = _parseExportExtension(file);
    exprt = wbExport_open(file, kind);

    wbExport_writeAsInteger(exprt, data, rows, columns);
    wbExport_close(exprt);
}

void wbExport(const char * file, wbReal_t * data, int rows) {
    wbExport(file, data, rows, 1);
    return ;
}

void wbExport(const char * file, wbReal_t * data, int rows, int columns) {
    wbExportKind_t kind;
    wbExport_t exprt;


    if (file == NULL) {
        fprintf(stderr, "Failed to import file.\n");
        wbExit();
    }


    kind = _parseExportExtension(file);
    exprt = wbExport_open(file, kind);

    wbExport_writeAsReal(exprt, data, rows, columns);
    wbExport_close(exprt);
}

void wbExport(const char * file, wbImage_t img) {
    wbExportKind_t kind;
    wbExport_t exprt;


    if (file == NULL) {
        fprintf(stderr, "Failed to import file.\n");
        wbExit();
    }


    kind = _parseExportExtension(file);
    exprt = wbExport_open(file, kind);

    wbAssert(kind == wbExportKind_ppm);

    wbExport_writeAsImage(exprt, img);
    wbExport_close(exprt);
}


