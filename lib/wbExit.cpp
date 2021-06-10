
#include    <wb.h>

void wb_atExit(void) {

#ifdef WB_USE_CUDA
    cudaDeviceSynchronize();
#endif /* WB_USE_CUDA */


    cout << "{\n"
         << wbString_quote("timer") << ":" << wbTimer_toJSON() << ",\n"
         << wbString_quote("logger") << ":" << wbLogger_toJSON() << ",\n";
#ifdef WB_USE_CUDA
    cout << wbString_quote("cuda_memory") << ":" << _cudaMallocSize << ",\n";
#endif
    if (solutionJSON) {
        cout << wbString_quote("solution_exists") << ": true,\n";
        cout << wbString_quote("solution") << ":" << solutionJSON << "\n";
    } else {
        cout << wbString_quote("solution_exists") << ": false\n";
    }
    cout << "}" << endl;

    //wbTimer_delete(_timer);
    //wbLogger_delete(_logger);

    _timer = NULL;
    _logger = NULL;

    //wbFile_atExit();

#ifdef WB_DEBUG_MEMMGR_SUPPORT_STATS
    memmgr_print_stats();
#endif /* WB_DEBUG_MEMMGR_SUPPORT_STATS */

    //exit(0);

    //assert(0);

    return ;
}


