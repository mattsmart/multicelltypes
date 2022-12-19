import os
from multiprocessing import cpu_count


# LIBRARY GLOBAL MODS
def set_deep_threads():
    system_threads = cpu_count()
    threads_optimized = {4: "2",    # for laptop
                         12: "2",   # for desktop ryzen 2600  TODO test
                         24: "1",   # for desktop ryzen 5900x TODO test
                         64: "1",   # for workstation
                         80: "1"}   # for cluster
    threads_int = threads_optimized[system_threads]
    print("init_multiprocessing.py - Setting os.environ['OPENBLAS_NUM_THREADS'] (and others) to", threads_int)
    os.environ['MKL_NUM_THREADS'] = threads_int
    os.environ["OMP_NUM_THREADS"] = threads_int
    os.environ["NUMEXPR_NUM_THREADS"] = threads_int
    os.environ["OPENBLAS_NUM_THREADS"] = threads_int
    #os.environ["MKL_THREADING_LAYER"] = "sequential"  # this should be off if NUM_THREADS is not 1
    return


set_deep_threads()  # this must be set before importing numpy for the first time (during execution)
