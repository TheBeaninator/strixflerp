/* A preloaded stand-in for what the StrixFold executor does to its process's libraries between flerp_open and
 * flerp_close, with no device and no weights (card 91a9cda544): the engine dlopens libhipblas.so.3
 * (zig/src/families/dsv41/hipblas.zig), which brings therock's libomp.so, and dlcloses it at close. libomp.so is
 * DF_1_NODELETE, so it stays, and its __do_fini faults at libc's exit(). The constructor runs before the host's main.
 * Used by tests/test_strixfold_host_exit.py and gate/k_close/run.sh. FLERP_EXIT_SHIM_LIB names another library. */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

__attribute__((constructor)) static void hipblas_opened_and_closed(void) {
    const char *lib = getenv("FLERP_EXIT_SHIM_LIB");
    if (!lib || !*lib) lib = "libhipblas.so.3";
    void *h = dlopen(lib, RTLD_LAZY | RTLD_LOCAL);
    fprintf(stderr, "shim: dlopen %s: %s\n", lib, h ? "ok" : dlerror());
    if (h) fprintf(stderr, "shim: dlclose: %d\n", dlclose(h));
}
