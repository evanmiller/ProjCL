
#include <stdio.h>
#include <math.h>
#include <string.h>

#include <OpenCL/OpenCL.h>
#include <sys/time.h>
#include "../src/projcl.h"

int compile_module(PLContext *ctx, unsigned int module, char *name) {
    PLCode *code = NULL;
    cl_int error = CL_SUCCESS;
    printf("-- %s... ", name);
    code = pl_compile_code(ctx, "kernel", module, &error);
    if (code == NULL) {
        printf("failed (%d)\n", error);
        return 1;
    }
    printf("ok\n");
    pl_release_code(code);

    return 0;
}

int main(int argc, char **argv) {
    cl_int error = CL_SUCCESS;
    PLCode *code = NULL;
    PLContext *ctx = pl_context_init(CL_DEVICE_TYPE_CPU, &error);
    if (ctx == NULL) {
        printf("Failed to initialize context: %d\n", error);
        return 1;
    }

    printf("Compiling individual modules...\n");
    int failures = 0;
    failures += compile_module(ctx, PL_MODULE_DATUM, "datum");
    failures += compile_module(ctx, PL_MODULE_GEODESIC, "geodesic");
    failures += compile_module(ctx, PL_MODULE_WARP, "warp");
    failures += compile_module(ctx, PL_MODULE_ALBERS_EQUAL_AREA, "Albers Equal Area");
    failures += compile_module(ctx, PL_MODULE_AMERICAN_POLYCONIC, "American Polyconic");
    failures += compile_module(ctx, PL_MODULE_LAMBERT_AZIMUTHAL_EQUAL_AREA, "Lambert Azimuthal Equal Area");
    failures += compile_module(ctx, PL_MODULE_LAMBERT_CONFORMAL_CONIC, "Lambert Conformal Conic");
    failures += compile_module(ctx, PL_MODULE_MERCATOR, "Mercator");
    failures += compile_module(ctx, PL_MODULE_ROBINSON, "Robinson");
    failures += compile_module(ctx, PL_MODULE_TRANSVERSE_MERCATOR, "Transverse Mercator");
    failures += compile_module(ctx, PL_MODULE_WINKEL_TRIPEL, "Winkel Tripel");
    failures += compile_module(ctx, PL_MODULE_NEAREST_NEIGHBOR, "nearest neighbor");
    failures += compile_module(ctx, PL_MODULE_BILINEAR, "bilinear");
    failures += compile_module(ctx, PL_MODULE_BICUBIC, "bicubic");
    failures += compile_module(ctx, PL_MODULE_QUASI_BICUBIC, "quasi-bicubic");

    printf("Failures: %d\n", failures);

    code = pl_compile_code(ctx, "kernel", 0xFFFF, &error);
    error = pl_load_code(ctx, code);
    pl_release_code(code);
    if (error != CL_SUCCESS) {
        printf("Failed to load code: %d\n", error);
        return 1;
    }

    pl_unload_code(ctx);
    pl_context_free(ctx);

    return 0;
}
