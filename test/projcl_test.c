
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <inttypes.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

#include <sys/time.h>
#include <projcl/projcl.h>
#include <projcl/projcl_warp.h>

#ifdef HAVE_PROJ4
#include <proj_api.h>
#endif

#ifndef RAD_TO_DEG
#define RAD_TO_DEG   57.29577951308232
#endif

#define TEST_POINTS 200000
#define DEGREES_TOL 2.5/3600. // two and a half arcseconds (about 60m at the equator)
#define METERS_TOL 11.0      // eleven meters

#define MIN_LAT -40.0 // shifted up for benefit of conic projections
#define MAX_LAT 85.0

#define MIN_LON -60.0 // kind of tight for benefit of transverse mercator
#define MAX_LON 60.0

typedef struct test_params_s {
  char   name[80];
  int    ell;
  double lon0;
  double lat0;
  double rlat1;
  double rlat2;
} test_params_t;

typedef struct test_group_s {
    char            label[80];
    PLProjection    proj;
    char            proj4_name[80];
    test_params_t   tests[10];
} test_group_t;

test_group_t _test_groups[] = {
    {
        .label = "Albers Equal Area",
        .proj4_name = "aea",
        .proj = PL_PROJECT_ALBERS_EQUAL_AREA,
        .tests = {
            { .name = "Spherical, centered", 
                .ell = PL_SPHEROID_SPHERE, 
                .lon0 = 0.0,
                .lat0 = 0.0,
                .rlat1 = 30.0,
                .rlat2 = 60.0 },

            { .name = "Spherical, off-center",
                .ell = PL_SPHEROID_SPHERE,
                .lon0 = 10.0,
                .lat0 = 10.0,
                .rlat1 = 30.0,
                .rlat2 = 60.0 },

            { .name = "Ellipsoidal, centered",
                .ell = PL_SPHEROID_WGS_84,
                .lon0 = 0.0,
                .lat0 = 0.0,
                .rlat1 = 30.0,
                .rlat2 = 60.0 },

            { .name = "Ellipsoidal, off-center",
                .ell = PL_SPHEROID_WGS_84,
                .lon0 = 10.0,
                .lat0 = 10.0,
                .rlat1 = 30.0,
                .rlat2 = 60.0 }
        }
    },
    {
        .label = "American Polyconic",
        .proj4_name = "poly",
        .proj = PL_PROJECT_AMERICAN_POLYCONIC,
        .tests = {
            { .name = "Spherical, centered",
                .ell = PL_SPHEROID_SPHERE,
                .lon0 = 0.0,
                .lat0 = 0.0,
                .rlat1 = NAN,
                .rlat2 = NAN },
            { .name = "Spherical, off-center",
                .ell = PL_SPHEROID_SPHERE,
                .lon0 = 10.0,
                .lat0 = 10.0,
                .rlat1 = NAN,
                .rlat2 = NAN },
            { .name = "Ellipsoidal, centered",
                .ell = PL_SPHEROID_WGS_84,
                .lon0 = 0.0,
                .lat0 = 0.0,
                .rlat1 = NAN,
                .rlat2 = NAN },
            { .name = "Ellipsoidal, off-center",
                .ell = PL_SPHEROID_WGS_84,
                .lon0 = 10.0,
                .lat0 = 10.0,
                .rlat1 = NAN,
                .rlat2 = NAN }
        }
    },
    {
        .label = "Lambert Azimuthal Equal Area",
        .proj4_name = "laea",
        .proj = PL_PROJECT_LAMBERT_AZIMUTHAL_EQUAL_AREA,
        .tests = {
            { .name = "Spherical, centered",
                .ell = PL_SPHEROID_SPHERE,
                .lon0 = 0.0,
                .lat0 = 0.0,
                .rlat1 = NAN,
                .rlat2 = NAN },
            { .name = "Spherical, off-center",
                .ell = PL_SPHEROID_SPHERE,
                .lon0 = 10.0,
                .lat0 = 10.0,
                .rlat1 = NAN,
                .rlat2 = NAN },
            { .name = "Ellipsoidal, centered",
                .ell = PL_SPHEROID_WGS_84,
                .lon0 = 0.0,
                .lat0 = 0.0,
                .rlat1 = NAN,
                .rlat2 = NAN },
            { .name = "Ellipsoidal, off-center",
                .ell = PL_SPHEROID_WGS_84,
                .lon0 = 10.0,
                .lat0 = 10.0,
                .rlat1 = NAN,
                .rlat2 = NAN }
        }
    },
    {
        .label = "Lambert Conformal Conic",
        .proj = PL_PROJECT_LAMBERT_CONFORMAL_CONIC,
        .proj4_name = "lcc",
        .tests = {
            { .name = "Spherical, centered",
                .ell = PL_SPHEROID_SPHERE,
                .lon0 = 0.0,
                .lat0 = 0.0,
                .rlat1 = 30.0,
                .rlat2 = 60.0 },
            { .name = "Spherical, off-center",
                .ell = PL_SPHEROID_SPHERE,
                .lon0 = 10.0,
                .lat0 = 10.0,
                .rlat1 = 30.0,
                .rlat2 = 60.0 },
#ifndef HAVE_PROJ4
            { .name = "Spherical, symmetric standard parallels",
                .ell = PL_SPHEROID_SPHERE,
                .lon0 = 0.0,
                .lat0 = 0.0,
                .rlat1 = -30.0,
                .rlat2 = 30.0 },
#endif
            { .name = "Ellipsoidal, centered",
                .ell = PL_SPHEROID_WGS_84,
                .lon0 = 0.0,
                .lat0 = 0.0,
                .rlat1 = 30.0,
                .rlat2 = 60.0 },
            { .name = "Ellipsoidal, off-center",
                .ell = PL_SPHEROID_WGS_84,
                .lon0 = 10.0,
                .lat0 = 10.0,
                .rlat1 = 30.0,
                .rlat2 = 60.0 },
#ifndef HAVE_PROJ4
            { .name = "Ellipsoidal, symmetric standard parallels",
                .ell = PL_SPHEROID_WGS_84,
                .lon0 = 0.0,
                .lat0 = 0.0,
                .rlat1 = -30.0,
                .rlat2 = 30.0 },
#endif
        },
    },
    {
        .label = "Mercator",
        .proj = PL_PROJECT_MERCATOR,
        .proj4_name = "merc",
        .tests = {
            { .name = "Spherical",
                .ell = PL_SPHEROID_SPHERE,
                .lon0 = NAN,
                .lat0 = NAN,
                .rlat1 = NAN,
                .rlat2 = NAN },
            { .name = "Ellipsoidal",
                .ell = PL_SPHEROID_WGS_84,
                .lon0 = NAN,
                .lat0 = NAN,
                .rlat1 = NAN,
                .rlat2 = NAN }
        }
    },
    {
        .label = "Oblique Stereographic",
        .proj = PL_PROJECT_OBLIQUE_STEREOGRAPHIC,
        .proj4_name = "sterea",
        .tests = {
            { .name = "Ellipsoidal, centered",
                .ell = PL_SPHEROID_WGS_84,
                .lon0 = 0.0,
                .lat0 = 0.0,
                .rlat1 = NAN,
                .rlat2 = NAN
            },
            { .name = "Ellipsoidal, off-center",
                .ell = PL_SPHEROID_WGS_84,
                .lon0 = 10.0,
                .lat0 = 10.0,
                .rlat1 = NAN,
                .rlat2 = NAN
            }
        }
    },
    {
        .label = "Robinson",
        .proj = PL_PROJECT_ROBINSON,
        .proj4_name = "robin",
        .tests = {
            { .name = "Spherical",
                .ell = PL_SPHEROID_SPHERE,
                .lon0 = NAN,
                .lat0 = NAN,
                .rlat1 = NAN,
                .rlat2 = NAN }
        }
    },
    {
        .label = "Transverse Mercator",
        .proj = PL_PROJECT_TRANSVERSE_MERCATOR,
        .proj4_name = "tmerc",
        .tests = {
            { .name = "Spherical, centered",
                .ell = PL_SPHEROID_SPHERE,
                .lon0 = 0.0,
                .lat0 = 0.0,
                .rlat1 = NAN,
                .rlat2 = NAN },
            { .name = "Spherical, off-center",
                .ell = PL_SPHEROID_SPHERE,
                .lon0 = 10.0,
                .lat0 = 0.0,
                .rlat1 = NAN,
                .rlat2 = NAN },
        }
    },
    {
        .label = "Transverse Mercator (Extended)",
        .proj = PL_PROJECT_TRANSVERSE_MERCATOR,
        .proj4_name = "etmerc",
        .tests = {
            { .name = "Ellipsoidal, centered",
                .ell = PL_SPHEROID_WGS_84,
                .lon0 = 0.0,
                .lat0 = 0.0,
                .rlat1 = NAN,
                .rlat2 = NAN },
            { .name = "Ellipsoidal, off-center",
                .ell = PL_SPHEROID_WGS_84,
                .lon0 = 10.0,
                .lat0 = 0.0,
                .rlat1 = NAN,
                .rlat2 = NAN }
        }
    },
    {
        .label = "Winkel Tripel",
        .proj = PL_PROJECT_WINKEL_TRIPEL,
        .proj4_name = "wintri",
        .tests = {
            { .name = "Centered",
                .ell = PL_SPHEROID_SPHERE,
                .lon0 = 0.0,
                .lat0 = NAN,
                .rlat1 = NAN,
                .rlat2 = NAN },
            { .name = "Off-center",
                .ell = PL_SPHEROID_SPHERE,
                .lon0 = 10.0,
                .lat0 = NAN,
                .rlat1 = NAN,
                .rlat2 = NAN }
        }
    }
};

int compile_module(PLContext *ctx, unsigned int module, char *name);
int compare_points(const float *points1, const float *points2, const float *ref_points,
        int count, char *name, double tolerance, double speedup);
uint64_t test_american_polyconic(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points);
uint64_t test_lambert_azimuthal_equal_area(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points);
uint64_t test_lambert_conformal_conic(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points);
uint64_t test_mercator(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points);
uint64_t test_oblique_stereographic(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points);
uint64_t test_robinson(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points);
uint64_t test_transverse_mercator(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points);
uint64_t test_winkel_tripel(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points);
uint64_t run_test_group(test_group_t *group, PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points);

int compare_points(const float *points1, const float *points2, const float *ref_points, int count,
        char *name, double tolerance, double speedup) {
    int i;
    int failures = 0;
    printf("-- %s... ", name);
    double max_delta_x = 0.0;
    double max_delta_y = 0.0;
    int max_delta_x_i = 0, max_delta_y_i = 0;
    for (i=0; i<count; i++) {
        double delta_x = fabs(points1[2*i] - points2[2*i]);
        double delta_y = fabs(points1[2*i+1] - points2[2*i+1]);
        if (delta_x > tolerance || delta_y > tolerance) {
            failures++;
            if (delta_x > max_delta_x) {
                max_delta_x = delta_x;
                max_delta_x_i = i;
            }
            if (delta_y > max_delta_y) {
                max_delta_y = delta_y;
                max_delta_y_i = i;
            }
        }
    }
    if (failures) {
        printf("%d failures\n", failures);
        printf("**** Max longitudinal error: (%f, %f) => (%f, %f)\n"
               "                             (%f, %f) => (%f, %f) [%.4lf @ %d]\n",
               ref_points[2*max_delta_x_i], ref_points[2*max_delta_x_i+1], points1[2*max_delta_x_i], points1[2*max_delta_x_i+1],
               ref_points[2*max_delta_x_i], ref_points[2*max_delta_x_i+1], points2[2*max_delta_x_i], points2[2*max_delta_x_i+1],
               max_delta_x, max_delta_x_i);
        printf("**** Max latitudinal error: (%f, %f) => (%f, %f)\n"
               "                            (%f, %f) => (%f, %f) [%.4lf @ %d]\n",
               ref_points[2*max_delta_y_i], ref_points[2*max_delta_y_i+1], points1[2*max_delta_y_i], points1[2*max_delta_y_i+1],
               ref_points[2*max_delta_y_i], ref_points[2*max_delta_y_i+1], points2[2*max_delta_y_i], points2[2*max_delta_y_i+1],
               max_delta_y, max_delta_y_i);
    } else {
        if (isnan(speedup)) {
            printf("ok\n");
        } else {
            printf("ok (%.1lfX faster)\n", speedup);
        }
    }

    return failures;
}

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
    cl_device_type devtype = CL_DEVICE_TYPE_GPU;
    if(argc > 1) {
      if (!strcmp(argv[1], "-CPU"))
	devtype = CL_DEVICE_TYPE_CPU;
    }
    PLContext *ctx = pl_context_init(devtype, &error);
    if (ctx == NULL) {
        printf("Failed to initialize context: %d\n", error);
        return 1;
    }

    printf("Compiling individual modules...\n");
    int compile_failures = 0;
    compile_failures += compile_module(ctx, PL_MODULE_DATUM, "datum");
    compile_failures += compile_module(ctx, PL_MODULE_GEODESIC, "geodesic");
    compile_failures += compile_module(ctx, PL_MODULE_WARP, "warp");
    compile_failures += compile_module(ctx, PL_MODULE_ALBERS_EQUAL_AREA, "Albers Equal Area");
    compile_failures += compile_module(ctx, PL_MODULE_AMERICAN_POLYCONIC, "American Polyconic");
    compile_failures += compile_module(ctx, PL_MODULE_LAMBERT_AZIMUTHAL_EQUAL_AREA, "Lambert Azimuthal Equal Area");
    compile_failures += compile_module(ctx, PL_MODULE_LAMBERT_CONFORMAL_CONIC, "Lambert Conformal Conic");
    compile_failures += compile_module(ctx, PL_MODULE_MERCATOR, "Mercator");
    compile_failures += compile_module(ctx, PL_MODULE_OBLIQUE_STEREOGRAPHIC, "Oblique Stereographic");
    compile_failures += compile_module(ctx, PL_MODULE_ROBINSON, "Robinson");
    compile_failures += compile_module(ctx, PL_MODULE_TRANSVERSE_MERCATOR, "Transverse Mercator");
    compile_failures += compile_module(ctx, PL_MODULE_WINKEL_TRIPEL, "Winkel Tripel");
    compile_failures += compile_module(ctx, PL_MODULE_NEAREST_NEIGHBOR, "nearest neighbor");
    compile_failures += compile_module(ctx, PL_MODULE_BILINEAR, "bilinear");
    compile_failures += compile_module(ctx, PL_MODULE_BICUBIC, "bicubic");
    compile_failures += compile_module(ctx, PL_MODULE_QUASI_BICUBIC, "quasi-bicubic");

    printf("Compilation failures: %d\n", compile_failures);

    code = pl_compile_code(ctx, "kernel", 0xFFFF, &error);
    error = pl_load_code(ctx, code);
    pl_release_code(code);
    if (error != CL_SUCCESS) {
        printf("Failed to load code: %d\n", error);
        return 1;
    }

    float *orig_points = malloc(2*TEST_POINTS*sizeof(float));
    int i;
    int offset = 0;
    PLProjectionBuffer *orig_buf = NULL;

    for (i=0; i<TEST_POINTS/2; i++) { /* grid */
        orig_points[offset++] = MIN_LON + (MAX_LON - MIN_LON) * (i%500) / (500-1);
        orig_points[offset++] = MIN_LAT + (MAX_LAT - MIN_LAT) * (i/500) / (TEST_POINTS/2/500-1);
    }
    for (i=0; i<TEST_POINTS/2; i++) { /* circle */
        orig_points[offset++] = MIN_LAT * cos(2 * M_PI * i/(TEST_POINTS/2-1));
        orig_points[offset++] = MIN_LAT * sin(2 * M_PI * i/(TEST_POINTS/2-1));
    }

    orig_buf = pl_load_projection_data(ctx, orig_points, TEST_POINTS, 1, &error);
    if (orig_buf == NULL) {
        printf("Failed to load points: %d\n", error);
        return 1;
    }

    uint64_t test_failures = 0;

    int g;
    for (g=0; g<sizeof(_test_groups)/sizeof(_test_groups[0]); g++) {
        test_group_t *group = &_test_groups[g];
        printf("\nTesting consistency of %s\n", group->label);

        test_failures += run_test_group(group, ctx, orig_buf, orig_points);
    }

    printf("\nTotal consistency failures: %" PRId64 "\n", test_failures & 0xFFFFFFFF);
    printf("Total Proj.4 mismatches: %" PRId64 "\n", test_failures >> 32);

    pl_unload_projection_data(orig_buf);

    pl_unload_code(ctx);
    pl_context_free(ctx);

    free(orig_points);

    return test_failures > 0;
}
int compare_proj4_inv(const float *proj_points, const float *orig_points,
        const char *desc1, const char *desc2, double projcl_secs) {
    int failures = 0;
#ifdef HAVE_PROJ4
    double points_proj4[2*TEST_POINTS];
    float orig_points_proj4[2*TEST_POINTS];
    projPJ pj_out, pj_in;
    int i;
    int error;
    struct timeval start_time, end_time;
    double proj4_secs;

    for (i=0; i<TEST_POINTS; i++) {
      points_proj4[2*i] = proj_points[2*i];
      points_proj4[2*i+1] = proj_points[2*i+1];
    }

    if ((pj_in = pj_init_plus(desc1)) == NULL) {
      printf("Failed to init Proj.4 input structure: %s\n", desc1);
      printf("Error code %d: %s\n", pj_errno, pj_strerrno(pj_errno));
      exit(1);
    }

    if ((pj_out = pj_init_plus(desc2)) == NULL) {
      printf("Failed to init Proj.4 output structure: %s\n", desc2);
      printf("Error code %d: %s\n", pj_errno, pj_strerrno(pj_errno));
      exit(1);
    }

    gettimeofday(&start_time, NULL);
    error = pj_transform(pj_in, pj_out, TEST_POINTS, 2, points_proj4, points_proj4 + 1, NULL);
    gettimeofday(&end_time, NULL);

    proj4_secs = (end_time.tv_sec + end_time.tv_usec * 1e-6) 
        - (start_time.tv_sec + start_time.tv_usec * 1e-6);

    if (error != 0) {
      printf("Error projecting: %s\n", pj_strerrno(error));
      exit(1);
    }

    pj_free(pj_in);
    pj_free(pj_out);

    for (i=0; i<TEST_POINTS; i++) {
      orig_points_proj4[2*i] = points_proj4[2*i] * RAD_TO_DEG;
      orig_points_proj4[2*i+1] = points_proj4[2*i+1] * RAD_TO_DEG;
    }

    failures = compare_points(orig_points, orig_points_proj4, proj_points, TEST_POINTS, 
            "...inverse same as Proj.4", DEGREES_TOL, proj4_secs / projcl_secs);
#endif
    return failures;
}

int compare_proj4_fwd(const float *orig_points, const float *proj_points,
        const char *desc1, const char *desc2, double projcl_secs) {
    int failures = 0;
#ifdef HAVE_PROJ4
    double points_proj4[2*TEST_POINTS];
    float proj_points_proj4[2*TEST_POINTS];
    projPJ pj_out, pj_in;
    int i;
    int error;
    struct timeval start_time, end_time;
    double proj4_secs;

    for (i=0; i<TEST_POINTS; i++) {
      points_proj4[2*i] = orig_points[2*i] * DEG_TO_RAD;
      points_proj4[2*i+1] = orig_points[2*i+1] * DEG_TO_RAD;
    }

    if ((pj_in = pj_init_plus(desc1)) == NULL) {
      printf("Failed to init Proj.4 input structure: %s\n", desc1);
      printf("Error code %d: %s\n", pj_errno, pj_strerrno(pj_errno));
      exit(1);
    }

    if ((pj_out = pj_init_plus(desc2)) == NULL) {
      printf("Failed to init Proj.4 output structure: %s\n", desc2);
      printf("Error code %d: %s\n", pj_errno, pj_strerrno(pj_errno));
      exit(1);
    }

    gettimeofday(&start_time, NULL);
    error = pj_transform(pj_in, pj_out, TEST_POINTS, 2, points_proj4, points_proj4 + 1, NULL);
    gettimeofday(&end_time, NULL);

    proj4_secs = (end_time.tv_sec + end_time.tv_usec * 1e-6) 
        - (start_time.tv_sec + start_time.tv_usec * 1e-6);

    if (error != 0) {
      printf("Error projecting: %s\n", pj_strerrno(error));
      exit(1);
    }

    pj_free(pj_in);
    pj_free(pj_out);

    for (i=0; i<TEST_POINTS; i++) {
      proj_points_proj4[2*i] = points_proj4[2*i];
      proj_points_proj4[2*i+1] = points_proj4[2*i+1];
    }

    failures = compare_points(proj_points, proj_points_proj4, orig_points, TEST_POINTS, 
            "...forward same as Proj.4", METERS_TOL, proj4_secs / projcl_secs);
#endif
    return failures;
}

int sprintf_proj4(char *buf, const char *name, test_params_t *params) {
    char *p = buf;
    p += sprintf(p, "+proj=%s", name);
    p += sprintf(p, " +ellps=%s", params->ell == PL_SPHEROID_SPHERE ? "sphere" : "WGS84");
    if (strcmp(name, "latlong") != 0) {
        if (!isnan(params->lat0) && params->lat0 != 0.0)
            p += sprintf(p, " +lat_0=%.1lf%c", fabs(params->lat0), params->lat0 > 0.0 ? 'n' : 's');
        if (!isnan(params->lon0) && params->lon0 != 0.0)
            p += sprintf(p, " +lon_0=%.1lf%c", fabs(params->lon0), params->lon0 > 0.0 ? 'e' : 'w');
        if (!isnan(params->rlat1))
            p += sprintf(p, " +lat_1=%.1lf%c", fabs(params->rlat1), params->rlat1 > 0.0 ? 'n' : 's');
        if (!isnan(params->rlat2))
            p += sprintf(p, " +lat_2=%.1lf%c", fabs(params->rlat2), params->rlat2 > 0.0 ? 'n' : 's');
    }

    return 0;
}

uint64_t run_test_group(test_group_t *group, PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points) {
    int error = CL_SUCCESS;
    uint64_t consistency_failures = 0, proj4_mismatches = 0;
    PLProjectionBuffer *proj_buf = NULL;
    float proj_points[2*TEST_POINTS];
    float orig_points2[2*TEST_POINTS];
    int i;
    char orig_string[80];
    char proj_string[80];
    double fwd_secs, inv_secs;
    PLProjectionParams *params = pl_params_init();

    for (i=0; i<10 && group->tests[i].name[0]; i++) {
        test_params_t *test = &group->tests[i];

        pl_params_set_false_northing(params, 0.0);
        pl_params_set_false_easting(params, 0.0);
        pl_params_set_scale(params, 1.0);
        pl_params_set_spheroid(params, test->ell);
        pl_params_set_latitude_of_origin(params, test->lat0);
        pl_params_set_longitude_of_origin(params, test->lon0);
        pl_params_set_standard_parallels(params, test->rlat1, test->rlat2);

        error = pl_project_points_forward(ctx, group->proj, params, orig_buf, proj_points);
        if (error == CL_SUCCESS) {
            fwd_secs = ctx->last_time;
        } else {
            printf("** Error Projecting: %d **\n", error);
            fwd_secs = NAN;
        }

        proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);

        error = pl_project_points_reverse(ctx, group->proj, params, proj_buf, orig_points2);
        if (error == CL_SUCCESS) {
            inv_secs = ctx->last_time;
        } else {
            printf("** Error Projecting: %d **\n", error);
            inv_secs = NAN;
        }

        pl_unload_projection_data(proj_buf);

        consistency_failures += compare_points(orig_points, orig_points2, proj_points, TEST_POINTS, 
                test->name, DEGREES_TOL, NAN);

        sprintf_proj4(orig_string, "latlong", test);
        sprintf_proj4(proj_string, group->proj4_name, test);

        proj4_mismatches += compare_proj4_fwd(orig_points, proj_points,
                orig_string, proj_string, fwd_secs);
        proj4_mismatches += compare_proj4_inv(proj_points, orig_points2,
                proj_string, orig_string, inv_secs);
    }

    return consistency_failures + (proj4_mismatches << 32);
}

/*
uint64_t test_american_polyconic(PLContext *ctx, 
        PLProjectionBuffer *orig_buf, float *orig_points) {
    int error = CL_SUCCESS;
    uint64_t consistency_failures = 0, proj4_mismatches = 0;
    PLProjectionBuffer *proj_buf = NULL;
    float proj_points[2*TEST_POINTS];
    float orig_points2[2*TEST_POINTS];

    int i;
    char orig_string[80];
    char proj_string[80];
    double fwd_secs, inv_secs;

    for (i=0; i<sizeof(american_polyconic_tests)/sizeof(test_params_t); i++) {
        test_params_t test = american_polyconic_tests[i];

        error = pl_project_american_polyconic(ctx, orig_buf, proj_points, 
              test.ell, 1.0, 0.0, 0.0, test.lon0, test.lat0);
        fwd_secs = ctx->last_time;

        proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);

        error = pl_unproject_american_polyconic(ctx, proj_buf, orig_points2, 
                test.ell, 1.0, 0.0, 0.0, test.lon0, test.lat0);
        inv_secs = ctx->last_time;

        pl_unload_projection_data(proj_buf);

        consistency_failures += compare_points(orig_points, orig_points2, proj_points,
                TEST_POINTS, test.name, DEGREES_TOL, NAN);

        sprintf_proj4(orig_string, "latlong", test);
        sprintf_proj4(proj_string, "poly", test);

        proj4_mismatches += compare_proj4_fwd(orig_points, proj_points,
                orig_string, proj_string, fwd_secs);
        proj4_mismatches += compare_proj4_inv(proj_points, orig_points2,
                proj_string, orig_string, inv_secs);
    }

    return consistency_failures + (proj4_mismatches << 32);
}

uint64_t test_lambert_azimuthal_equal_area(PLContext *ctx, 
        PLProjectionBuffer *orig_buf, float *orig_points) {
    int error = CL_SUCCESS;
    uint64_t consistency_failures = 0, proj4_mismatches = 0;
    PLProjectionBuffer *proj_buf = NULL;
    float proj_points[2*TEST_POINTS];
    float orig_points2[2*TEST_POINTS];

    int i;
    char orig_string[80];
    char proj_string[80];
    double fwd_secs, inv_secs;

    for (i=0; i<sizeof(lambert_azimuthal_equal_area_tests)/sizeof(test_params_t); i++) {
        test_params_t test = lambert_azimuthal_equal_area_tests[i];

        error = pl_project_lambert_azimuthal_equal_area(ctx, orig_buf, proj_points, 
              test.ell, 1.0, 0.0, 0.0, test.lon0, test.lat0);
        fwd_secs = ctx->last_time;

        proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);

        error = pl_unproject_lambert_azimuthal_equal_area(ctx, proj_buf, orig_points2, 
                test.ell, 1.0, 0.0, 0.0, test.lon0, test.lat0);
        inv_secs = ctx->last_time;

        pl_unload_projection_data(proj_buf);

        consistency_failures += compare_points(orig_points, orig_points2, proj_points,
                TEST_POINTS, test.name, DEGREES_TOL, NAN);

        sprintf_proj4(orig_string, "latlong", test);
        sprintf_proj4(proj_string, "laea", test);

        proj4_mismatches += compare_proj4_fwd(orig_points, proj_points,
                orig_string, proj_string, fwd_secs);
        proj4_mismatches += compare_proj4_inv(proj_points, orig_points2,
                proj_string, orig_string, inv_secs);
    }

    return consistency_failures + (proj4_mismatches << 32);
}

uint64_t test_lambert_conformal_conic(PLContext *ctx, 
        PLProjectionBuffer *orig_buf, float *orig_points) {
    int error = CL_SUCCESS;
    uint64_t consistency_failures = 0, proj4_mismatches = 0;
    PLProjectionBuffer *proj_buf = NULL;
    float proj_points[2*TEST_POINTS];
    float orig_points2[2*TEST_POINTS];
    int i;
    char orig_string[80];
    char proj_string[80];
    double fwd_secs, inv_secs;

    for (i=0; i<sizeof(lambert_conformal_conic_tests)/sizeof(test_params_t); i++) {
        test_params_t test = lambert_conformal_conic_tests[i];

        error = pl_project_lambert_conformal_conic(ctx, orig_buf, proj_points, 
              test.ell, 1.0, 0.0, 0.0, test.lon0, test.lat0, test.rlat1, test.rlat2);
        fwd_secs = ctx->last_time;

        proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);

        error = pl_unproject_lambert_conformal_conic(ctx, proj_buf, orig_points2, 
                test.ell, 1.0, 0.0, 0.0, test.lon0, test.lat0, test.rlat1, test.rlat2);
        inv_secs = ctx->last_time;

        pl_unload_projection_data(proj_buf);

        consistency_failures += compare_points(orig_points, orig_points2, proj_points,
                TEST_POINTS, test.name, DEGREES_TOL, NAN);

        sprintf_proj4(orig_string, "latlong", test);
        sprintf_proj4(proj_string, "lcc", test);

        consistency_failures += compare_proj4_fwd(orig_points, proj_points,
                orig_string, proj_string, fwd_secs);
        consistency_failures += compare_proj4_inv(proj_points, orig_points2,
                proj_string, orig_string, inv_secs);
    }

    return consistency_failures + (proj4_mismatches << 32);
}

uint64_t test_mercator(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points) {
    int error = CL_SUCCESS;
    uint64_t consistency_failures = 0, proj4_mismatches = 0;
    PLProjectionBuffer *proj_buf = NULL;
    float proj_points[2*TEST_POINTS];
    float orig_points2[2*TEST_POINTS];
    int i;
    char orig_string[80];
    char proj_string[80];
    double fwd_secs, inv_secs;

    for (i=0; i<sizeof(mercator_tests)/sizeof(test_params_t); i++) {
        test_params_t test = mercator_tests[i];

        error = pl_project_mercator(ctx, orig_buf, proj_points, test.ell, 1.0, 0.0, 0.0);
        fwd_secs = ctx->last_time;

        proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);

        error = pl_unproject_mercator(ctx, proj_buf, orig_points2, test.ell, 1.0, 0.0, 0.0);
        inv_secs = ctx->last_time;

        pl_unload_projection_data(proj_buf);

        consistency_failures += compare_points(orig_points, orig_points2, proj_points,
                TEST_POINTS, test.name, DEGREES_TOL, NAN);

        sprintf_proj4(orig_string, "latlong", test);
        sprintf_proj4(proj_string, "merc", test);

        proj4_mismatches += compare_proj4_fwd(orig_points, proj_points,
                orig_string, proj_string, fwd_secs);
        proj4_mismatches += compare_proj4_inv(proj_points, orig_points2,
                proj_string, orig_string, inv_secs);
    }

    return consistency_failures + (proj4_mismatches << 32);
}

uint64_t test_oblique_stereographic(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points) {
    int error = CL_SUCCESS;
    uint64_t consistency_failures = 0, proj4_mismatches = 0;
    PLProjectionBuffer *proj_buf = NULL;
    float proj_points[2*TEST_POINTS];
    float orig_points2[2*TEST_POINTS];
    int i;
    char orig_string[80];
    char proj_string[80];
    double fwd_secs, inv_secs;

    for (i=0; i<sizeof(oblique_stereographic_tests)/sizeof(test_params_t); i++) {
        test_params_t test = oblique_stereographic_tests[i];

        error = pl_project_oblique_stereographic(ctx, orig_buf, proj_points, 
              test.ell, 1.0, 0.0, 0.0, test.lon0, test.lat0);
        fwd_secs = ctx->last_time;

        proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);

        error = pl_unproject_oblique_stereographic(ctx, proj_buf, orig_points2, 
                test.ell, 1.0, 0.0, 0.0, test.lon0, test.lat0);
        inv_secs = ctx->last_time;

        pl_unload_projection_data(proj_buf);

        consistency_failures += compare_points(orig_points, orig_points2, proj_points,
                TEST_POINTS, test.name, DEGREES_TOL, NAN);

        sprintf_proj4(orig_string, "latlong", test);
        sprintf_proj4(proj_string, "sterea", test);

        proj4_mismatches += compare_proj4_fwd(orig_points, proj_points,
                orig_string, proj_string, fwd_secs);
        proj4_mismatches += compare_proj4_inv(proj_points, orig_points2,
                proj_string, orig_string, inv_secs);
    }

    return consistency_failures + (proj4_mismatches << 32);
}

uint64_t test_robinson(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points) {
    int error = CL_SUCCESS;
    uint64_t consistency_failures = 0, proj4_mismatches = 0;
    PLProjectionBuffer *proj_buf = NULL;
    float proj_points[2*TEST_POINTS];
    float orig_points2[2*TEST_POINTS];
    int i;
    char orig_string[80];
    char proj_string[80];
    double fwd_secs, inv_secs;

    for (i=0; i<sizeof(robinson_tests)/sizeof(test_params_t); i++) {
        test_params_t test = robinson_tests[i];

        error = pl_project_robinson(ctx, orig_buf, proj_points, 1.0, 0.0, 0.0);
        fwd_secs = ctx->last_time;

        proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);

        error = pl_unproject_robinson(ctx, proj_buf, orig_points2, 1.0, 0.0, 0.0);
        inv_secs = ctx->last_time;

        pl_unload_projection_data(proj_buf);

        consistency_failures += compare_points(orig_points, orig_points2, proj_points,
                TEST_POINTS, test.name, DEGREES_TOL, NAN);

        sprintf_proj4(orig_string, "latlong", test);
        sprintf_proj4(proj_string, "robin", test);
        proj4_mismatches += compare_proj4_fwd(orig_points, proj_points,
                orig_string, proj_string, fwd_secs);
        proj4_mismatches += compare_proj4_inv(proj_points, orig_points2,
                proj_string, orig_string, inv_secs);
    }

    return consistency_failures + (proj4_mismatches << 32);
}

uint64_t test_transverse_mercator(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points) {
    int error = CL_SUCCESS;
    uint64_t consistency_failures = 0, proj4_mismatches = 0;
    PLProjectionBuffer *proj_buf = NULL;
    float proj_points[2*TEST_POINTS];
    float orig_points2[2*TEST_POINTS];
    int i;
    char orig_string[80];
    char proj_string[80];
    double fwd_secs, inv_secs;

    for (i=0; i<sizeof(transverse_mercator_tests)/sizeof(test_params_t); i++) {
        test_params_t test = transverse_mercator_tests[i];

        error = pl_project_transverse_mercator(ctx, orig_buf, proj_points, 
              test.ell, 1.0, 0.0, 0.0, test.lon0);
        fwd_secs = ctx->last_time;

        proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);

        error = pl_unproject_transverse_mercator(ctx, proj_buf, orig_points2, 
                test.ell, 1.0, 0.0, 0.0, test.lon0);
        inv_secs = ctx->last_time;

        pl_unload_projection_data(proj_buf);

        consistency_failures += compare_points(orig_points, orig_points2, proj_points,
                TEST_POINTS, test.name, DEGREES_TOL, NAN);

        sprintf_proj4(orig_string, "latlong", test);
        sprintf_proj4(proj_string, test.ell == PL_SPHEROID_SPHERE ? "tmerc" : "etmerc", test);

        proj4_mismatches += compare_proj4_fwd(orig_points, proj_points,
                orig_string, proj_string, fwd_secs);
        proj4_mismatches += compare_proj4_inv(proj_points, orig_points2,
                proj_string, orig_string, inv_secs);
    }

    return consistency_failures + (proj4_mismatches << 32);
}

uint64_t test_winkel_tripel(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points) {
    int error = CL_SUCCESS;
    uint64_t consistency_failures = 0, proj4_mismatches = 0;
    PLProjectionBuffer *proj_buf = NULL;
    float proj_points[2*TEST_POINTS];
    float orig_points2[2*TEST_POINTS];
    double fwd_secs, inv_secs;

    int i;
    char orig_string[80];
    char proj_string[80];

    for (i=0; i<sizeof(winkel_tripel_tests)/sizeof(test_params_t); i++) {
        test_params_t test = winkel_tripel_tests[i];

        error = pl_project_winkel_tripel(ctx, orig_buf, proj_points, 
              1.0, 0.0, 0.0, test.lon0, acos(M_2_PI) * RAD_TO_DEG);
        fwd_secs = ctx->last_time;

        proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);

        error = pl_unproject_winkel_tripel(ctx, proj_buf, orig_points2, 
                1.0, 0.0, 0.0, test.lon0, acos(M_2_PI) * RAD_TO_DEG);
        inv_secs = ctx->last_time;

        pl_unload_projection_data(proj_buf);

        consistency_failures += compare_points(orig_points, orig_points2, proj_points,
                TEST_POINTS, test.name, DEGREES_TOL, NAN);

        sprintf_proj4(orig_string, "latlong", test);
        sprintf_proj4(proj_string, "wintri", test);

        proj4_mismatches += compare_proj4_fwd(orig_points, proj_points,
                orig_string, proj_string, fwd_secs);
        proj4_mismatches += compare_proj4_inv(proj_points, orig_points2,
                proj_string, orig_string, inv_secs);
    }

    return consistency_failures + (proj4_mismatches << 32);
}
*/
