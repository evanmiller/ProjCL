
#include <stdio.h>
#include <math.h>
#include <string.h>

#include <OpenCL/OpenCL.h>
#include <sys/time.h>
#include "../src/projcl.h"

#define TEST_POINTS 1000
#define TOL 1.e-4

int compile_module(PLContext *ctx, unsigned int module, char *name);
int compare_points(float *points1, float *points2, int count, char *name);
int test_albers_equal_area(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points);
int test_american_polyconic(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points);
int test_lambert_azimuthal_equal_area(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points);
int test_lambert_conformal_conic(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points);
int test_mercator(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points);
int test_robinson(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points);
int test_transverse_mercator(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points);
int test_winkel_tripel(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points);

int test_albers_equal_area(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points) {
    int error = CL_SUCCESS;
    int consistency_failures = 0;
    PLProjectionBuffer *proj_buf = NULL;
    float proj_points[2*TEST_POINTS];
    float orig_points2[2*TEST_POINTS];

    error = pl_project_albers_equal_area(ctx, orig_buf, proj_points, PL_SPHEROID_SPHERE, 
            1.0, 0.0, 0.0, 0.0, 0.0, 30.0, 60.0);
    proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
    error = pl_unproject_albers_equal_area(ctx, proj_buf, orig_points2, PL_SPHEROID_SPHERE, 
            1.0, 0.0, 0.0, 0.0, 0.0, 30.0, 60.0);
    pl_unload_projection_data(proj_buf);

    consistency_failures += compare_points(orig_points, orig_points2, TEST_POINTS, 
            "Spherical, curved, centered");

    error = pl_project_albers_equal_area(ctx, orig_buf, proj_points, PL_SPHEROID_SPHERE, 
            1.0, 0.0, 0.0, 10.0, 10.0, 30.0, 60.0);
    proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
    error = pl_unproject_albers_equal_area(ctx, proj_buf, orig_points2, PL_SPHEROID_SPHERE, 
            1.0, 0.0, 0.0, 10.0, 10.0, 30.0, 60.0);
    pl_unload_projection_data(proj_buf);

    consistency_failures += compare_points(orig_points, orig_points2, TEST_POINTS, 
            "Spherical, curved, off-center");

    error = pl_project_albers_equal_area(ctx, orig_buf, proj_points, PL_SPHEROID_SPHERE, 
            1.0, 0.0, 0.0, 0.0, 0.0, -30.0, 30.0);
    proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
    error = pl_unproject_albers_equal_area(ctx, proj_buf, orig_points2, PL_SPHEROID_SPHERE, 
            1.0, 0.0, 0.0, 0.0, 0.0, -30.0, 30.0);
    pl_unload_projection_data(proj_buf);

    consistency_failures += compare_points(orig_points, orig_points2, TEST_POINTS, 
            "Spherical, flat, centered");

    error = pl_project_albers_equal_area(ctx, orig_buf, proj_points, PL_SPHEROID_WGS_84, 
            1.0, 0.0, 0.0, 0.0, 0.0, 30.0, 60.0);
    proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
    error = pl_unproject_albers_equal_area(ctx, proj_buf, orig_points2, PL_SPHEROID_WGS_84, 
            1.0, 0.0, 0.0, 0.0, 0.0, 30.0, 60.0);
    pl_unload_projection_data(proj_buf);

    consistency_failures += compare_points(orig_points, orig_points2, TEST_POINTS, "Ellipsoidal, curved, centered");

    error = pl_project_albers_equal_area(ctx, orig_buf, proj_points, PL_SPHEROID_WGS_84, 1.0, 0.0, 0.0, 10.0, 10.0, 30.0, 60.0);
    proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
    error = pl_unproject_albers_equal_area(ctx, proj_buf, orig_points2, PL_SPHEROID_WGS_84, 1.0, 0.0, 0.0, 10.0, 10.0, 30.0, 60.0);
    pl_unload_projection_data(proj_buf);

    consistency_failures += compare_points(orig_points, orig_points2, TEST_POINTS, "Ellispoidal, curved, off-center");

    error = pl_project_albers_equal_area(ctx, orig_buf, proj_points, PL_SPHEROID_WGS_84, 1.0, 0.0, 0.0, 0.0, 0.0, -30.0, 30.0);
    proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
    error = pl_unproject_albers_equal_area(ctx, proj_buf, orig_points2, PL_SPHEROID_WGS_84, 1.0, 0.0, 0.0, 0.0, 0.0, -30.0, 30.0);
    pl_unload_projection_data(proj_buf);

    consistency_failures += compare_points(orig_points, orig_points2, TEST_POINTS, "Ellipsoidal, flat, centered");

    return consistency_failures;
}

int test_american_polyconic(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points) {
    int error = CL_SUCCESS;
    int consistency_failures = 0;
    PLProjectionBuffer *proj_buf = NULL;
    float proj_points[2*TEST_POINTS];
    float orig_points2[2*TEST_POINTS];

    error = pl_project_american_polyconic(ctx, orig_buf, proj_points, PL_SPHEROID_SPHERE, 1.0, 0.0, 0.0, 0.0, 0.0);
    proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
    error = pl_unproject_american_polyconic(ctx, proj_buf, orig_points2, PL_SPHEROID_SPHERE, 1.0, 0.0, 0.0, 0.0, 0.0);
    pl_unload_projection_data(proj_buf);

    consistency_failures += compare_points(orig_points, orig_points2, TEST_POINTS, "Spherical, centered");

    error = pl_project_american_polyconic(ctx, orig_buf, proj_points, PL_SPHEROID_SPHERE, 1.0, 0.0, 0.0, 10.0, 10.0);
    proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
    error = pl_unproject_american_polyconic(ctx, proj_buf, orig_points2, PL_SPHEROID_SPHERE, 1.0, 0.0, 0.0, 10.0, 10.0);
    pl_unload_projection_data(proj_buf);

    consistency_failures += compare_points(orig_points, orig_points2, TEST_POINTS, "Spherical, off-center");

    error = pl_project_american_polyconic(ctx, orig_buf, proj_points, PL_SPHEROID_WGS_84, 1.0, 0.0, 0.0, 0.0, 0.0);
    proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
    error = pl_unproject_american_polyconic(ctx, proj_buf, orig_points2, PL_SPHEROID_WGS_84, 1.0, 0.0, 0.0, 0.0, 0.0);
    pl_unload_projection_data(proj_buf);

    consistency_failures += compare_points(orig_points, orig_points2, TEST_POINTS, "Ellipsoidal, centered");

    error = pl_project_american_polyconic(ctx, orig_buf, proj_points, PL_SPHEROID_WGS_84, 1.0, 0.0, 0.0, 10.0, 10.0);
    proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
    error = pl_unproject_american_polyconic(ctx, proj_buf, orig_points2, PL_SPHEROID_WGS_84, 1.0, 0.0, 0.0, 10.0, 10.0);
    pl_unload_projection_data(proj_buf);

    consistency_failures += compare_points(orig_points, orig_points2, TEST_POINTS, "Ellipsoidal, off-center");

    return consistency_failures;
}

int test_lambert_azimuthal_equal_area(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points) {
    int error = CL_SUCCESS;
    int consistency_failures = 0;
    PLProjectionBuffer *proj_buf = NULL;
    float proj_points[2*TEST_POINTS];
    float orig_points2[2*TEST_POINTS];

    error = pl_project_lambert_azimuthal_equal_area(ctx, orig_buf, proj_points, PL_SPHEROID_SPHERE, 1.0, 0.0, 0.0, 0.0, 0.0);
    proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
    error = pl_unproject_lambert_azimuthal_equal_area(ctx, proj_buf, orig_points2, PL_SPHEROID_SPHERE, 1.0, 0.0, 0.0, 0.0, 0.0);
    pl_unload_projection_data(proj_buf);

    consistency_failures += compare_points(orig_points, orig_points2, TEST_POINTS, "Spherical, centered");

    error = pl_project_lambert_azimuthal_equal_area(ctx, orig_buf, proj_points, PL_SPHEROID_SPHERE, 1.0, 0.0, 0.0, 10.0, 10.0);
    proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
    error = pl_unproject_lambert_azimuthal_equal_area(ctx, proj_buf, orig_points2, PL_SPHEROID_SPHERE, 1.0, 0.0, 0.0, 10.0, 10.0);
    pl_unload_projection_data(proj_buf);

    consistency_failures += compare_points(orig_points, orig_points2, TEST_POINTS, "Spherical, off-center");

    error = pl_project_lambert_azimuthal_equal_area(ctx, orig_buf, proj_points, PL_SPHEROID_WGS_84, 1.0, 0.0, 0.0, 0.0, 0.0);
    proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
    error = pl_unproject_lambert_azimuthal_equal_area(ctx, proj_buf, orig_points2, PL_SPHEROID_WGS_84, 1.0, 0.0, 0.0, 0.0, 0.0);
    pl_unload_projection_data(proj_buf);

    consistency_failures += compare_points(orig_points, orig_points2, TEST_POINTS, "Ellipsoidal, centered");

    error = pl_project_lambert_azimuthal_equal_area(ctx, orig_buf, proj_points, PL_SPHEROID_WGS_84, 1.0, 0.0, 0.0, 10.0, 10.0);
    proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
    error = pl_unproject_lambert_azimuthal_equal_area(ctx, proj_buf, orig_points2, PL_SPHEROID_WGS_84, 1.0, 0.0, 0.0, 10.0, 10.0);
    pl_unload_projection_data(proj_buf);

    consistency_failures += compare_points(orig_points, orig_points2, TEST_POINTS, "Ellipsoidal, off-center");

    return consistency_failures;
}

int test_lambert_conformal_conic(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points) {
    int error = CL_SUCCESS;
    int consistency_failures = 0;
    PLProjectionBuffer *proj_buf = NULL;
    float proj_points[2*TEST_POINTS];
    float orig_points2[2*TEST_POINTS];

    error = pl_project_lambert_conformal_conic(ctx, orig_buf, proj_points, PL_SPHEROID_SPHERE, 1.0, 0.0, 0.0, 0.0, 0.0, 30.0, 60.0);
    proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
    error = pl_unproject_lambert_conformal_conic(ctx, proj_buf, orig_points2, PL_SPHEROID_SPHERE, 1.0, 0.0, 0.0, 0.0, 0.0, 30.0, 60.0);
    pl_unload_projection_data(proj_buf);

    consistency_failures += compare_points(orig_points, orig_points2, TEST_POINTS, "Spherical, curved, centered");

    error = pl_project_lambert_conformal_conic(ctx, orig_buf, proj_points, PL_SPHEROID_SPHERE, 1.0, 0.0, 0.0, 10.0, 10.0, 30.0, 60.0);
    proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
    error = pl_unproject_lambert_conformal_conic(ctx, proj_buf, orig_points2, PL_SPHEROID_SPHERE, 1.0, 0.0, 0.0, 10.0, 10.0, 30.0, 60.0);
    pl_unload_projection_data(proj_buf);

    consistency_failures += compare_points(orig_points, orig_points2, TEST_POINTS, "Spherical, curved, off-center");

    error = pl_project_lambert_conformal_conic(ctx, orig_buf, proj_points, PL_SPHEROID_SPHERE, 1.0, 0.0, 0.0, 0.0, 0.0, -30.0, 30.0);
    proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
    error = pl_unproject_lambert_conformal_conic(ctx, proj_buf, orig_points2, PL_SPHEROID_SPHERE, 1.0, 0.0, 0.0, 0.0, 0.0, -30.0, 30.0);
    pl_unload_projection_data(proj_buf);

    consistency_failures += compare_points(orig_points, orig_points2, TEST_POINTS, "Spherical, flat, centered");

    error = pl_project_lambert_conformal_conic(ctx, orig_buf, proj_points, PL_SPHEROID_WGS_84, 1.0, 0.0, 0.0, 0.0, 0.0, 30.0, 60.0);
    proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
    error = pl_unproject_lambert_conformal_conic(ctx, proj_buf, orig_points2, PL_SPHEROID_WGS_84, 1.0, 0.0, 0.0, 0.0, 0.0, 30.0, 60.0);
    pl_unload_projection_data(proj_buf);

    consistency_failures += compare_points(orig_points, orig_points2, TEST_POINTS, "Ellipsoidal, curved, centered");

    error = pl_project_lambert_conformal_conic(ctx, orig_buf, proj_points, PL_SPHEROID_WGS_84, 1.0, 0.0, 0.0, 10.0, 10.0, 30.0, 60.0);
    proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
    error = pl_unproject_lambert_conformal_conic(ctx, proj_buf, orig_points2, PL_SPHEROID_WGS_84, 1.0, 0.0, 0.0, 10.0, 10.0, 30.0, 60.0);
    pl_unload_projection_data(proj_buf);

    consistency_failures += compare_points(orig_points, orig_points2, TEST_POINTS, "Ellispoidal, curved, off-center");

    error = pl_project_lambert_conformal_conic(ctx, orig_buf, proj_points, PL_SPHEROID_WGS_84, 1.0, 0.0, 0.0, 0.0, 0.0, -30.0, 30.0);
    proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
    error = pl_unproject_lambert_conformal_conic(ctx, proj_buf, orig_points2, PL_SPHEROID_WGS_84, 1.0, 0.0, 0.0, 0.0, 0.0, -30.0, 30.0);
    pl_unload_projection_data(proj_buf);

    consistency_failures += compare_points(orig_points, orig_points2, TEST_POINTS, "Ellipsoidal, flat, centered");

    return consistency_failures;
}

int test_mercator(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points) {
    int error = CL_SUCCESS;
    int consistency_failures = 0;
    PLProjectionBuffer *proj_buf = NULL;
    float proj_points[2*TEST_POINTS];
    float orig_points2[2*TEST_POINTS];

    error = pl_project_mercator(ctx, orig_buf, proj_points, PL_SPHEROID_SPHERE, 1.0, 0.0, 0.0);
    proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
    error = pl_unproject_mercator(ctx, proj_buf, orig_points2, PL_SPHEROID_SPHERE, 1.0, 0.0, 0.0);
    pl_unload_projection_data(proj_buf);

    consistency_failures += compare_points(orig_points, orig_points2, TEST_POINTS, "Spherical");

    error = pl_project_mercator(ctx, orig_buf, proj_points, PL_SPHEROID_WGS_84, 1.0, 0.0, 0.0);
    proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
    error = pl_unproject_mercator(ctx, proj_buf, orig_points2, PL_SPHEROID_WGS_84, 1.0, 0.0, 0.0);
    pl_unload_projection_data(proj_buf);

    consistency_failures += compare_points(orig_points, orig_points2, TEST_POINTS, "Ellipsoidal");

    return consistency_failures;
}

int test_robinson(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points) {
    int error = CL_SUCCESS;
    int consistency_failures = 0;
    PLProjectionBuffer *proj_buf = NULL;
    float proj_points[2*TEST_POINTS];
    float orig_points2[2*TEST_POINTS];

    error = pl_project_robinson(ctx, orig_buf, proj_points);
    proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
    error = pl_unproject_robinson(ctx, proj_buf, orig_points2);
    pl_unload_projection_data(proj_buf);

    consistency_failures += compare_points(orig_points, orig_points2, TEST_POINTS, "(No parameters)");

    return consistency_failures;
}

int test_transverse_mercator(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points) {
    int error = CL_SUCCESS;
    int consistency_failures = 0;
    PLProjectionBuffer *proj_buf = NULL;
    float proj_points[2*TEST_POINTS];
    float orig_points2[2*TEST_POINTS];

    error = pl_project_transverse_mercator(ctx, orig_buf, proj_points, PL_SPHEROID_SPHERE, 1.0, 0.0, 0.0, 0.0, 0.0);
    proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
    error = pl_unproject_transverse_mercator(ctx, proj_buf, orig_points2, PL_SPHEROID_SPHERE, 1.0, 0.0, 0.0, 0.0, 0.0);
    pl_unload_projection_data(proj_buf);

    consistency_failures += compare_points(orig_points, orig_points2, TEST_POINTS, "Spherical, centered");

    error = pl_project_transverse_mercator(ctx, orig_buf, proj_points, PL_SPHEROID_SPHERE, 1.0, 0.0, 0.0, 10.0, 10.0);
    proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
    error = pl_unproject_transverse_mercator(ctx, proj_buf, orig_points2, PL_SPHEROID_SPHERE, 1.0, 0.0, 0.0, 10.0, 10.0);
    pl_unload_projection_data(proj_buf);

    consistency_failures += compare_points(orig_points, orig_points2, TEST_POINTS, "Spherical, off-center");

    error = pl_project_transverse_mercator(ctx, orig_buf, proj_points, PL_SPHEROID_WGS_84, 1.0, 0.0, 0.0, 0.0, 0.0);
    proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
    error = pl_unproject_transverse_mercator(ctx, proj_buf, orig_points2, PL_SPHEROID_WGS_84, 1.0, 0.0, 0.0, 0.0, 0.0);
    pl_unload_projection_data(proj_buf);

    consistency_failures += compare_points(orig_points, orig_points2, TEST_POINTS, "Ellipsoidal, centered");

    error = pl_project_transverse_mercator(ctx, orig_buf, proj_points, PL_SPHEROID_WGS_84, 1.0, 0.0, 0.0, 10.0, 10.0);
    proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
    error = pl_unproject_transverse_mercator(ctx, proj_buf, orig_points2, PL_SPHEROID_WGS_84, 1.0, 0.0, 0.0, 10.0, 10.0);
    pl_unload_projection_data(proj_buf);

    consistency_failures += compare_points(orig_points, orig_points2, TEST_POINTS, "Ellipsoidal, off-center");

    return consistency_failures;
}

int test_winkel_tripel(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points) {
    int error = CL_SUCCESS;
    int consistency_failures = 0;
    PLProjectionBuffer *proj_buf = NULL;
    float proj_points[2*TEST_POINTS];
    float orig_points2[2*TEST_POINTS];

    error = pl_project_winkel_tripel(ctx, orig_buf, proj_points,
            1.0, 0.0, 0.0, 0.0, 60.0);
    proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
    error = pl_unproject_winkel_tripel(ctx, proj_buf, orig_points2,
            1.0, 0.0, 0.0, 0.0, 60.0);
    pl_unload_projection_data(proj_buf);

    consistency_failures += compare_points(orig_points, orig_points2, TEST_POINTS, 
            "Centered");

    error = pl_project_winkel_tripel(ctx, orig_buf, proj_points,
            1.0, 0.0, 0.0, 10.0, 60.0);
    proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
    error = pl_unproject_winkel_tripel(ctx, proj_buf, orig_points2,
            1.0, 0.0, 0.0, 10.0, 60.0);
    pl_unload_projection_data(proj_buf);

    consistency_failures += compare_points(orig_points, orig_points2, TEST_POINTS, 
            "Off-center");

    return consistency_failures;
}

int compare_points(float *points1, float *points2, int count, char *name) {
    int i;
    int failures = 0;
    printf("-- %s... ", name);
    double max_delta_x = 0.0;
    double max_delta_y = 0.0;
    double max_delta_x_lat, max_delta_x_lon;
    double max_delta_y_lat, max_delta_y_lon;
    for (i=0; i<count; i++) {
        double delta_x = fabs(points1[2*i] - points2[2*i]);
        double delta_y = fabs(points1[2*i+1] - points2[2*i+1]);
        if (delta_x > TOL || delta_y > TOL) {
            failures++;
            if (delta_x > max_delta_x) {
                max_delta_x = delta_x;
                max_delta_x_lon = points1[2*i];
                max_delta_x_lat = points1[2*i+1];
            }
            if (delta_y > max_delta_y) {
                max_delta_y = delta_y;
                max_delta_y_lon = points1[2*i];
                max_delta_y_lat = points1[2*i+1];
            }
        }
    }
    if (failures) {
        printf("%d failures\n", failures);
        printf("**** Max longitudinal error: %lf at (%lf, %lf)\n", max_delta_x, max_delta_x_lon, max_delta_x_lat);
        printf("**** Max latitudinal error: %lf at (%lf, %lf)\n", max_delta_y, max_delta_y_lon, max_delta_y_lat);
    } else {
        printf("ok\n");
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

    printf("Compilation failures: %d\n", failures);

    code = pl_compile_code(ctx, "kernel", 0xFFFF, &error);
    error = pl_load_code(ctx, code);
    pl_release_code(code);
    if (error != CL_SUCCESS) {
        printf("Failed to load code: %d\n", error);
        return 1;
    }

    float orig_points[2*TEST_POINTS];
    int i;
    PLProjectionBuffer *orig_buf = NULL;

    for (i=0; i<TEST_POINTS; i++) {
        orig_points[2*i] = 45 * sin(2 * M_PI * i / TEST_POINTS);
        orig_points[2*i+1] = 45 * cos(2 * M_PI * i / TEST_POINTS);
    }

    orig_buf = pl_load_projection_data(ctx, orig_points, TEST_POINTS, 1, &error);
    if (orig_buf == NULL) {
        printf("Failed to load points: %d\n", error);
        return 1;
    }

    int consistency_failures = 0;

    printf("\nTesting consistency of Albers Equal Area\n");
    consistency_failures += test_albers_equal_area(ctx, orig_buf, orig_points);

    printf("\nTesting consistency of American Polyconic\n");
    consistency_failures += test_american_polyconic(ctx, orig_buf, orig_points);

    printf("\nTesting consistency of Lambert Azimuthal Equal Area\n");
    consistency_failures += test_lambert_azimuthal_equal_area(ctx, orig_buf, orig_points);

    printf("\nTesting consistency of Lambert Conformal Conic\n");
    consistency_failures += test_lambert_conformal_conic(ctx, orig_buf, orig_points);

    printf("\nTesting consistency of Mercator\n");
    consistency_failures += test_mercator(ctx, orig_buf, orig_points);

    printf("\nTesting consistency of Robinson\n");
    consistency_failures += test_robinson(ctx, orig_buf, orig_points);

    printf("\nTesting consistency of Transverse Mercator\n");
    consistency_failures += test_transverse_mercator(ctx, orig_buf, orig_points);

    printf("\nTesting consistency of Winkel Tripel\n");
    consistency_failures += test_winkel_tripel(ctx, orig_buf, orig_points);

    printf("\nTotal consistency failures: %d\n", consistency_failures);

    pl_unload_projection_data(orig_buf);

    pl_unload_code(ctx);
    pl_context_free(ctx);

    return 0;
}
