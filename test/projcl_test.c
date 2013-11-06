
#include <stdio.h>
#include <math.h>
#include <string.h>

#include <OpenCL/OpenCL.h>
#include <sys/time.h>
#include <projcl/projcl.h>
#include <projcl/projcl_warp.h>

#ifdef HAVE_PROJ4
#include <proj_api.h>
#endif

#define RAD_TO_DEG   57.29577951308232
#define TEST_POINTS 1000
#define TOL 1.e-5

typedef struct test_params_s {
  char   name[80];
  int    ell;
  double lon0;
  double lat0;
  double rlat1;
  double rlat2;
} test_params_t;

static test_params_t albers_equal_area_tests[] = { 
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
};

static test_params_t american_polyconic_tests[] = {
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
};

static test_params_t lambert_azimuthal_equal_area_tests[] = {
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
};

static test_params_t lambert_conformal_conic_tests[] = {
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
  { .name = "Spherical, symmetric standard parallels",
    .ell = PL_SPHEROID_SPHERE,
    .lon0 = 0.0,
    .lat0 = 0.0,
    .rlat1 = -30.0,
    .rlat2 = 30.0 },
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
  { .name = "Ellipsoidal, symmetric standard parallels",
    .ell = PL_SPHEROID_WGS_84,
    .lon0 = 0.0,
    .lat0 = 0.0,
    .rlat1 = -30.0,
    .rlat2 = 30.0 },
};

static test_params_t mercator_tests[] = {
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
};

static test_params_t robinson_tests[] = {
  { .name = "Spherical",
    .ell = PL_SPHEROID_SPHERE,
    .lon0 = NAN,
    .lat0 = NAN,
    .rlat1 = NAN,
    .rlat2 = NAN }
};

static test_params_t transverse_mercator_tests[] = {
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
};

static test_params_t winkel_tripel_tests[] = {
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
};

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

int compare_proj4_inv(float *proj_points, float *orig_points, char *desc1, char *desc2) {
    int failures = 0;
#ifdef HAVE_PROJ4
    double points_proj4[2*TEST_POINTS];
    float orig_points_proj4[2*TEST_POINTS];
    projPJ pj_out, pj_in;
    int i;
    int error;

    for (i=0; i<TEST_POINTS; i++) {
      points_proj4[2*i] = proj_points[2*i];
      points_proj4[2*i+1] = proj_points[2*i+1];
    }

    if ((pj_in = pj_init_plus(desc1)) == NULL) {
      printf("Failed to init Proj.4 input structure: %s\n", desc1);
      exit(1);
    }

    if ((pj_out = pj_init_plus(desc2)) == NULL) {
      printf("Failed to init Proj.4 output structure: %s\n", desc2);
      exit(1);
    }

    error = pj_transform(pj_in, pj_out, TEST_POINTS, 2, points_proj4, points_proj4 + 1, NULL);
    if (error != 0) {
      printf("Error projecting: %d\n", error);
      exit(1);
    }

    pj_free(pj_in);
    pj_free(pj_out);

    for (i=0; i<TEST_POINTS; i++) {
      orig_points_proj4[2*i] = points_proj4[2*i] * RAD_TO_DEG;
      orig_points_proj4[2*i+1] = points_proj4[2*i+1] * RAD_TO_DEG;
    }

    failures = compare_points(orig_points_proj4, orig_points, 
            TEST_POINTS, "...inverse same as Proj.4");
#endif
    return failures;
}

int compare_proj4_fwd(float *orig_points, float *proj_points, char *desc1, char *desc2) {
    int failures = 0;
#ifdef HAVE_PROJ4
    double points_proj4[2*TEST_POINTS];
    float proj_points_proj4[2*TEST_POINTS];
    projPJ pj_out, pj_in;
    int i;
    int error;

    for (i=0; i<TEST_POINTS; i++) {
      points_proj4[2*i] = orig_points[2*i] * DEG_TO_RAD;
      points_proj4[2*i+1] = orig_points[2*i+1] * DEG_TO_RAD;
    }

    if ((pj_in = pj_init_plus(desc1)) == NULL) {
      printf("Failed to init Proj.4 input structure: %s\n", desc1);
      exit(1);
    }

    if ((pj_out = pj_init_plus(desc2)) == NULL) {
      printf("Failed to init Proj.4 output structure: %s\n", desc2);
      exit(1);
    }

    error = pj_transform(pj_in, pj_out, TEST_POINTS, 2, points_proj4, points_proj4 + 1, NULL);
    if (error != 0) {
      printf("Error projecting: %d\n", error);
      exit(1);
    }

    pj_free(pj_in);
    pj_free(pj_out);

    for (i=0; i<TEST_POINTS; i++) {
      proj_points_proj4[2*i] = points_proj4[2*i];
      proj_points_proj4[2*i+1] = points_proj4[2*i+1];
    }

    failures = compare_points(proj_points_proj4, proj_points, 
            TEST_POINTS, "...same as Proj.4");
#endif
    return failures;
}

int sprintf_proj4(char *buf, const char *name, test_params_t params) {
    char *p = buf;
    p += sprintf(p, "+proj=%s", name);
    if (params.ell != -1)
        p += sprintf(p, " +ellps=%s", params.ell == PL_SPHEROID_SPHERE ? "sphere" : "WGS84");
    if (strcmp(name, "latlong") != 0) {
        if (!isnan(params.lat0) && params.lat0 != 0.0)
            p += sprintf(p, " +lat_0=%.1lf%c", fabs(params.lat0), params.lat0 > 0.0 ? 'n' : 's');
        if (!isnan(params.lon0) && params.lon0 != 0.0)
            p += sprintf(p, " +lon_0=%.1lf%c", fabs(params.lon0), params.lon0 > 0.0 ? 'e' : 'w');
        if (!isnan(params.rlat1))
            p += sprintf(p, " +lat_1=%.1lf%c", fabs(params.rlat1), params.rlat1 > 0.0 ? 'n' : 's');
        if (!isnan(params.rlat2))
            p += sprintf(p, " +lat_2=%.1lf%c", fabs(params.rlat2), params.rlat2 > 0.0 ? 'n' : 's');
    }

    return 0;
}

int test_albers_equal_area(PLContext *ctx, 
        PLProjectionBuffer *orig_buf, float *orig_points) {
    int error = CL_SUCCESS;
    int consistency_failures = 0;
    PLProjectionBuffer *proj_buf = NULL;
    float proj_points[2*TEST_POINTS];
    float orig_points2[2*TEST_POINTS];
    int i;
    char orig_string[80];
    char proj_string[80];

    for (i=0; i<sizeof(albers_equal_area_tests)/sizeof(test_params_t); i++) {
        test_params_t test = albers_equal_area_tests[i];
        error = pl_project_albers_equal_area(ctx, orig_buf, proj_points, 
              test.ell, 1.0, 0.0, 0.0, test.lon0, test.lat0, test.rlat1, test.rlat2);
        proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
        error = pl_unproject_albers_equal_area(ctx, proj_buf, orig_points2, 
                test.ell, 1.0, 0.0, 0.0, test.lon0, test.lat0, test.rlat1, test.rlat2);
        pl_unload_projection_data(proj_buf);

        consistency_failures += compare_points(orig_points, orig_points2, 
                TEST_POINTS, test.name);
        sprintf_proj4(orig_string, "latlong", test);
        sprintf_proj4(proj_string, "aea", test);
        consistency_failures += compare_proj4_fwd(orig_points, proj_points, orig_string, proj_string);
    }

    return consistency_failures;
}

int test_american_polyconic(PLContext *ctx, 
        PLProjectionBuffer *orig_buf, float *orig_points) {
    int error = CL_SUCCESS;
    int consistency_failures = 0;
    PLProjectionBuffer *proj_buf = NULL;
    float proj_points[2*TEST_POINTS];
    float orig_points2[2*TEST_POINTS];

    int i;
    char orig_string[80];
    char proj_string[80];

    for (i=0; i<sizeof(american_polyconic_tests)/sizeof(test_params_t); i++) {
        test_params_t test = american_polyconic_tests[i];
        error = pl_project_american_polyconic(ctx, orig_buf, proj_points, 
              test.ell, 1.0, 0.0, 0.0, test.lon0, test.lat0);
        proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
        error = pl_unproject_american_polyconic(ctx, proj_buf, orig_points2, 
                test.ell, 1.0, 0.0, 0.0, test.lon0, test.lat0);
        pl_unload_projection_data(proj_buf);

        consistency_failures += compare_points(orig_points, orig_points2, 
                TEST_POINTS, test.name);
        sprintf_proj4(orig_string, "latlong", test);
        sprintf_proj4(proj_string, "poly", test);
        consistency_failures += compare_proj4_fwd(orig_points, proj_points, orig_string, proj_string);
    }

    return consistency_failures;
}

int test_lambert_azimuthal_equal_area(PLContext *ctx, 
        PLProjectionBuffer *orig_buf, float *orig_points) {
    int error = CL_SUCCESS;
    int consistency_failures = 0;
    PLProjectionBuffer *proj_buf = NULL;
    float proj_points[2*TEST_POINTS];
    float orig_points2[2*TEST_POINTS];

    int i;
    char orig_string[80];
    char proj_string[80];

    for (i=0; i<sizeof(lambert_azimuthal_equal_area_tests)/sizeof(test_params_t); i++) {
        test_params_t test = lambert_azimuthal_equal_area_tests[i];
        error = pl_project_lambert_azimuthal_equal_area(ctx, orig_buf, proj_points, 
              test.ell, 1.0, 0.0, 0.0, test.lon0, test.lat0);
        proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
        error = pl_unproject_lambert_azimuthal_equal_area(ctx, proj_buf, orig_points2, 
                test.ell, 1.0, 0.0, 0.0, test.lon0, test.lat0);
        pl_unload_projection_data(proj_buf);

        consistency_failures += compare_points(orig_points, orig_points2, 
                TEST_POINTS, test.name);
        sprintf_proj4(orig_string, "latlong", test);
        sprintf_proj4(proj_string, "laea", test);
        consistency_failures += compare_proj4_fwd(orig_points, proj_points, orig_string, proj_string);
    }

    return consistency_failures;
}

int test_lambert_conformal_conic(PLContext *ctx, 
        PLProjectionBuffer *orig_buf, float *orig_points) {
    int error = CL_SUCCESS;
    int consistency_failures = 0;
    PLProjectionBuffer *proj_buf = NULL;
    float proj_points[2*TEST_POINTS];
    float orig_points2[2*TEST_POINTS];
    int i;
    char orig_string[80];
    char proj_string[80];

    for (i=0; i<sizeof(lambert_conformal_conic_tests)/sizeof(test_params_t); i++) {
        test_params_t test = lambert_conformal_conic_tests[i];
        error = pl_project_lambert_conformal_conic(ctx, orig_buf, proj_points, 
              test.ell, 1.0, 0.0, 0.0, test.lon0, test.lat0, test.rlat1, test.rlat2);
        proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
        error = pl_unproject_lambert_conformal_conic(ctx, proj_buf, orig_points2, 
                test.ell, 1.0, 0.0, 0.0, test.lon0, test.lat0, test.rlat1, test.rlat2);
        pl_unload_projection_data(proj_buf);

        consistency_failures += compare_points(orig_points, orig_points2, 
                TEST_POINTS, test.name);
        sprintf_proj4(orig_string, "latlong", test);
        sprintf_proj4(proj_string, "lcc", test);
        consistency_failures += compare_proj4_fwd(orig_points, proj_points, orig_string, proj_string);
    }

    return consistency_failures;
}

int test_mercator(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points) {
    int error = CL_SUCCESS;
    int consistency_failures = 0;
    PLProjectionBuffer *proj_buf = NULL;
    float proj_points[2*TEST_POINTS];
    float orig_points2[2*TEST_POINTS];
    int i;
    char orig_string[80];
    char proj_string[80];

    for (i=0; i<sizeof(mercator_tests)/sizeof(test_params_t); i++) {
        test_params_t test = mercator_tests[i];
        error = pl_project_mercator(ctx, orig_buf, proj_points, 
              test.ell, 1.0, 0.0, 0.0);
        proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
        error = pl_unproject_mercator(ctx, proj_buf, orig_points2, 
                test.ell, 1.0, 0.0, 0.0);
        pl_unload_projection_data(proj_buf);

        consistency_failures += compare_points(orig_points, orig_points2, 
                TEST_POINTS, test.name);
        sprintf_proj4(orig_string, "latlong", test);
        sprintf_proj4(proj_string, "merc", test);
        consistency_failures += compare_proj4_fwd(orig_points, proj_points, orig_string, proj_string);
    }

    return consistency_failures;
}

int test_robinson(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points) {
    int error = CL_SUCCESS;
    int consistency_failures = 0;
    PLProjectionBuffer *proj_buf = NULL;
    float proj_points[2*TEST_POINTS];
    float orig_points2[2*TEST_POINTS];
    int i;
    char orig_string[80];
    char proj_string[80];

    for (i=0; i<sizeof(robinson_tests)/sizeof(test_params_t); i++) {
        test_params_t test = robinson_tests[i];
        error = pl_project_robinson(ctx, orig_buf, proj_points, 1000.0, 0.0, 0.0);
        proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
        error = pl_unproject_robinson(ctx, proj_buf, orig_points2, 1000.0, 0.0, 0.0);
        pl_unload_projection_data(proj_buf);

        consistency_failures += compare_points(orig_points, orig_points2, 
                TEST_POINTS, test.name);
        sprintf_proj4(orig_string, "latlong", test);
        sprintf_proj4(proj_string, "robin", test);
        consistency_failures += compare_proj4_fwd(orig_points, proj_points, orig_string, proj_string);
    }

    return consistency_failures;
}

int test_transverse_mercator(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points) {
    int error = CL_SUCCESS;
    int consistency_failures = 0;
    PLProjectionBuffer *proj_buf = NULL;
    float proj_points[2*TEST_POINTS];
    float orig_points2[2*TEST_POINTS];
    int i;
    char orig_string[80];
    char proj_string[80];

    for (i=0; i<sizeof(transverse_mercator_tests)/sizeof(test_params_t); i++) {
        test_params_t test = transverse_mercator_tests[i];
        error = pl_project_transverse_mercator(ctx, orig_buf, proj_points, 
              test.ell, 1.0, 0.0, 0.0, test.lon0, test.lat0);
        proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
        error = pl_unproject_transverse_mercator(ctx, proj_buf, orig_points2, 
                test.ell, 1.0, 0.0, 0.0, test.lon0, test.lat0);
        pl_unload_projection_data(proj_buf);

        consistency_failures += compare_points(orig_points, orig_points2, 
                TEST_POINTS, test.name);
        sprintf_proj4(orig_string, "latlong", test);
        sprintf_proj4(proj_string, "tmerc", test);
        consistency_failures += compare_proj4_fwd(orig_points, proj_points, orig_string, proj_string);
        consistency_failures += compare_proj4_inv(proj_points, orig_points2, proj_string, orig_string);
    }

    return consistency_failures;
}

int test_winkel_tripel(PLContext *ctx, PLProjectionBuffer *orig_buf, float *orig_points) {
    int error = CL_SUCCESS;
    int consistency_failures = 0;
    PLProjectionBuffer *proj_buf = NULL;
    float proj_points[2*TEST_POINTS];
    float orig_points2[2*TEST_POINTS];

    int i;
    char orig_string[80];
    char proj_string[80];

    for (i=0; i<sizeof(winkel_tripel_tests)/sizeof(test_params_t); i++) {
        test_params_t test = winkel_tripel_tests[i];
        error = pl_project_winkel_tripel(ctx, orig_buf, proj_points, 
              1.0, 0.0, 0.0, test.lon0, acos(M_2_PI) * RAD_TO_DEG);
        proj_buf = pl_load_projection_data(ctx, proj_points, TEST_POINTS, 1, &error);
        error = pl_unproject_winkel_tripel(ctx, proj_buf, orig_points2, 
                1.0, 0.0, 0.0, test.lon0, acos(M_2_PI) * RAD_TO_DEG);
        pl_unload_projection_data(proj_buf);

        consistency_failures += compare_points(orig_points, orig_points2, 
                TEST_POINTS, test.name);
        sprintf_proj4(orig_string, "latlong", test);
        sprintf_proj4(proj_string, "wintri", test);
        consistency_failures += compare_proj4_fwd(orig_points, proj_points, orig_string, proj_string);
    }

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
        double norm = sqrt(points1[2*i] * points1[2*i] + points1[2*i+1] * points1[2*i+1]);
        if (delta_x / norm > TOL || delta_y / norm > TOL) {
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
