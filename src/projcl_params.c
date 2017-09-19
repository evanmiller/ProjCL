
#include <projcl/projcl.h>
#include "projcl_util.h"
#include "projcl_spheroid.h"
#include <math.h>

PLProjectionParams *pl_params_init() {
    PLProjectionParams *params = calloc(1, sizeof(PLProjectionParams));
    params->scale = 1.0;
    params->rlat1 = NAN;
    params->rlat2 = NAN;
    return params;
}

void pl_params_free(PLProjectionParams *params) {
    free(params);
}

void pl_params_set_scale(PLProjectionParams *params, double k0) {
    params->scale = k0;
}

void pl_params_set_spheroid(PLProjectionParams *params, PLSpheroid spheroid) {
    params->spheroid = spheroid;
}

void pl_params_set_false_easting(PLProjectionParams *params, double x0) {
    params->x0 = x0;
}

void pl_params_set_false_northing(PLProjectionParams *params, double y0) {
    params->y0 = y0;
}

void pl_params_set_latitude_of_origin(PLProjectionParams *params, double lat0) {
    params->lat0 = lat0;
}

void pl_params_set_longitude_of_origin(PLProjectionParams *params, double lon0) {
    params->lon0 = lon0;
}

void pl_params_set_standard_parallel(PLProjectionParams *params, double rlat1) {
    params->rlat1 = rlat1;
}

void pl_params_set_standard_parallels(PLProjectionParams *params, double rlat1, double rlat2) {
    params->rlat1 = rlat1;
    params->rlat2 = rlat2;
}

void pl_params_set_mercator_params_from_pathological_lambert_conformal_conic_params(
        PLProjectionParams *dst, PLProjectionParams *src) {
    double cosphi1 = cos(src->rlat1 * DEG_TO_RAD);
    PLSpheroidInfo info = _pl_get_spheroid_info(src->spheroid);

    dst->scale = src->scale*cosphi1;
    dst->x0 = src->x0 - src->scale*info.major_axis*cosphi1*src->lon0*DEG_TO_RAD;
    dst->y0 = src->y0 - src->scale*info.major_axis*cosphi1*log(tan(0.5*src->lat0*DEG_TO_RAD+M_PI_4));
    dst->spheroid = src->spheroid;
}
