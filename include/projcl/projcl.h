/*
 *  projcl.h
 *  ProjCL
 *
 *  Created by Evan Miller on 2/7/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "projcl_types.h"

PLContext *pl_context_init(cl_device_type type, cl_int *outError);
void pl_context_free(PLContext *pl_ctx);

PLCode *pl_compile_code(PLContext *pl_ctx, const char *path, long modules, cl_int *outError);
void pl_release_code(PLCode *pl_code);

cl_int pl_load_code(PLContext *pl_ctx, PLCode *pl_code);
void pl_unload_code(PLContext *pl_ctx);




PLDatumShiftBuffer *pl_load_datum_shift_data(PLContext *pl_ctx, PLSpheroid src_spheroid, 
        const float *xy, size_t count, cl_int *outError);
void pl_unload_datum_shift_data(PLDatumShiftBuffer *pl_buf);

cl_int pl_shift_datum(PLContext *pl_ctx, PLDatum src_datum, PLDatum dst_datum, PLSpheroid dst_spheroid,
        PLDatumShiftBuffer *pl_buf, float *xy_out);




PLProjectionBuffer *pl_load_projection_data(PLContext *pl_ctx, const float *xy, size_t count, cl_bool copy, int *outError);
PLProjectionBuffer *pl_load_projection_data_2(PLContext *pl_ctx, const float *x, const float *y, size_t count, int *outError);
void pl_unload_projection_data(PLProjectionBuffer *pl_buf);
cl_int pl_compare_projection_buffers(PLContext *pl_ctx, PLProjectionBuffer *pl_buf_1, PLProjectionBuffer *pl_buf_2, char **error_string);


PLProjectionParams *pl_params_init();
void pl_params_free(PLProjectionParams *params);
void pl_params_set_scale(PLProjectionParams *params, double k0);
void pl_params_set_spheroid(PLProjectionParams *params, PLSpheroid spheroid);
void pl_params_set_false_easting(PLProjectionParams *params, double x0);
void pl_params_set_false_northing(PLProjectionParams *params, double y0);
void pl_params_set_latitude_of_origin(PLProjectionParams *params, double lat0);
void pl_params_set_longitude_of_origin(PLProjectionParams *params, double lon0);
void pl_params_set_standard_parallel(PLProjectionParams *params, double rlat1);
void pl_params_set_standard_parallels(PLProjectionParams *params, double rlat1, double rlat2);
void pl_params_set_mercator_params_from_pathological_lambert_conformal_conic_params(
        PLProjectionParams *dst, PLProjectionParams *src);


cl_int pl_project_points_forward(PLContext *pl_ctx, PLProjection proj, PLProjectionParams *params,
        PLProjectionBuffer *pl_buf, float *xy_out);
cl_int pl_project_points_reverse(PLContext *pl_ctx, PLProjection proj, PLProjectionParams *params,
        PLProjectionBuffer *pl_buf, float *xy_out);
// And the X/Y interface
cl_int pl_project_points_forward_2(PLContext *pl_ctx, PLProjection proj, PLProjectionParams *params,
	PLProjectionBuffer *pl_buf, float *x_out, float *y_out);
cl_int pl_project_points_reverse_2(PLContext *pl_ctx, PLProjection proj, PLProjectionParams *params,
	PLProjectionBuffer *pl_buf, float *x_out, float *y_out);

PLForwardGeodesicFixedDistanceBuffer *pl_load_forward_geodesic_fixed_distance_data(PLContext *pl_ctx,
    const float *xy_in, size_t xy_count, const float *az_in, size_t az_count, cl_int *outError);
void pl_unload_forward_geodesic_fixed_distance_data(PLForwardGeodesicFixedDistanceBuffer *pl_buf);
cl_int pl_forward_geodesic_fixed_distance(PLContext *pl_ctx, PLForwardGeodesicFixedDistanceBuffer *pl_buf,
        float *xy_out, PLSpheroid pl_ell, double distance);


PLForwardGeodesicFixedAngleBuffer *pl_load_forward_geodesic_fixed_angle_data(PLContext *pl_ctx, 
    const float *dist_in, size_t dist_count, cl_int *outError);
void pl_unload_forward_geodesic_fixed_angle_data(PLForwardGeodesicFixedAngleBuffer *pl_buf);
cl_int pl_forward_geodesic_fixed_angle(PLContext *pl_ctx, PLForwardGeodesicFixedAngleBuffer *pl_buf,
        float *xy_in, float *xy_out, PLSpheroid pl_ell, double angle);



PLInverseGeodesicBuffer *pl_load_inverse_geodesic_data(PLContext *pl_ctx,
        const float *xy1_in, size_t xy1_count, cl_bool xy1_copy,
        const float *xy2_in, size_t xy2_count,
        cl_int *outError);
void pl_unload_inverse_geodesic_data(PLInverseGeodesicBuffer *pl_buf);

cl_int pl_inverse_geodesic(PLContext *pl_ctx, PLInverseGeodesicBuffer *pl_buf, float *dist_out,
        PLSpheroid pl_ell, double scale);
