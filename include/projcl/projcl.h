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
                                             const float *xy, int n, int *outError);
void pl_unload_datum_shift_data(PLDatumShiftBuffer *pl_buf);

cl_int pl_shift_datum(PLContext *pl_ctx, PLDatum src_datum, PLDatum dst_datum, PLSpheroid dst_spheroid,
                      PLDatumShiftBuffer *pl_buf, float *xy_out);




PLProjectionBuffer *pl_load_projection_data(PLContext *pl_ctx, const float *xy, int n, int copy, int *outError);
void pl_unload_projection_data(PLProjectionBuffer *pl_buf);

cl_int pl_project_albers_equal_area(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
    PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0, float rlat1, float rlat2);
cl_int pl_unproject_albers_equal_area(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
    PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0, float rlat1, float rlat2);
cl_int pl_project_american_polyconic(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
    PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0);
cl_int pl_unproject_american_polyconic(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
    PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0);
cl_int pl_project_lambert_azimuthal_equal_area(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
    PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0);
cl_int pl_unproject_lambert_azimuthal_equal_area(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
    PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0);
cl_int pl_project_lambert_conformal_conic(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
    PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0, float rlat1, float rlat2);
cl_int pl_unproject_lambert_conformal_conic(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
    PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0, float rlat1, float rlat2);
cl_int pl_project_mercator(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
    PLSpheroid pl_ell, float scale, float x0, float y0);
cl_int pl_unproject_mercator(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
    PLSpheroid pl_ell, float scale, float x0, float y0);
cl_int pl_project_oblique_stereographic(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
    PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0);
cl_int pl_unproject_oblique_stereographic(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
    PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0);
cl_int pl_project_robinson(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
        float scale, float x0, float y0);
cl_int pl_unproject_robinson(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
        float scale, float x0, float y0);
cl_int pl_project_transverse_mercator(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out, 
    PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0);
cl_int pl_unproject_transverse_mercator(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out, 
    PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0);
cl_int pl_project_winkel_tripel(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
    float scale, float x0, float y0, float lon0, float rlat1);
cl_int pl_unproject_winkel_tripel(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
    float scale, float x0, float y0, float lon0, float rlat1);


PLForwardGeodesicFixedDistanceBuffer *pl_load_forward_geodesic_fixed_distance_data(PLContext *pl_ctx,
    const float *xy_in, int xy_count, const float *az_in, int az_count, cl_int *outError);
void pl_unload_forward_geodesic_fixed_distance_data(PLForwardGeodesicFixedDistanceBuffer *pl_buf);
cl_int pl_forward_geodesic_fixed_distance(PLContext *pl_ctx, PLForwardGeodesicFixedDistanceBuffer *pl_buf, float *xy_out, 
    PLSpheroid pl_ell, float distance);


PLForwardGeodesicFixedAngleBuffer *pl_load_forward_geodesic_fixed_angle_data(PLContext *pl_ctx, 
    const float *dist_in, int dist_count, cl_int *outError);
void pl_unload_forward_geodesic_fixed_angle_data(PLForwardGeodesicFixedAngleBuffer *pl_buf);
cl_int pl_forward_geodesic_fixed_angle(PLContext *pl_ctx, PLForwardGeodesicFixedAngleBuffer *pl_buf, float *xy_in, float *xy_out, 
    PLSpheroid pl_ell, float angle);



PLInverseGeodesicBuffer *pl_load_inverse_geodesic_data(PLContext *pl_ctx,
													   const float *xy1_in, int xy1_count, int xy1_copy,
													   const float *xy2_in, int xy2_count,
													   cl_int *outError);
void pl_unload_inverse_geodesic_data(PLInverseGeodesicBuffer *pl_buf);

cl_int pl_inverse_geodesic(PLContext *pl_ctx, PLInverseGeodesicBuffer *pl_buf, float *dist_out,
						   PLSpheroid pl_ell, float scale);
