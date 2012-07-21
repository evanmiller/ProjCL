/*
 *  projcl.h
 *  ProjCL
 *
 *  Created by Evan Miller on 2/7/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#import <OpenCL/OpenCL.h>

#import "projcl_types.h"

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
cl_int pl_project_robinson(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out);
cl_int pl_unproject_robinson(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out);
cl_int pl_project_transverse_mercator(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out, 
    PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0);
cl_int pl_unproject_transverse_mercator(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out, 
    PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0);
cl_int pl_project_winkel_tripel(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
    float scale, float x0, float y0, float lon0, float rlat1);
cl_int pl_unproject_winkel_tripel(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
    float scale, float x0, float y0, float lon0, float rlat1);


PLImageBuffer *pl_load_image(PLContext *pl_ctx, 
                             int channel_order,
                             int channel_type,
                             size_t width,
                             size_t height,
                             size_t row_pitch,
                             const void* pvData,
                             int copy,
                             int *outError);
PLImageArrayBuffer *pl_load_image_array(PLContext *pl_ctx,
                                        int channel_order,
                                        int channel_type,
                                        size_t width,
                                        size_t height,
                                        size_t row_pitch, 
                                        size_t slice_pitch,
                                        size_t tiles_across,
                                        size_t tiles_down,
                                        const void *pvData,
                                        int do_copy,
                                        cl_int *outError);
void pl_unload_image(PLImageBuffer *buf);
void pl_unload_image_array(PLImageArrayBuffer *buf);

cl_int pl_sample_image(PLContext *pl_ctx, PLPointGridBuffer *grid, PLImageBuffer *img, PLImageFilter filter, unsigned char *outData);
cl_int pl_sample_image_array(PLContext *pl_ctx, PLPointGridBuffer *grid, PLImageArrayBuffer *bufs, PLImageFilter filter,
                             unsigned char *outData);

PLPointGridBuffer *pl_load_empty_grid(PLContext *pl_ctx, int count_x, int count_y, int *outError);
PLPointGridBuffer *pl_load_grid(PLContext *pl_ctx,
                                float origin_x, float width, int count_x,
                                float origin_y, float height, int count_y,
                                int *outError);
void pl_unload_grid(PLPointGridBuffer *grid);
cl_int pl_transform_grid(PLContext *pl_ctx, PLPointGridBuffer *src, PLPointGridBuffer *dst, float sx, float sy, float tx, float ty);
cl_int pl_shift_grid_datum(PLContext *pl_ctx, PLPointGridBuffer *src, PLDatum src_datum, PLSpheroid src_spheroid,
                           PLPointGridBuffer *dst, PLDatum dst_datum, PLSpheroid dst_spheroid);
cl_int pl_project_grid_albers_equal_area(PLContext *pl_ctx, PLPointGridBuffer *src, PLPointGridBuffer *dst,
                                         PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0,
                                         float rlat1, float rlat2);
cl_int pl_unproject_grid_albers_equal_area(PLContext *pl_ctx, PLPointGridBuffer *src, PLPointGridBuffer *dst,
                                           PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0,
                                           float rlat1, float rlat2);
cl_int pl_project_grid_american_polyconic(PLContext *pl_ctx, PLPointGridBuffer *src, PLPointGridBuffer *dst,
                                          PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0);
cl_int pl_unproject_grid_american_polyconic(PLContext *pl_ctx, PLPointGridBuffer *src, PLPointGridBuffer *dst,
                                            PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0);
cl_int pl_project_grid_lambert_azimuthal_equal_area(PLContext *pl_ctx, PLPointGridBuffer *src, PLPointGridBuffer *dst,
                                                    PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0);
cl_int pl_unproject_grid_lambert_azimuthal_equal_area(PLContext *pl_ctx, PLPointGridBuffer *src, PLPointGridBuffer *dst,
                                                      PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0);
cl_int pl_project_grid_lambert_conformal_conic(PLContext *pl_ctx, PLPointGridBuffer *src, PLPointGridBuffer *dst,
                                               PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0,
                                               float rlat1, float rlat2);
cl_int pl_unproject_grid_lambert_conformal_conic(PLContext *pl_ctx, PLPointGridBuffer *src, PLPointGridBuffer *dst,
                                                 PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0,
                                                 float rlat1, float rlat2);
cl_int pl_project_grid_mercator(PLContext *pl_ctx, PLPointGridBuffer *src, PLPointGridBuffer *dst,
                                PLSpheroid pl_ell, float scale, float x0, float y0);
cl_int pl_unproject_grid_mercator(PLContext *pl_ctx, PLPointGridBuffer *src, PLPointGridBuffer *dst,
                                  PLSpheroid pl_ell, float scale, float x0, float y0);
cl_int pl_project_grid_robinson(PLContext *pl_ctx, PLPointGridBuffer *src, PLPointGridBuffer *dst);
cl_int pl_unproject_grid_robinson(PLContext *pl_ctx, PLPointGridBuffer *src, PLPointGridBuffer *dst);
cl_int pl_project_grid_transverse_mercator(PLContext *pl_ctx, PLPointGridBuffer *src, PLPointGridBuffer *dst,
                                           PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0);
cl_int pl_unproject_grid_transverse_mercator(PLContext *pl_ctx, PLPointGridBuffer *src, PLPointGridBuffer *dst, 
                                             PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0);
cl_int pl_project_grid_winkel_tripel(PLContext *pl_ctx, PLPointGridBuffer *src, PLPointGridBuffer *dst, 
                                     float scale, float x0, float y0, float lon0, float rlat1);
cl_int pl_unproject_grid_winkel_tripel(PLContext *pl_ctx, PLPointGridBuffer *src, PLPointGridBuffer *dst, 
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
