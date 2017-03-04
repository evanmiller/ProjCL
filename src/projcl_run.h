/*
 *  projcl_run.h
 *  Magic Maps
 *
 *  Created by Evan Miller on 2/12/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#import <OpenCL/opencl.h>
#import <projcl/projcl_types.h>

cl_int pl_read_buffer(cl_command_queue queue, cl_mem xy_out_buf, float *xy_out, size_t out_count);

cl_int pl_enqueue_kernel_albers_equal_area(cl_kernel kernel, PLContext *pl_ctx, cl_mem xy_in, cl_mem xy_out, size_t count,
    PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0, float rlat1, float rlat2);
cl_int pl_enqueue_kernel_american_polyconic(cl_kernel kernel, PLContext *pl_ctx, cl_mem xy_in, cl_mem xy_out, size_t count,
    PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0);
cl_int pl_enqueue_kernel_lambert_conformal_conic(cl_kernel kernel, PLContext *pl_ctx, cl_mem xy_in, cl_mem xy_out, size_t count,
    PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0, float rlat1, float rlat2);
cl_int pl_enqueue_kernel_lambert_azimuthal_equal_area(cl_kernel kernel, PLContext *pl_ctx, cl_mem xy_in, cl_mem xy_out, size_t count,
    PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0);
cl_int pl_enqueue_kernel_mercator(cl_kernel kernel, PLContext *pl_ctx, cl_mem xy_in, cl_mem xy_out, size_t count,
    PLSpheroid pl_ell, float scale, float x0, float y0);
cl_int pl_enqueue_kernel_oblique_stereographic(cl_kernel kernel, PLContext *pl_ctx, cl_mem xy_in, cl_mem xy_out, size_t count,
    PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0);
cl_int pl_enqueue_kernel_robinson(cl_kernel kernel, PLContext *pl_ctx, cl_mem xy_in, cl_mem xy_out, size_t count,
        float scale, float x0, float y0);
cl_int pl_enqueue_kernel_transverse_mercator(cl_kernel kernel, PLContext *pl_ctx, cl_mem xy_in, cl_mem xy_out, size_t count,
    PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0);
cl_int pl_enqueue_kernel_winkel_tripel(cl_kernel kernel, PLContext *pl_ctx, cl_mem xy_in, cl_mem xy_out, size_t count,
    float scale, float x0, float y0, float lon0, float rlat1);

cl_int pl_run_kernel_forward_geodesic_fixed_distance(cl_kernel kernel, PLContext *pl_ctx, 
    PLForwardGeodesicFixedDistanceBuffer *pl_buf, float *xy_out, PLSpheroid pl_ell, float distance);
cl_int pl_run_kernel_forward_geodesic_fixed_angle(cl_kernel kernel, PLContext *pl_ctx,
    PLForwardGeodesicFixedAngleBuffer *pl_buf, float *xy_in, float *xy_out, PLSpheroid pl_ell, float angle);
cl_int pl_run_kernel_inverse_geodesic(cl_kernel inv_kernel, PLContext *pl_ctx, PLInverseGeodesicBuffer *pl_buf,
    float *dist_out, PLSpheroid pl_ell, float scale);

cl_int pl_run_kernel_geodesic_to_cartesian(cl_kernel g2c_kernel, PLContext *pl_ctx, PLDatumShiftBuffer *pl_buf,
                               PLSpheroid pl_ell);
cl_int pl_run_kernel_transform_cartesian(cl_kernel transform_kernel, PLContext *pl_ctx, PLDatumShiftBuffer *pl_buf,
                                         PLDatum src_datum, PLDatum dst_datum);
cl_int pl_run_kernel_cartesian_to_geodesic(cl_kernel c2g_kernel, PLContext *pl_ctx, PLDatumShiftBuffer *pl_buf,
                                           float *xy_out, PLSpheroid pl_ell);
