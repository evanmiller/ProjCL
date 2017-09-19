/*
 *  projcl_run.h
 *  Magic Maps
 *
 *  Created by Evan Miller on 2/12/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

cl_kernel pl_find_kernel(PLContext *pl_ctx, const char *requested_name);
cl_kernel pl_find_projection_kernel(PLContext *pl_ctx, PLProjection proj, int fwd, PLSpheroid ell);

cl_int pl_read_buffer(cl_command_queue queue, cl_mem xy_out_buf, float *xy_out, size_t out_count);

cl_int pl_enqueue_projection_kernel_points(PLContext *pl_ctx, cl_kernel kernel,
        PLProjection proj, PLProjectionParams *params, PLProjectionBuffer *pl_buf);
cl_int pl_enqueue_projection_kernel_grid(PLContext *pl_ctx, cl_kernel kernel,
        PLProjection proj, PLProjectionParams *params, PLPointGridBuffer *src, PLPointGridBuffer *dst);

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
