//
//  projcl_project.c
//  Magic Maps
//
//  Created by Evan Miller on 3/31/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#import <projcl/projcl.h>
#include "projcl_run.h"
#include "projcl_util.h"
#include <stdlib.h>

PLProjectionBuffer *pl_load_projection_data(PLContext *pl_ctx, const float *xy, int count, int copy, int *outError) {
	float *xy_pad = NULL;
	PLProjectionBuffer *pl_buf = NULL;
	
	int xy_pad_count = ck_padding(count, PL_FLOAT_VECTOR_SIZE);
    
	cl_int error = CL_SUCCESS;
	
    int needs_free = 0;
    if (xy_pad_count == count) {
        xy_pad = (float *)xy;
    } else {
        if ((xy_pad = malloc(xy_pad_count * sizeof(float) * 2)) == NULL) {
            error = CL_OUT_OF_HOST_MEMORY;
            goto cleanup;
        }
        _pl_copy_pad(xy_pad, xy_pad_count * 2, xy, count * 2);
        copy = 1;
        needs_free = 1;
    }
	if ((pl_buf = malloc(sizeof(PLProjectionBuffer))) == NULL) {
		error = CL_OUT_OF_HOST_MEMORY;
		goto cleanup;
	}
	
	pl_buf->xy_in = clCreateBuffer(pl_ctx->ctx, CL_MEM_READ_ONLY | (copy ? CL_MEM_COPY_HOST_PTR : CL_MEM_USE_HOST_PTR), 
								   sizeof(cl_float) * xy_pad_count * 2, xy_pad, &error);
	if (error != CL_SUCCESS) {
		goto cleanup;
	}
	
	pl_buf->xy_out = clCreateBuffer(pl_ctx->ctx, CL_MEM_WRITE_ONLY, sizeof(cl_float) * count * 2, NULL, &error);
	if (error != CL_SUCCESS) {
		goto cleanup;
	}
	
	pl_buf->count = count;
	
cleanup:
	if (xy_pad && needs_free)
		free(xy_pad);
	
	if (error != CL_SUCCESS) {
		if (pl_buf && pl_buf->xy_in)
			clReleaseMemObject(pl_buf->xy_in);
		free(pl_buf);
		if (outError != NULL)
			*outError = error;
		return NULL;
	}
	
	if (outError != NULL)
		*outError = CL_SUCCESS;
	
	return pl_buf;
}

void pl_unload_projection_data(PLProjectionBuffer *pl_buf) {
	if (pl_buf == NULL)
		return;
	
	clReleaseMemObject(pl_buf->xy_in);
	clReleaseMemObject(pl_buf->xy_out);
	free(pl_buf);
}

cl_int pl_project_albers_equal_area(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
									PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0,
                                    float rlat1, float rlat2) {
	cl_kernel kernel = _pl_find_projection_kernel(pl_ctx, "albers_equal_area", 1, pl_ell);
	if (kernel == NULL)
		return CL_INVALID_KERNEL_NAME;
	
	cl_int error = pl_enqueue_kernel_albers_equal_area(kernel, pl_ctx, pl_buf->xy_in, pl_buf->xy_out, pl_buf->count,
                                                       pl_ell, scale, x0, y0, lon0, lat0, rlat1, rlat2);
    if (error != CL_SUCCESS)
        return error;
    
    return pl_read_buffer(pl_ctx->queue, pl_buf->xy_out, xy_out, 2 * pl_buf->count * sizeof(cl_float));
}

cl_int pl_unproject_albers_equal_area(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
									  PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0, float rlat1, float rlat2) {
	cl_kernel kernel = _pl_find_projection_kernel(pl_ctx, "albers_equal_area", 0, pl_ell);
	if (kernel == NULL)
		return CL_INVALID_KERNEL_NAME;
	
	cl_int error = pl_enqueue_kernel_albers_equal_area(kernel, pl_ctx, pl_buf->xy_in, pl_buf->xy_out, pl_buf->count,
                                                       pl_ell, scale, x0, y0, lon0, lat0, rlat1, rlat2);
    if (error != CL_SUCCESS)
        return error;
    
    return pl_read_buffer(pl_ctx->queue, pl_buf->xy_out, xy_out, 2 * pl_buf->count * sizeof(cl_float));
}

cl_int pl_project_american_polyconic(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
                                     PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0) {
	cl_kernel kernel = _pl_find_projection_kernel(pl_ctx, "american_polyconic", 1, pl_ell);
	if (kernel == NULL)
		return CL_INVALID_KERNEL_NAME;
	
	cl_int error = pl_enqueue_kernel_american_polyconic(kernel, pl_ctx, pl_buf->xy_in, pl_buf->xy_out, pl_buf->count,
                                                        pl_ell, scale, x0, y0, lon0, lat0);
    if (error != CL_SUCCESS)
        return error;
    
    return pl_read_buffer(pl_ctx->queue, pl_buf->xy_out, xy_out, 2 * pl_buf->count * sizeof(cl_float));
}

cl_int pl_unproject_american_polyconic(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
                                       PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0) {
	cl_kernel kernel = _pl_find_projection_kernel(pl_ctx, "american_polyconic", 0, pl_ell);
	if (kernel == NULL)
		return CL_INVALID_KERNEL_NAME;
	
	cl_int error = pl_enqueue_kernel_american_polyconic(kernel, pl_ctx, pl_buf->xy_in, pl_buf->xy_out, pl_buf->count,
                                                    pl_ell, scale, x0, y0, lon0, lat0);
    if (error != CL_SUCCESS)
        return error;
    
    return pl_read_buffer(pl_ctx->queue, pl_buf->xy_out, xy_out, 2 * pl_buf->count * sizeof(cl_float));
}

cl_int pl_project_lambert_azimuthal_equal_area(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
                                               PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0) {
    cl_kernel kernel = _pl_find_projection_kernel(pl_ctx, "lambert_azimuthal_equal_area", 1, pl_ell);
    if (kernel == NULL)
        return CL_INVALID_KERNEL_NAME;

    cl_int error = pl_enqueue_kernel_lambert_azimuthal_equal_area(kernel, pl_ctx, pl_buf->xy_in, pl_buf->xy_out, pl_buf->count,
                                                                  pl_ell, scale, x0, y0, lon0, lat0);
    if (error != CL_SUCCESS)
        return error;
    
    return pl_read_buffer(pl_ctx->queue, pl_buf->xy_out, xy_out, 2 * pl_buf->count * sizeof(cl_float));
}

cl_int pl_unproject_lambert_azimuthal_equal_area(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
                                                 PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0) {
    cl_kernel kernel = _pl_find_projection_kernel(pl_ctx, "lambert_azimuthal_equal_area", 0, pl_ell);
    if (kernel == NULL)
        return CL_INVALID_KERNEL_NAME;
    
    cl_int error = pl_enqueue_kernel_lambert_azimuthal_equal_area(kernel, pl_ctx, pl_buf->xy_in, pl_buf->xy_out, pl_buf->count,
                                                              pl_ell, scale, x0, y0, lon0, lat0);
    if (error != CL_SUCCESS)
        return error;
    
    return pl_read_buffer(pl_ctx->queue, pl_buf->xy_out, xy_out, 2 * pl_buf->count * sizeof(cl_float));
}

cl_int pl_project_lambert_conformal_conic(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
                                          PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0,
                                          float rlat1, float rlat2) {
    cl_kernel kernel = _pl_find_projection_kernel(pl_ctx, "lambert_conformal_conic", 1, pl_ell);
	if (kernel == NULL)
		return CL_INVALID_KERNEL_NAME;
    
    cl_int error = pl_enqueue_kernel_lambert_conformal_conic(kernel, pl_ctx, pl_buf->xy_in, pl_buf->xy_out, pl_buf->count,
                                                         pl_ell, scale, x0, y0, lon0, lat0, rlat1, rlat2);
    if (error != CL_SUCCESS)
        return error;
    
    return pl_read_buffer(pl_ctx->queue, pl_buf->xy_out, xy_out, 2 * pl_buf->count * sizeof(cl_float));
}

cl_int pl_unproject_lambert_conformal_conic(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
                                            PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0,
                                            float rlat1, float rlat2) {
    cl_kernel kernel = _pl_find_projection_kernel(pl_ctx, "lambert_conformal_conic", 0, pl_ell);
	if (kernel == NULL)
		return CL_INVALID_KERNEL_NAME;
    
    cl_int error = pl_enqueue_kernel_lambert_conformal_conic(kernel, pl_ctx, pl_buf->xy_in, pl_buf->xy_out, pl_buf->count,
                                                         pl_ell, scale, x0, y0, lon0, lat0, rlat1, rlat2);
    if (error != CL_SUCCESS)
        return error;
    
    return pl_read_buffer(pl_ctx->queue, pl_buf->xy_out, xy_out, 2 * pl_buf->count * sizeof(cl_float));
}

cl_int pl_project_mercator(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
						   PLSpheroid pl_ell, float scale, float x0, float y0) {
	cl_kernel kernel = _pl_find_projection_kernel(pl_ctx, "mercator", 1, pl_ell);
	if (kernel == NULL)
		return CL_INVALID_KERNEL_NAME;
	
	cl_int error = pl_enqueue_kernel_mercator(kernel, pl_ctx, pl_buf->xy_in, pl_buf->xy_out, pl_buf->count,
                                          pl_ell, scale, x0, y0);
    if (error != CL_SUCCESS)
        return error;
    
    return pl_read_buffer(pl_ctx->queue, pl_buf->xy_out, xy_out, 2 * pl_buf->count * sizeof(cl_float));
}

cl_int pl_unproject_mercator(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
							 PLSpheroid pl_ell, float scale, float x0, float y0) {
	cl_kernel kernel = _pl_find_projection_kernel(pl_ctx, "mercator", 0, pl_ell);
	if (kernel == NULL)
		return CL_INVALID_KERNEL_NAME;
	
	cl_int error = pl_enqueue_kernel_mercator(kernel, pl_ctx, pl_buf->xy_in, pl_buf->xy_out, pl_buf->count,
                                          pl_ell, scale, x0, y0);
    if (error != CL_SUCCESS)
        return error;
    
    return pl_read_buffer(pl_ctx->queue, pl_buf->xy_out, xy_out, 2 * pl_buf->count * sizeof(cl_float));
}

cl_int pl_project_robinson(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
        float scale, float x0, float y0) {
	cl_kernel kernel = _pl_find_projection_kernel(pl_ctx, "robinson", 1, PL_SPHEROID_SPHERE);
	if (kernel == NULL)
		return CL_INVALID_KERNEL_NAME;
	
	cl_int error = pl_enqueue_kernel_robinson(kernel, pl_ctx, pl_buf->xy_in, pl_buf->xy_out, pl_buf->count,
            scale, x0, y0);
    if (error != CL_SUCCESS)
        return error;
    
    return pl_read_buffer(pl_ctx->queue, pl_buf->xy_out, xy_out, 2 * pl_buf->count * sizeof(cl_float));
}

cl_int pl_unproject_robinson(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
        float scale, float x0, float y0) {
	cl_kernel kernel = _pl_find_projection_kernel(pl_ctx, "robinson", 0, PL_SPHEROID_SPHERE);
	if (kernel == NULL)
		return CL_INVALID_KERNEL_NAME;
	
	cl_int error = pl_enqueue_kernel_robinson(kernel, pl_ctx, pl_buf->xy_in, pl_buf->xy_out, pl_buf->count,
            scale, x0, y0);
    if (error != CL_SUCCESS)
        return error;
    
    return pl_read_buffer(pl_ctx->queue, pl_buf->xy_out, xy_out, 2 * pl_buf->count * sizeof(cl_float));
}

cl_int pl_project_transverse_mercator(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out, 
                                      PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0) {
    cl_kernel kernel = _pl_find_projection_kernel(pl_ctx, "transverse_mercator", 1, pl_ell);
    if (kernel == NULL)
        return CL_INVALID_KERNEL_NAME;
    
    cl_int error = pl_enqueue_kernel_transverse_mercator(kernel, pl_ctx, pl_buf->xy_in, pl_buf->xy_out, pl_buf->count,
                                                     pl_ell, scale, x0, y0, lon0, lat0);
    if (error != CL_SUCCESS)
        return error;
    
    return pl_read_buffer(pl_ctx->queue, pl_buf->xy_out, xy_out, 2 * pl_buf->count * sizeof(cl_float));
}

cl_int pl_unproject_transverse_mercator(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out, 
                                        PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0) {
    cl_kernel kernel = _pl_find_projection_kernel(pl_ctx, "transverse_mercator", 0, pl_ell);
    if (kernel == NULL)
        return CL_INVALID_KERNEL_NAME;
    
    cl_int error = pl_enqueue_kernel_transverse_mercator(kernel, pl_ctx, pl_buf->xy_in, pl_buf->xy_out, pl_buf->count,
                                                     pl_ell, scale, x0, y0, lon0, lat0);
    if (error != CL_SUCCESS)
        return error;
    
    return pl_read_buffer(pl_ctx->queue, pl_buf->xy_out, xy_out, 2 * pl_buf->count * sizeof(cl_float));
}

cl_int pl_project_winkel_tripel(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
                                float scale, float x0, float y0, float lon0, float rlat1) {
	cl_kernel kernel = _pl_find_projection_kernel(pl_ctx, "winkel_tripel", 1, PL_SPHEROID_SPHERE);
	if (kernel == NULL)
		return CL_INVALID_KERNEL_NAME;
	
	cl_int error = pl_enqueue_kernel_winkel_tripel(kernel, pl_ctx, pl_buf->xy_in, pl_buf->xy_out, pl_buf->count,
                                                   scale, x0, y0, lon0, rlat1);
    if (error != CL_SUCCESS)
        return error;
    
    return pl_read_buffer(pl_ctx->queue, pl_buf->xy_out, xy_out, 2 * pl_buf->count * sizeof(cl_float));
}

cl_int pl_unproject_winkel_tripel(PLContext *pl_ctx, PLProjectionBuffer *pl_buf, float *xy_out,
                                  float scale, float x0, float y0, float lon0, float rlat1) {
	cl_kernel kernel = _pl_find_projection_kernel(pl_ctx, "winkel_tripel", 0, PL_SPHEROID_SPHERE);
	if (kernel == NULL)
		return CL_INVALID_KERNEL_NAME;
	
	cl_int error = pl_enqueue_kernel_winkel_tripel(kernel, pl_ctx, pl_buf->xy_in, pl_buf->xy_out, pl_buf->count,
                                                   scale, x0, y0, lon0, rlat1);
    if (error != CL_SUCCESS)
        return error;
    
    return pl_read_buffer(pl_ctx->queue, pl_buf->xy_out, xy_out, 2 * pl_buf->count * sizeof(cl_float));
}
