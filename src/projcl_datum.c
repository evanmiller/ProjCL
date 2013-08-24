//
//  projcl_datum.c
//  Magic Maps
//
//  Created by Evan Miller on 3/31/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#include <projcl/projcl.h>
#include <stdlib.h>
#include "projcl_run.h"
#include "projcl_util.h"

PLDatumShiftBuffer *pl_load_datum_shift_data(PLContext *pl_ctx, PLSpheroid src_spheroid, 
                                             const float *xy, int n, cl_int *outError) {
    PLDatumShiftBuffer *pl_buf = NULL;
    
    cl_int error = CL_SUCCESS;
    
    float *xy_pad_in = NULL;
    int xy_pad_count = ck_padding(n, PL_FLOAT_VECTOR_SIZE);
    
    cl_mem xy_in = NULL, x_rw = NULL, y_rw = NULL, z_rw = NULL;
    
    if ((xy_pad_in = malloc(xy_pad_count * sizeof(float) * 2)) == NULL) {
        error = CL_OUT_OF_HOST_MEMORY;
        goto cleanup;
    }
    
    _pl_copy_pad(xy_pad_in, xy_pad_count * 2, xy, n * 2);
        
    xy_in = clCreateBuffer(pl_ctx->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(cl_float) * xy_pad_count * 2, xy_pad_in, &error);
    if (error != CL_SUCCESS) {
        goto cleanup;
    }
    
    x_rw = clCreateBuffer(pl_ctx->ctx, CL_MEM_READ_WRITE, 
                            sizeof(cl_float) * xy_pad_count, NULL, &error);
    if (error != CL_SUCCESS) {
        goto cleanup;
    }
    y_rw = clCreateBuffer(pl_ctx->ctx, CL_MEM_READ_WRITE, 
                          sizeof(cl_float) * xy_pad_count, NULL, &error);
    if (error != CL_SUCCESS) {
        goto cleanup;
    }
    z_rw = clCreateBuffer(pl_ctx->ctx, CL_MEM_READ_WRITE, 
                          sizeof(cl_float) * xy_pad_count, NULL, &error);
    if (error != CL_SUCCESS) {
        goto cleanup;
    }
    
    cl_kernel cartesian_kernel = _pl_find_kernel(pl_ctx, "pl_geodesic_to_cartesian");
    if (cartesian_kernel == NULL) {
        error = CL_INVALID_KERNEL_NAME;
        goto cleanup;
    }
    if ((pl_buf = malloc(sizeof(PLDatumShiftBuffer))) == NULL) {
        error = CL_OUT_OF_HOST_MEMORY;
        goto cleanup;
    }
    pl_buf->count = n;
    pl_buf->xy_in = xy_in;
    pl_buf->xy_out = NULL;
    pl_buf->x_rw = x_rw;
    pl_buf->y_rw = y_rw;
    pl_buf->z_rw = z_rw;
    
    error = pl_run_kernel_geodesic_to_cartesian(cartesian_kernel, pl_ctx, pl_buf, src_spheroid);
    
    pl_buf->xy_out = clCreateBuffer(pl_ctx->ctx, CL_MEM_WRITE_ONLY, 
                                    sizeof(cl_float) * xy_pad_count * 2, NULL, &error);
    if (error != CL_SUCCESS)
        goto cleanup;
    
cleanup:
    if (xy_pad_in)
        free(xy_pad_in);
    if (xy_in)
        clReleaseMemObject(xy_in);
    if (error != CL_SUCCESS) {
        if (x_rw)
            clReleaseMemObject(x_rw);
        if (y_rw)
            clReleaseMemObject(y_rw);
        if (z_rw)
            clReleaseMemObject(z_rw);
        if (pl_buf) {
            if (pl_buf->xy_out)
                clReleaseMemObject(pl_buf->xy_out);
            free(pl_buf);
        }
        if (outError != NULL)
            *outError = error;
        return NULL;
    }
    
    return pl_buf;
}

void pl_unload_datum_shift_data(PLDatumShiftBuffer *pl_buf) {
    if (pl_buf == NULL)
        return;
    
    clReleaseMemObject(pl_buf->x_rw);
    clReleaseMemObject(pl_buf->y_rw);
    clReleaseMemObject(pl_buf->z_rw);
    clReleaseMemObject(pl_buf->xy_out);
    free(pl_buf);
}

cl_int pl_shift_datum(PLContext *pl_ctx, PLDatum src_datum, PLDatum dst_datum, PLSpheroid dst_spheroid,
                      PLDatumShiftBuffer *pl_buf, float *xy_out) {
    cl_int error = CL_SUCCESS;
    cl_kernel transform_kernel = _pl_find_kernel(pl_ctx, "pl_cartesian_apply_affine_transform");
    cl_kernel geodesic_kernel = _pl_find_kernel(pl_ctx, "pl_cartesian_to_geodesic");
    
    if (geodesic_kernel == NULL || transform_kernel == NULL)
        return CL_INVALID_KERNEL_NAME;
    
    error = pl_run_kernel_transform_cartesian(transform_kernel, pl_ctx, pl_buf, src_datum, dst_datum);
    if (error != CL_SUCCESS)
        return error;
    
    error = pl_run_kernel_cartesian_to_geodesic(geodesic_kernel, pl_ctx, pl_buf, xy_out, dst_spheroid);
    if (error != CL_SUCCESS)
        return error;
    
    return CL_SUCCESS;
}

