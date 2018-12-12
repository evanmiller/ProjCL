//
//  projcl_warp.c
//  Magic Maps
//
//  Created by Evan Miller on 9/19/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#include <projcl/projcl.h>
#include <projcl/projcl_warp.h>
#include "projcl_util.h"
#include "projcl_run.h"
#include "projcl_kernel.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void pl_swap_grid_buffers(PLPointGridBuffer *grid);

PLImageArrayBuffer *pl_load_image_array(PLContext *pl_ctx, cl_channel_order channel_order, cl_channel_type channel_type,
        size_t width, size_t height, size_t row_pitch, 
        size_t slice_pitch, size_t tiles_across, size_t tiles_down,
        const void *pvData, cl_bool copy, cl_int *outError) {
    cl_int error = CL_SUCCESS;
    
    PLImageArrayBuffer *buf;
    if ((buf = malloc(sizeof(PLImageArrayBuffer))) == NULL) {
        if (outError != NULL)
            *outError = CL_OUT_OF_HOST_MEMORY;
        return NULL;
    }
    
    buf->tiles_across = tiles_across;
    buf->tiles_down = tiles_down;

    cl_image_format image_format = {0};
    image_format.image_channel_data_type = channel_type;
    image_format.image_channel_order = channel_order;

    cl_image_desc image_desc = {0};
    image_desc.image_type = CL_MEM_OBJECT_IMAGE3D;
    image_desc.image_width = width;
    image_desc.image_height = height;
    image_desc.image_depth = tiles_across * tiles_down;
    image_desc.image_row_pitch = row_pitch;
    image_desc.image_slice_pitch = slice_pitch;

    buf->image_format = image_format;
    buf->image_desc = image_desc;
    buf->image = clCreateImage(pl_ctx->ctx, CL_MEM_READ_ONLY | (copy ? CL_MEM_COPY_HOST_PTR : CL_MEM_USE_HOST_PTR),
            &buf->image_format, &buf->image_desc, (void *)pvData, &error);
    
    if (buf->image == NULL) {
        free(buf);
        if (outError) {
            *outError = error;
        }
        return NULL;
    }
    
    if (outError)
        *outError = error;
    
    return buf;

}

PLImageBuffer *pl_load_image(PLContext *pl_ctx, cl_channel_order channel_order, cl_channel_type channel_type,
        size_t width, size_t height, size_t row_pitch,
        const void* pvData, cl_bool copy, cl_int *outError) {
    cl_int error = CL_SUCCESS;
    
    PLImageBuffer *buf;
    if ((buf = malloc(sizeof(PLImageBuffer))) == NULL) {
        if (outError != NULL)
            *outError = CL_OUT_OF_HOST_MEMORY;
        return NULL;
    }
    
    cl_image_format image_format = {0};
    image_format.image_channel_data_type = channel_type;
    image_format.image_channel_order = channel_order;

    cl_image_desc image_desc = {0};
    image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    image_desc.image_width = width;
    image_desc.image_height = height;
    image_desc.image_row_pitch =  row_pitch;

    buf->image_format = image_format;
    buf->image_desc = image_desc;
    buf->image = clCreateImage(pl_ctx->ctx, CL_MEM_READ_ONLY | (copy ? CL_MEM_COPY_HOST_PTR : CL_MEM_USE_HOST_PTR),
            &buf->image_format, &buf->image_desc, (void *)pvData, &error);
    
    if (buf->image == NULL) {
        free(buf);
        if (outError) {
            *outError = error;
        }
        return NULL;
    }
    
    if (outError)
        *outError = error;
    
    return buf;
}

PLPointGridBuffer *pl_load_empty_grid(PLContext *pl_ctx, size_t count_x, size_t count_y, int *outError) {
    cl_int error = CL_SUCCESS;
    PLPointGridBuffer *grid;
    if ((grid = malloc(sizeof(PLPointGridBuffer))) == NULL) {
        if (outError)
            *outError = CL_OUT_OF_HOST_MEMORY;
        return NULL;
    }
    grid->grid = NULL;
    grid->width = count_x;
    grid->height = count_y;
    
    size_t count = ck_padding(count_x * count_y, 8);
    
    grid->grid = clCreateBuffer(pl_ctx->ctx, CL_MEM_READ_WRITE, 2 * sizeof(cl_float) * count, NULL, &error);
    if (grid->grid == NULL) {
        pl_unload_grid(grid);
        if (outError)
            *outError = error;
        return NULL;
    }
    return grid;
}

PLPointGridBuffer *pl_load_grid(PLContext *pl_ctx,
        double origin_x, double width, size_t count_x,
        double origin_y, double height, size_t count_y,
        int *outError) {
    int error = CL_SUCCESS;
    PLPointGridBuffer *grid = pl_load_empty_grid(pl_ctx, count_x, count_y, &error);
    if (grid == NULL) {
        if (outError)
            *outError = error;
        return NULL;
    }
    cl_kernel load_grid_kernel = pl_find_kernel(pl_ctx, "pl_load_grid");
    if (load_grid_kernel == NULL) {
        pl_unload_grid(grid);
        if (outError)
            *outError = CL_INVALID_KERNEL_NAME;
        return NULL;
    }
    
    cl_uint argc = 0;
    
    double origin[2] = { origin_x, origin_y };
    double size[2] = { width, height };
    
    error |= pl_set_kernel_arg_mem(pl_ctx, load_grid_kernel, argc++, grid->grid);
    error |= pl_set_kernel_arg_float2(pl_ctx, load_grid_kernel, argc++, origin);
    error |= pl_set_kernel_arg_float2(pl_ctx, load_grid_kernel, argc++, size);
    
    if (error != CL_SUCCESS) {
        pl_unload_grid(grid);
        if (outError)
            *outError = error;
        return NULL;
    }
    
    size_t dim_count[2] = { count_y, count_x };
    
    error = clEnqueueNDRangeKernel(pl_ctx->queue, load_grid_kernel, 2, NULL, dim_count, NULL, 0, NULL, NULL);
    if (error != CL_SUCCESS) {
        pl_unload_grid(grid);
        if (outError)
            *outError = error;
        return NULL;
    }
    
    return grid;
}

cl_int pl_transform_grid(PLContext *pl_ctx, PLPointGridBuffer *src, PLPointGridBuffer *dst,
        double sx, double sy, double tx, double ty) {
    double matrix[8] = { 
        sx, 0.0, tx,
        0.0, sy, ty,
        0.0, 0.0 };
    
    cl_kernel transform_kernel = pl_find_kernel(pl_ctx, "pl_cartesian_apply_affine_transform_2d");
    if (transform_kernel == NULL) {
        return CL_INVALID_KERNEL_NAME;
    }
    
    cl_int error = CL_SUCCESS;
    
    cl_uint argc = 0;
    
    error |= pl_set_kernel_arg_mem(pl_ctx, transform_kernel, argc++, src->grid);
    error |= pl_set_kernel_arg_mem(pl_ctx, transform_kernel, argc++, dst->grid);
    error |= pl_set_kernel_arg_float8(pl_ctx, transform_kernel, argc++, matrix);
    
    if (error != CL_SUCCESS) {
        return error;
    }
        
    size_t vec_count = ck_padding(src->height * src->width, 8) / 8;
    
    cl_int retval = clEnqueueNDRangeKernel(pl_ctx->queue, transform_kernel, 1, NULL, &vec_count, 
                                           NULL, 0, NULL, NULL);
        
    return retval;
}

cl_int pl_shift_grid_datum(PLContext *pl_ctx, PLPointGridBuffer *src, PLDatum src_datum, PLSpheroid src_spheroid,
                           PLPointGridBuffer *dst, PLDatum dst_datum, PLSpheroid dst_spheroid) {    
    cl_int error = CL_SUCCESS;
    
    int xy_pad_count = ck_padding(src->height * src->width, PL_FLOAT_VECTOR_SIZE);
    
    cl_mem x_rw = NULL, y_rw = NULL, z_rw = NULL;
    PLDatumShiftBuffer *pl_buf = NULL;

    cl_kernel cartesian_kernel = pl_find_kernel(pl_ctx, "pl_geodesic_to_cartesian");
    cl_kernel transform_kernel = pl_find_kernel(pl_ctx, "pl_cartesian_apply_affine_transform");
    cl_kernel geodesic_kernel = pl_find_kernel(pl_ctx, "pl_cartesian_to_geodesic");
    
    if (cartesian_kernel == NULL || geodesic_kernel == NULL || transform_kernel == NULL)
        return CL_INVALID_KERNEL_NAME;

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
    
    pl_buf = calloc(1,  sizeof(PLDatumShiftBuffer));
    pl_buf->count = src->height * src->width;
    pl_buf->xy_in = src->grid;
    pl_buf->xy_out = dst->grid;
    pl_buf->x_rw = x_rw;
    pl_buf->y_rw = y_rw;
    pl_buf->z_rw = z_rw;
    
    error = pl_run_kernel_geodesic_to_cartesian(pl_ctx, cartesian_kernel, pl_buf, src_spheroid);
    if (error != CL_SUCCESS)
        goto cleanup;
    
    error = pl_run_kernel_transform_cartesian(pl_ctx, transform_kernel, pl_buf, src_datum, dst_datum);
    if (error != CL_SUCCESS)
        goto cleanup;
    
    error = pl_run_kernel_cartesian_to_geodesic(pl_ctx, geodesic_kernel, pl_buf, NULL, dst_spheroid);
    if (error != CL_SUCCESS)
        goto cleanup;
    
cleanup:
    if (x_rw)
        clReleaseMemObject(x_rw);
    if (y_rw)
        clReleaseMemObject(y_rw);
    if (z_rw)
        clReleaseMemObject(z_rw);
    if (pl_buf)
        free(pl_buf);
        
    return error;
}

static cl_int _pl_project_grid(PLContext *pl_ctx, PLProjection proj, PLProjectionParams *params,
        PLPointGridBuffer *src, PLPointGridBuffer *dst, int fwd) {
    cl_kernel kernel = NULL;
    cl_int error = CL_SUCCESS;

    if (proj == PL_PROJECT_LAMBERT_CONFORMAL_CONIC && fabs((params->rlat1 + params->rlat2) * DEG_TO_RAD) < 1.e-7) {
        /* With symmetrical standard parallels the LCC equations break down.
         * But, in this case it reduces to a Mercator projection with an appropriate shift. */
        PLProjectionParams *params2 = pl_params_init();
        pl_params_set_mercator_params_from_pathological_lambert_conformal_conic_params(params2, params);

        error = _pl_project_grid(pl_ctx, PL_PROJECT_MERCATOR, params2, src, dst, fwd);

        pl_params_free(params2);
        return error;
    }

    kernel = pl_find_projection_kernel(pl_ctx, proj, fwd, params->spheroid);
    if (kernel == NULL) {
        return CL_INVALID_KERNEL_NAME;
    }
    
    error = pl_enqueue_projection_kernel_grid(pl_ctx, kernel, proj, params, src, dst);

    return error;
}

cl_int pl_project_grid_forward(PLContext *pl_ctx, PLProjection proj, PLProjectionParams *params,
        PLPointGridBuffer *src, PLPointGridBuffer *dst) {
    return _pl_project_grid(pl_ctx, proj, params, src, dst, 1);
}

cl_int pl_project_grid_reverse(PLContext *pl_ctx, PLProjection proj, PLProjectionParams *params,
        PLPointGridBuffer *src, PLPointGridBuffer *dst) {
    return _pl_project_grid(pl_ctx, proj, params, src, dst, 0);
}

cl_int pl_sample_image(PLContext *pl_ctx, PLPointGridBuffer *grid, PLImageBuffer *buf, PLImageFilter filter,
                              unsigned char *outData) {
    cl_int error = CL_SUCCESS;

    cl_image_desc image_desc = {0};
    image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    image_desc.image_width = grid->width;
    image_desc.image_height = grid->height;

    cl_mem out_image = clCreateImage(pl_ctx->ctx, CL_MEM_WRITE_ONLY, &buf->image_format, 
            &image_desc, NULL, &error);
    if (out_image == NULL) {
        return error;
    }
    
    cl_kernel kernel = NULL;

    if (filter == PL_IMAGE_FILTER_NEAREST_NEIGHBOR) {
        kernel = pl_find_kernel(pl_ctx, "pl_sample_image_nearest");
    } else if (filter == PL_IMAGE_FILTER_BILINEAR) {
        kernel = pl_find_kernel(pl_ctx, "pl_sample_image_linear");
    } else if (filter == PL_IMAGE_FILTER_BICUBIC) {
        kernel = pl_find_kernel(pl_ctx, "pl_sample_image_bicubic");
    } else if (filter == PL_IMAGE_FILTER_QUASI_BICUBIC) {
        kernel = pl_find_kernel(pl_ctx, "pl_sample_image_quasi_bicubic");
    }
    if (kernel == NULL) {
        clReleaseMemObject(out_image);
        return CL_INVALID_KERNEL_NAME;
    }
    
    cl_uint argc = 0;
    error |= pl_set_kernel_arg_mem(pl_ctx, kernel, argc++, grid->grid);
    error |= pl_set_kernel_arg_mem(pl_ctx, kernel, argc++, buf->image);
    error |= pl_set_kernel_arg_mem(pl_ctx, kernel, argc++, out_image);
    
    if (error != CL_SUCCESS) {
        clReleaseMemObject(out_image);
        return error;
    }
    
    size_t dim_count[2] = { grid->height, grid->width };
    
    error = clEnqueueNDRangeKernel(pl_ctx->queue, kernel, 2, NULL, dim_count, NULL, 0, NULL, NULL);
    if (error != CL_SUCCESS) {
        clReleaseMemObject(out_image);
        return error;
    }
    
    size_t origin[3] = { 0, 0, 0 };
    size_t region[3] = { grid->width, grid->height, 1 };
    
    error = clEnqueueReadImage(pl_ctx->queue, out_image, CL_TRUE, origin, region, 0, 0, outData, 0, NULL, NULL);
    if (error != CL_SUCCESS) {
        clReleaseMemObject(out_image);
        return error;
    }
    
    error = clFinish(pl_ctx->queue);
    clReleaseMemObject(out_image);
    return error;
}

cl_int pl_sample_image_array(PLContext *pl_ctx, PLPointGridBuffer *grid, PLImageArrayBuffer *buf, PLImageFilter filter,
                                    unsigned char *outData) {
    cl_int error = CL_SUCCESS;

    cl_image_desc image_desc = {0};
    image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    image_desc.image_width = grid->width;
    image_desc.image_height = grid->height;

    cl_mem out_image = clCreateImage(pl_ctx->ctx, CL_MEM_WRITE_ONLY, &buf->image_format, &image_desc, NULL, &error);
    if (out_image == NULL) {
        return error;
    }
    
    cl_kernel kernel = NULL;
    if (filter == PL_IMAGE_FILTER_NEAREST_NEIGHBOR) {
        kernel = pl_find_kernel(pl_ctx, "pl_sample_image_array_nearest");
    } else if (filter == PL_IMAGE_FILTER_BILINEAR) {
        kernel = pl_find_kernel(pl_ctx, "pl_sample_image_array_linear");
    } else if (filter == PL_IMAGE_FILTER_BICUBIC) {
        kernel = pl_find_kernel(pl_ctx, "pl_sample_image_array_bicubic");
    } else if (filter == PL_IMAGE_FILTER_QUASI_BICUBIC) {
        kernel = pl_find_kernel(pl_ctx, "pl_sample_image_array_quasi_bicubic");
    }
    if (kernel == NULL) {
        clReleaseMemObject(out_image);
        return CL_INVALID_KERNEL_NAME;
    }
    
    cl_uint argc = 0;
    error |= pl_set_kernel_arg_mem(pl_ctx, kernel, argc++, grid->grid);
    error |= pl_set_kernel_arg_mem(pl_ctx, kernel, argc++, buf->image);
    error |= pl_set_kernel_arg_uint(pl_ctx, kernel, argc++, buf->tiles_across);
    error |= pl_set_kernel_arg_mem(pl_ctx, kernel, argc++, out_image);
    
    if (error != CL_SUCCESS) {
        clReleaseMemObject(out_image);
        return error;
    }
    
    size_t dim_count[2] = { grid->height, grid->width };
    
    error = clEnqueueNDRangeKernel(pl_ctx->queue, kernel, 2, NULL, dim_count, NULL, 0, NULL, NULL);
    if (error != CL_SUCCESS) {
        clReleaseMemObject(out_image);
        return error;
    }
    
    size_t origin[3] = { 0, 0, 0 };
    size_t region[3] = { grid->width, grid->height, 1 };
    
    error = clEnqueueReadImage(pl_ctx->queue, out_image, CL_TRUE, origin, region, 0, 0, outData, 0, NULL, NULL);
    if (error != CL_SUCCESS) {
        clReleaseMemObject(out_image);
        return error;
    }
    
    error = clFinish(pl_ctx->queue);
    clReleaseMemObject(out_image);
    return error;
}

void pl_unload_grid(PLPointGridBuffer *grid) {
    if (grid->grid)
        clReleaseMemObject(grid->grid);
    free(grid);
}

void pl_unload_image(PLImageBuffer *buf) {
    clReleaseMemObject(buf->image);
    free(buf);
}

void pl_unload_image_array(PLImageArrayBuffer *buf) {
    clReleaseMemObject(buf->image);
    free(buf);
}
