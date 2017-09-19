//
//  projcl_project.c
//  Magic Maps
//
//  Created by Evan Miller on 3/31/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#include <projcl/projcl.h>
#include "projcl_run.h"
#include "projcl_util.h"
#include "projcl_spheroid.h"
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

PLProjectionBuffer *pl_load_projection_data(PLContext *pl_ctx, const float *xy, int count, int copy, int *outError) {
	float *xy_pad = NULL;
	int xy_pad_count = ck_padding(count, PL_FLOAT_VECTOR_SIZE);

	PLProjectionBuffer *pl_buf = NULL;
	cl_int error = CL_SUCCESS;
	
    int needs_free = 0;

    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    pl_ctx->last_time = NAN;

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

    gettimeofday(&end_time, NULL);

    pl_ctx->last_time = (end_time.tv_sec + end_time.tv_usec * 1e-6)
        - (start_time.tv_sec + start_time.tv_usec * 1e-6);
	
	return pl_buf;
}

void pl_unload_projection_data(PLProjectionBuffer *pl_buf) {
	if (pl_buf == NULL)
		return;
	
	clReleaseMemObject(pl_buf->xy_in);
	clReleaseMemObject(pl_buf->xy_out);
	free(pl_buf);
}

static cl_int _pl_project(PLContext *pl_ctx, PLProjection proj, PLProjectionParams *params,
        PLProjectionBuffer *pl_buf, float *xy_out, int fwd) {
    struct timeval start_time, end_time;
    const char *name = _pl_proj_name(proj);
    cl_kernel kernel = NULL;
    cl_int error = CL_SUCCESS;

    if (name == NULL)
        return CL_INVALID_KERNEL_NAME;

    if (proj == PL_PROJECT_LAMBERT_CONFORMAL_CONIC && fabs((params->rlat1 + params->rlat2) * DEG_TO_RAD) < 1.e-7) {
        /* With symmetrical standard parallels the LCC equations break down.
         * But, in this case it reduces to a Mercator projection with an appropriate shift. */
        PLProjectionParams *params2 = pl_params_init();
        pl_params_set_mercator_params_from_pathological_lambert_conformal_conic_params(params2, params);

        error = _pl_project(pl_ctx, PL_PROJECT_MERCATOR, params2, pl_buf, xy_out, fwd);

        pl_params_free(params2);
        return error;
    }

	kernel = _pl_find_projection_kernel(pl_ctx, name, fwd, params->spheroid);
    if (kernel == NULL)
        return CL_INVALID_KERNEL_NAME;

    pl_ctx->last_time = NAN;
    gettimeofday(&start_time, NULL);

    error = pl_enqueue_projection_kernel_points(pl_ctx, kernel, proj, params, pl_buf);

    if (error != CL_SUCCESS)
        return error;

    error = pl_read_buffer(pl_ctx->queue, pl_buf->xy_out, xy_out, 2 * pl_buf->count * sizeof(cl_float));

    gettimeofday(&end_time, NULL);
    pl_ctx->last_time = (end_time.tv_sec + end_time.tv_usec * 1e-6)
        - (start_time.tv_sec + start_time.tv_usec * 1e-6);

    return error;
}

cl_int pl_project_points_forward(PLContext *pl_ctx, PLProjection proj, PLProjectionParams *params,
        PLProjectionBuffer *pl_buf, float *xy_out) {
    return _pl_project(pl_ctx, proj, params, pl_buf, xy_out, 1);
}

cl_int pl_project_points_reverse(PLContext *pl_ctx, PLProjection proj, PLProjectionParams *params,
        PLProjectionBuffer *pl_buf, float *xy_out) {
    return _pl_project(pl_ctx, proj, params, pl_buf, xy_out, 0);
}
