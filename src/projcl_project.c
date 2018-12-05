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

// The _2 interface is similar to the standard one above, except:
//
// No copy flag - data is always copied from 2 arrays to 1 interleaved
// No local padded copies - data is copied to padded buffer,
//  any pad area in buffer is zeroed out
//
PLProjectionBuffer *pl_load_projection_data_2(PLContext *pl_ctx, const float *x, const float *y, int count, int *outError) {
	int xy_pad_count = ck_padding(count, PL_FLOAT_VECTOR_SIZE);

	PLProjectionBuffer *pl_buf = NULL;
	cl_int error = CL_SUCCESS;
	
	int needs_free = 0;
	
	struct timeval start_time, end_time;
	gettimeofday(&start_time, NULL);
	
	pl_ctx->last_time = NAN;
	
	if ((pl_buf = malloc(sizeof(PLProjectionBuffer))) == NULL) {
		error = CL_OUT_OF_HOST_MEMORY;
		goto cleanup;
	}

	// For this case, create the buffer empty and writable,
	// fill it with clEnqueueWriteBufferRect
	pl_buf->xy_in = clCreateBuffer(pl_ctx->ctx,
				       CL_MEM_READ_ONLY,
				       sizeof(cl_float) * xy_pad_count * 2,
				       NULL,
				       &error);
	if (error != CL_SUCCESS) {
		goto cleanup;
	}
	size_t buffer_origin[3] = {0,0,0};
	size_t host_origin[3] = {0,0,0};
	size_t bh_region[3] = {sizeof(float), count, 1};
	// First, the x's
	error = clEnqueueWriteBufferRect(pl_ctx->queue,
					 pl_buf->xy_in,
					 CL_TRUE,  // non-blocking
					 buffer_origin,
					 host_origin,
					 bh_region,
					 2*sizeof(float), // buffer_row_pitch,
					 0,               // buffer_slice_pitch,
					 1*sizeof(float), // host_row_pitch,
					 0,               // host_slice_pitch,
					 x,
					 0,     // num_events_in_wait_list,
					 NULL,  // event_wait_list,
					 NULL   // event
					 );
	if (error != CL_SUCCESS) {
	  goto cleanup;
	}
	// Then, the y's, which are like the x's with minor offsets
	buffer_origin[0] = 1 * sizeof(float);
	error = clEnqueueWriteBufferRect(pl_ctx->queue,
					 pl_buf->xy_in,
					 CL_TRUE,  // non-blocking
					 buffer_origin,
					 host_origin,
					 bh_region,
					 2*sizeof(float), // buffer_row_pitch,
					 0,               // buffer_slice_pitch,
					 1*sizeof(float), // host_row_pitch,
					 0,               // host_slice_pitch,
					 y,
					 0,       // num_events_in_wait_list,
					 NULL,    // event_wait_list,
					 NULL     //event
					 );
	if (error != CL_SUCCESS) {
	  goto cleanup;
	}

	// Pad with clEnqueueFillBuffer here
	/*
	if (xy_pad_count != count) {
	  float pad_value = 0.0;
	  error = clEnqueueFillBuffer(pl_ctx->queue,
				      pl_buf->xy_in,
				      &pad_value,            // pattern
				      sizeof(float),         // pattern_size
				      count * sizeof(float), // offset
				      (xy_pad_count - count) * sizeof(float), // size
				      0,       // num_events_in_wait_list,
				      NULL,    // event_wait_list,
				      NULL     //event
				      );
	  if (error != CL_SUCCESS) {
	    goto cleanup;
	  }
	}
	*/
	pl_buf->xy_out = clCreateBuffer(pl_ctx->ctx, CL_MEM_WRITE_ONLY, sizeof(cl_float) * count * 2, NULL, &error);
	if (error != CL_SUCCESS) {
		goto cleanup;
	}
	
	pl_buf->count = count;
	
cleanup:
	
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

cl_int pl_compare_projection_buffers(PLContext *pl_ctx, PLProjectionBuffer *pl_buf_1, PLProjectionBuffer *pl_buf_2, char **error_string) {

  if(pl_buf_1->count != pl_buf_2->count) {
    if(error_string != NULL)
      *error_string = "Buffer sizes do not match";
    return 1;
  }

  cl_uint eltcount = 2 * pl_buf_1->count;

  float *bc1 = malloc(sizeof(float) * eltcount);
  float *bc2 = malloc(sizeof(float) * eltcount);
  if(bc1 == NULL || bc2 == NULL) {
    if(error_string != NULL)
      *error_string = "Could not allocate memory to read buffers";
    return 1;
  }

  cl_int error = pl_read_buffer(pl_ctx->queue, pl_buf_1->xy_in, bc1, eltcount * sizeof(float));
  if(error != CL_SUCCESS) {
    if(error_string != NULL)
      *error_string = "OpenCL error reading buffer 1";
    return error;
  }

  error = pl_read_buffer(pl_ctx->queue, pl_buf_2->xy_in, bc2, eltcount * sizeof(float));
  if(error != CL_SUCCESS) {
    if(error_string != NULL)
      *error_string = "OpenCL error reading buffer 2";
    return error;
  }

  error = 0;
  int i;
  for(i=0; i< eltcount; i++) {
    if(isnan(bc1[i])){
      if(!isnan(bc2[i]))
	++error;
    }
    else {
      if(bc1[i] != bc2[i])
	++error;
    }
  }

  free(bc2);
  free(bc1);
  if(error != 0 && error_string != NULL)
    *error_string = "Contents of buffers did not match";
  return error;
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

static cl_int _pl_project_2(PLContext *pl_ctx, PLProjection proj, PLProjectionParams *params,
			    PLProjectionBuffer *pl_buf, float *x_out, float *y_out, int fwd) {
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

        error = _pl_project_2(pl_ctx, PL_PROJECT_MERCATOR, params2, pl_buf, x_out, y_out, fwd);

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

    error = pl_read_buffer_2(pl_ctx->queue, pl_buf->xy_out, x_out, y_out, 2 * pl_buf->count * sizeof(cl_float));

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

cl_int pl_project_points_forward_2(PLContext *pl_ctx, PLProjection proj, PLProjectionParams *params,
				   PLProjectionBuffer *pl_buf, float *x_out, float *y_out) {
  return _pl_project_2(pl_ctx, proj, params, pl_buf, x_out, y_out, 1);
}

cl_int pl_project_points_reverse_2(PLContext *pl_ctx, PLProjection proj, PLProjectionParams *params,
				   PLProjectionBuffer *pl_buf, float *x_out, float *y_out) {
  return _pl_project_2(pl_ctx, proj, params, pl_buf, x_out, y_out, 0);
}
