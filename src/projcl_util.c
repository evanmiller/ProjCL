//
//  projcl_util.c
//  Magic Maps
//
//  Created by Evan Miller on 3/31/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#include "projcl_util.h"
#include <strings.h>
#include <stdio.h>

int _pl_spheroid_is_spherical(PLSpheroid ell) {
    return ell == PL_SPHEROID_SPHERE || ell == PL_SPHEROID_WGS_84_MAJOR_AUXILIARY_SPHERE;
}

cl_kernel _pl_find_kernel(PLContext *pl_ctx, const char *requested_name) {
	char buf[128];
	size_t len;
	cl_int error;
	cl_kernel kernel;
	int i;
	for (i=0; i<pl_ctx->kernel_count; i++) {
		kernel = pl_ctx->kernels[i];
		error = clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 
								sizeof(buf), buf, &len);
		if (strncmp(requested_name, buf, len) == 0) {
			return kernel;
		}
	}
	return NULL;
}

cl_kernel _pl_find_projection_kernel(PLContext *pl_ctx, const char *name, int fwd, PLSpheroid ell) {
	char requested_name[128];
	if (fwd) {
		sprintf(requested_name, "pl_project_%s_%c", name, _pl_spheroid_is_spherical(ell) ? 's' : 'e');
	} else {
		sprintf(requested_name, "pl_unproject_%s_%c", name, _pl_spheroid_is_spherical(ell) ? 's' : 'e');
	}
	return _pl_find_kernel(pl_ctx, requested_name);
}

void _pl_copy_pad(float *dest, size_t dest_count, const float *src, size_t src_count) {
	int i;
	for (i=0; i<src_count; i++) {
		dest[i] = src[i];
	}
	for (i=src_count; i<dest_count; i++) {
		dest[i] = 0.0;
	}
}

