//
//  projcl_util.c
//  Magic Maps
//
//  Created by Evan Miller on 3/31/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#include <projcl/projcl.h>
#include "projcl_util.h"
#include "projcl_spheroid.h"
#include <strings.h>
#include <stdio.h>

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

const char *_pl_proj_name(PLProjection proj) {
    if (proj == PL_PROJECT_ALBERS_EQUAL_AREA)
        return "albers_equal_area";
    if (proj == PL_PROJECT_AMERICAN_POLYCONIC)
        return "american_polyconic";
    if (proj == PL_PROJECT_LAMBERT_AZIMUTHAL_EQUAL_AREA)
        return "lambert_azimuthal_equal_area";
    if (proj == PL_PROJECT_LAMBERT_CONFORMAL_CONIC)
        return "lambert_conformal_conic";
    if (proj == PL_PROJECT_MERCATOR)
        return "mercator";
    if (proj == PL_PROJECT_OBLIQUE_STEREOGRAPHIC)
        return "oblique_stereographic";
    if (proj == PL_PROJECT_ROBINSON)
        return "robinson";
    if (proj == PL_PROJECT_TRANSVERSE_MERCATOR)
        return "transverse_mercator";
    if (proj == PL_PROJECT_WINKEL_TRIPEL)
        return "winkel_tripel";

    return NULL;
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

