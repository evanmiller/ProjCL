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

void _pl_copy_pad(float *dest, size_t dest_count, const float *src, size_t src_count) {
	int i;
	for (i=0; i<src_count; i++) {
		dest[i] = src[i];
	}
	for (i=src_count; i<dest_count; i++) {
		dest[i] = 0.0;
	}
}

