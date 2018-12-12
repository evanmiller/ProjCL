/*
 *  projcl.c
 *  Magic Maps
 *
 *  Created by Evan Miller on 2/7/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


#include <projcl/projcl.h>
#include <projcl/projcl_warp.h>

#include <dirent.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/uio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/time.h>
#include <math.h>

typedef struct pl_module_s {
    char    file[80];
    int     module;
} pl_module_t;

static pl_module_t _pl_modules[] = {
    {
        .file = "pl_datum.opencl",
        .module = PL_MODULE_DATUM
    },
    {
        .file = "pl_geodesic.opencl",
        .module = PL_MODULE_GEODESIC
    },
    {
        .file = "pl_warp.opencl",
        .module = PL_MODULE_WARP
    },
    {
        .file = "pl_sample_nearest.opencl",
        .module = PL_MODULE_NEAREST_NEIGHBOR
    },
    {
        .file = "pl_sample_linear.opencl",
        .module = PL_MODULE_BILINEAR
    },
    {
        .file = "pl_sample_bicubic.opencl",
        .module = PL_MODULE_BICUBIC
    },
    {
        .file = "pl_sample_quasi_bicubic.opencl",
        .module = PL_MODULE_QUASI_BICUBIC
    },
    {
        .file = "pl_project_albers_equal_area.opencl",
        .module = PL_MODULE_ALBERS_EQUAL_AREA
    },
    {
        .file = "pl_project_american_polyconic.opencl",
        .module = PL_MODULE_AMERICAN_POLYCONIC
    },
    {
        .file = "pl_project_lambert_azimuthal_equal_area.opencl",
        .module = PL_MODULE_LAMBERT_AZIMUTHAL_EQUAL_AREA
    },
    {
        .file = "pl_project_lambert_conformal_conic.opencl",
        .module = PL_MODULE_LAMBERT_CONFORMAL_CONIC
    },
    {
        .file = "pl_project_mercator.opencl",
        .module = PL_MODULE_MERCATOR
    },
    {
        .file = "pl_project_oblique_stereographic.opencl",
        .module = PL_MODULE_OBLIQUE_STEREOGRAPHIC
    },
    {
        .file = "pl_project_robinson.opencl",
        .module = PL_MODULE_ROBINSON
    },
    {
        .file = "pl_project_transverse_mercator.opencl",
        .module = PL_MODULE_TRANSVERSE_MERCATOR
    },
    {
        .file = "pl_project_winkel_tripel.opencl",
        .module = PL_MODULE_WINKEL_TRIPEL
    }
};

#define PL_DEBUG 0

#define PL_OPENCL_FILE_EXTENSION ".opencl"
#define PL_OPENCL_KERNEL_HEADER_FILE "peel.opencl"
#define PL_OPENCL_KERNEL_FILE_PREFIX "pl_"

#define PL_OPENCL_BUILD_OPTIONS "-Werror -cl-std=CL1.1 -cl-finite-math-only -cl-no-signed-zeros -cl-single-precision-constant"

int check_cl_error(cl_int error, cl_int *outError) {
  if(error != CL_SUCCESS) {
    if (outError != NULL) {
      *outError = error;
    }
    return 1;
  }
  return 0;
}

#define PLATFORM_INDEX 0
#define DEVICE_INDEX 0

PLContext *pl_context_init(cl_device_type type, cl_int *outError) {
	cl_int error;

	cl_uint num_platforms;
	error = clGetPlatformIDs(0, NULL, &num_platforms);
	if(check_cl_error(error, outError))
	  return NULL;
#if PL_DEBUG
	printf("OpenCL num_platforms: %d\n", num_platforms);
#endif

	cl_platform_id platform_id[num_platforms];
	error = clGetPlatformIDs(num_platforms, platform_id, NULL);
	if(check_cl_error(error, outError))
	  return NULL;

	cl_uint num_devices;
	error = clGetDeviceIDs(platform_id[PLATFORM_INDEX], type, 0, NULL, &num_devices);
	if(check_cl_error(error, outError))
	  return NULL;
#if PL_DEBUG
	printf("OpenCL num_devices: %d\n", num_devices);
#endif

	cl_device_id device_id;	
	error = clGetDeviceIDs(platform_id[PLATFORM_INDEX], type, num_devices, &device_id, NULL);
	if(check_cl_error(error, outError))
	  return NULL;
    
	cl_context ctx = clCreateContext(NULL, 1, &device_id, NULL, NULL, &error);
	if(check_cl_error(error, outError))
	  return NULL;
	
	cl_command_queue queue = clCreateCommandQueue(ctx, device_id, 0, &error);
	if(check_cl_error(error, outError)) {
	  clReleaseContext(ctx);
	  return NULL;
	}
    
#if PL_DEBUG
    int size;
    long lsize;
    printf("OpenCL device debug info\n");
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(int), &size, NULL);
    printf("-- Compute units: %d\n", size);
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(long), &lsize, NULL);
    printf("-- Work group size: %ld\n", lsize);

    cl_uint vector_width;
    clGetDeviceInfo(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, sizeof(cl_uint), &vector_width, NULL);
    printf("-- Preferred vector width - CHAR:   %u\n", vector_width);
    clGetDeviceInfo(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, sizeof(cl_uint), &vector_width, NULL);
    printf("-- Preferred vector width - SHORT:  %u\n", vector_width);
    clGetDeviceInfo(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(cl_uint), &vector_width, NULL);
    printf("-- Preferred vector width - INT:    %u\n", vector_width);
    clGetDeviceInfo(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, sizeof(cl_uint), &vector_width, NULL);
    printf("-- Preferred vector width - LONG:   %u\n", vector_width);
    clGetDeviceInfo(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(cl_uint), &vector_width, NULL);
    printf("-- Preferred vector width - FLOAT:  %u\n", vector_width);
    clGetDeviceInfo(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint), &vector_width, NULL);
    printf("-- Preferred vector width - DOUBLE: %u\n", vector_width);
    clGetDeviceInfo(device_id, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(long), &lsize, NULL);
    printf("-- Max 2D image width: %ld\n", lsize);
    clGetDeviceInfo(device_id, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(long), &lsize, NULL);
    printf("-- Max 2D image height: %ld\n", lsize);
    clGetDeviceInfo(device_id, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(long), &lsize, NULL);
    printf("-- Max 3D image width: %ld\n", lsize);
    clGetDeviceInfo(device_id, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(long), &lsize, NULL);
    printf("-- Max 3D image height: %ld\n", lsize);
    clGetDeviceInfo(device_id, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(long), &lsize, NULL);
    printf("-- Max 3D image depth: %ld\n", lsize);
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(int), &size, NULL);
    printf("-- Max read image args: %d\n", size);
#endif

	PLContext *pl_ctx;
	
	if ((pl_ctx = calloc(1, sizeof(PLContext))) == NULL) {
		clReleaseContext(ctx);
		clReleaseCommandQueue(queue);
		if (outError != NULL)
			*outError = CL_OUT_OF_HOST_MEMORY;
		return NULL;
	}
	
	pl_ctx->ctx = ctx;
	pl_ctx->queue = queue;
	pl_ctx->device_id = device_id;
	return pl_ctx;
}

void pl_context_free(PLContext *pl_ctx) {
	clReleaseContext(pl_ctx->ctx);
	clReleaseCommandQueue(pl_ctx->queue);
	free(pl_ctx);
}

PLCode *pl_compile_code(PLContext *pl_ctx, const char *path, long modules, cl_int *outError) {
	cl_int error;
	cl_program program;
	
	DIR *dir = opendir(path);
	if (dir == NULL) {
        if (outError)
            *outError = 1;
		return NULL;
	}
	struct dirent *entry;
    size_t bytes_read;
	char buffer[1024*1024];
    char filename[1024];
	char * pointers[256];
	size_t buf_used = 0;
	int entry_index = 0;
    int kernel_count = 0;
    struct timeval start_time, end_time;
    
    pl_ctx->last_time = NAN;
    gettimeofday(&start_time, NULL);

    if (modules == 0)
        modules = ~0;
    
    /* First read the header file */
    if (strlen(path) + sizeof(PL_OPENCL_KERNEL_HEADER_FILE) + 1 < sizeof(filename)) {
        pointers[entry_index++] = buffer + buf_used;
        sprintf(filename, "%s/" PL_OPENCL_KERNEL_HEADER_FILE, path);
        int fd = open(filename, O_RDONLY);
        if (fd == -1) {
            return NULL;
        }
        while ((bytes_read = read(fd, buffer + buf_used, sizeof(buffer) - buf_used - 1)) > 0) {
            buf_used += bytes_read;
        }
        buffer[buf_used++] = '\0';
        close(fd);
    }
    
    /* Then read the routine files */
	while ((entry = readdir(dir)) != NULL) {
		if (entry_index >= sizeof(pointers)/sizeof(char *) - 1) {
			break;
		}
		if (buf_used >= sizeof(buffer)) {
			break;
		}
		
		size_t len = strlen(entry->d_name);
		const char *name = entry->d_name;
        
		if (len > sizeof(PL_OPENCL_FILE_EXTENSION)-1 
            && strncasecmp(name, PL_OPENCL_KERNEL_FILE_PREFIX, sizeof(PL_OPENCL_KERNEL_FILE_PREFIX) - 1) == 0
            && strcasecmp(name + len - (sizeof(PL_OPENCL_FILE_EXTENSION)-1), PL_OPENCL_FILE_EXTENSION) == 0) {
            int i;
            int compile = 0;
            for (i=0; i<sizeof(_pl_modules)/sizeof(_pl_modules[0]); i++) {
                if (strcmp(name, _pl_modules[i].file) == 0) {
                    compile = !!(modules & _pl_modules[i].module);
                    break;
                }
            }
            if (!compile)
                continue;

            if (strlen(path) + strlen(name) + 2 < sizeof(filename)) {
				pointers[entry_index++] = buffer + buf_used;
				sprintf(filename, "%s/%s", path, name);
				int fd = open(filename, O_RDONLY);
				if (fd == -1) {
					continue;
				}
				while ((bytes_read = read(fd, buffer + buf_used, sizeof(buffer) - buf_used - 1)) > 0) {
					buf_used += bytes_read;
				}
				buffer[buf_used++] = '\0';
				close(fd);
                
                char *p = pointers[entry_index-1];
				while ((p = strstr(p, "__kernel")) != NULL) {
					kernel_count++;
                    p += sizeof("__kernel")-1;
				}
			}
		}
	} 
	
	closedir(dir);
	
	if (entry_index == 0) {
        if (outError)
            *outError = 2;
		return NULL;
	}
	
	pointers[entry_index] = NULL;
	program = clCreateProgramWithSource(pl_ctx->ctx, entry_index, (const char **)pointers, NULL, &error);
	
	if (error != CL_SUCCESS) {
		if (outError != NULL)
			*outError = error;
		return NULL;
	}
	
	error = clBuildProgram(program, 0, NULL, PL_OPENCL_BUILD_OPTIONS, NULL, NULL);
	
    size_t error_len;
    char error_buf[4*8192];
    clGetProgramBuildInfo(program, pl_ctx->device_id, CL_PROGRAM_BUILD_LOG, sizeof(error_buf), 
            error_buf, &error_len);
#if PL_DEBUG
    if (error_len > 1)
        printf("%s\n", error_buf);
#endif

	if (error != CL_SUCCESS) {
#if !PL_DEBUG
        printf("%s\n", error_buf);
#endif
		printf("Error: Failed to build program executable!\n");
		
		if (outError != NULL)
			*outError = error;
		clReleaseProgram(program);
		return NULL;
	}
	
	PLCode *pl_code;
	if ((pl_code = malloc(sizeof(PLCode))) == NULL) {
		if (outError != NULL)
			*outError = CL_OUT_OF_HOST_MEMORY;
		return NULL;
	}
	
	pl_code->program = program;
	pl_code->kernel_count = kernel_count;
    
    if (outError)
        *outError = CL_SUCCESS;

    gettimeofday(&end_time, NULL);

    pl_ctx->last_time = (end_time.tv_sec + end_time.tv_usec * 1e-6)
        - (start_time.tv_sec + start_time.tv_usec * 1e-6);
	
	return pl_code;
}

cl_int pl_load_code(PLContext *pl_ctx, PLCode *pl_code) {
	cl_int error;

    struct timeval start_time, end_time;
    
    pl_ctx->last_time = NAN;
    gettimeofday(&start_time, NULL);
	
	cl_uint kernel_count_ret;
	
	cl_kernel *kernels;
	if ((kernels = malloc(sizeof(cl_kernel) * pl_code->kernel_count)) == NULL) {
		return CL_OUT_OF_HOST_MEMORY;
	}
	
	error = clCreateKernelsInProgram(pl_code->program, pl_code->kernel_count, kernels, &kernel_count_ret);
	
	if (error != CL_SUCCESS) {
		free(kernels);
		return error;
	}
	
	pl_ctx->kernel_count = kernel_count_ret;
	pl_ctx->kernels = kernels;

    gettimeofday(&end_time, NULL);

    pl_ctx->last_time = (end_time.tv_sec + end_time.tv_usec * 1e-6)
        - (start_time.tv_sec + start_time.tv_usec * 1e-6);
	
	return CL_SUCCESS;
}

void pl_unload_code(PLContext *pl_ctx) {
	int i;
	for (i=0; i<pl_ctx->kernel_count; i++) {
		clReleaseKernel(pl_ctx->kernels[i]);
	}
	free(pl_ctx->kernels);
	
	pl_ctx->kernel_count = 0;
	pl_ctx->kernels = NULL;
}

void pl_release_code(PLCode *pl_code) {
    if (!pl_code)
        return;
    if (pl_code->program)
        clReleaseProgram(pl_code->program);
    free(pl_code);
}
