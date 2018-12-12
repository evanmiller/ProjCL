
#include <projcl/projcl.h>
#include "projcl_kernel.h"

#include <string.h>

cl_kernel pl_find_kernel(PLContext *pl_ctx, const char *requested_name) {
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

cl_int pl_set_kernel_arg_mem(PLContext *ctx, cl_kernel kernel, int argc, cl_mem buffer) {
    return clSetKernelArg(kernel, argc, sizeof(cl_mem), &buffer);
}

cl_int pl_set_kernel_arg_float16(PLContext *ctx, cl_kernel kernel, int argc, double value[16]) {
    float value_f[16] = { 
        value[0], value[1], value[2], value[3],
        value[4], value[5], value[6], value[7],
        value[8], value[9], value[10], value[11],
        value[12], value[13], value[14], value[15] };
    return clSetKernelArg(kernel, argc, sizeof(cl_float16), value_f);
}

cl_int pl_set_kernel_arg_float8(PLContext *ctx, cl_kernel kernel, int argc, double value[8]) {
    float value_f[8] = { value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7] };
    return clSetKernelArg(kernel, argc, sizeof(cl_float8), value_f);
}

cl_int pl_set_kernel_arg_float4(PLContext *ctx, cl_kernel kernel, int argc, double value[4]) {
    float value_f[4] = { value[0], value[1], value[2], value[3] };
    return clSetKernelArg(kernel, argc, sizeof(cl_float4), value_f);
}

cl_int pl_set_kernel_arg_float2(PLContext *ctx, cl_kernel kernel, int argc, double value[2]) {
    float value_f[2] = { value[0], value[1] };
    return clSetKernelArg(kernel, argc, sizeof(cl_float2), value_f);
}

cl_int pl_set_kernel_arg_float(PLContext *ctx, cl_kernel kernel, int argc, double value) {
    float value_f = value;
    return clSetKernelArg(kernel, argc, sizeof(cl_float), &value_f);
}

cl_int pl_set_kernel_arg_uint(PLContext *ctx, cl_kernel kernel, int argc, cl_uint value) {
    return clSetKernelArg(kernel, argc, sizeof(cl_uint), &value);
}
