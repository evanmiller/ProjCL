
cl_kernel pl_find_kernel(PLContext *pl_ctx, const char *requested_name);

cl_int pl_set_kernel_arg_float(PLContext *ctx, cl_kernel kernel, int argc, double value);
cl_int pl_set_kernel_arg_float2(PLContext *ctx, cl_kernel kernel, int argc, double value[2]);
cl_int pl_set_kernel_arg_float4(PLContext *ctx, cl_kernel kernel, int argc, double value[4]);
cl_int pl_set_kernel_arg_float8(PLContext *ctx, cl_kernel kernel, int argc, double value[8]);
cl_int pl_set_kernel_arg_float16(PLContext *ctx, cl_kernel kernel, int argc, double value[16]);
cl_int pl_set_kernel_arg_uint(PLContext *ctx, cl_kernel kernel, int argc, cl_uint value);
cl_int pl_set_kernel_arg_mem(PLContext *ctx, cl_kernel kernel, int argc, cl_mem buffer);
