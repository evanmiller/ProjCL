__kernel void pl_project_mercator_s(
	__global float16 *xy_in,
	__global float16 *xy_out,
	const unsigned int count,
		
	float scale, float x0, float y0)
{
	int i = get_global_id(0);
	
	float8 lambda = radians(xy_in[i].even);
	float8 phi    = radians(xy_in[i].odd);
	
	float8 x, y;
	
	x = lambda;
	y = asinh(tan(phi));

	xy_out[i].even = x0 + scale * x;
	xy_out[i].odd  = y0 + scale * y;
}

__kernel void pl_unproject_mercator_s(
	__global float16 *xy_in,
	__global float16 *xy_out,
	const unsigned int count,
		
	float scale, float x0, float y0)
{
	int i = get_global_id(0);
	
	float8 x = (xy_in[i].even - x0) / scale;
	float8 y = (xy_in[i].odd - y0) / scale;
	
	float8 lambda, phi;
	
	phi = atan(sinh(y));
	lambda = x;

	xy_out[i].even = degrees(lambda);
	xy_out[i].odd = degrees(phi);
}

__kernel void pl_project_mercator_e(
	__global float16 *xy_in,
	__global float16 *xy_out,
	const unsigned int count,
	
	float ecc,
	float ecc2,
	float one_ecc2,
	
	float scale, float x0, float y0)
{
	int i = get_global_id(0);
	
	float8 lambda = radians(xy_in[i].even);
	float8 phi    = radians(xy_in[i].odd);
	
	float8 x, y;
	
	x = lambda;
	y = asinh(tan(phi)) - ecc * atanh(ecc * sin(phi));
	
	xy_out[i].even = x0 + scale * x;
	xy_out[i].odd  = y0 + scale * y;
}

__kernel void pl_unproject_mercator_e(
	__global float16 *xy_in,
	__global float16 *xy_out,
	const unsigned int count,
	
	float ecc,
	float ecc2,
	float one_ecc2,
	
	float scale, float x0, float y0)
{
	int i = get_global_id(0);
	
	float8 x = (xy_in[i].even - x0) / scale;
	float8 y = (xy_in[i].odd - y0) / scale;
	
	float8 lambda, phi;
	
	phi = pl_phi2(-y, ecc);
	lambda = x;
	
	xy_out[i].even = degrees(lambda);
	xy_out[i].odd = degrees(phi);
}
