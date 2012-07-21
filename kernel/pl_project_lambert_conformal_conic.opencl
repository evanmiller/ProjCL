
__kernel void pl_project_lambert_conformal_conic_e(
    __global float16 *xy_in,
    __global float16 *xy_out,
    const unsigned int count,

    float ecc,
    float ecc2,
    float one_ecc2,

    float scale,
    float x0,
    float y0,

    float lambda0,                                                   
    float rho0,
    float c,
    float n)
{
    int i = get_global_id(0);

    float8 lambda = radians(xy_in[i].even) - lambda0;
    float8 phi    = radians(xy_in[i].odd);
    
    float8 x, y;
    
    float8 rho, sinLambda, cosLambda;
    
    rho = c * pow(pl_tsfn(phi, sin(phi), ecc), (float8)n);
    sinLambda = sincos(lambda * n, &cosLambda);
    
    x = rho * sinLambda;
    y = rho0 - rho * cosLambda;
    
    xy_out[i].even = x0 + scale * x;
	xy_out[i].odd  = y0 + scale * y;
}

__kernel void pl_project_lambert_conformal_conic_s(
    __global float16 *xy_in,
    __global float16 *xy_out,
    const unsigned int count,

    float scale,
    float x0,
    float y0,

    float lambda0,                                                   
    float rho0,
    float c,
    float n)
{
    int i = get_global_id(0);
    
    float8 lambda = radians(xy_in[i].even) - lambda0;
    float8 phi    = radians(xy_in[i].odd);
    
    float8 x, y;
    
    float8 rho, sinLambda, cosLambda;

    rho = c * pow(tan(M_PI_4F + .5f * phi), (float8)(-n));    
    sinLambda = sincos(lambda * n, &cosLambda);
    
    x = rho * sinLambda;
    y = rho0 - rho * cosLambda;
    
    xy_out[i].even = x0 + scale * x;
	xy_out[i].odd  = y0 + scale * y;
}

__kernel void pl_unproject_lambert_conformal_conic_e(
    __global float16 *xy_in,
    __global float16 *xy_out,
    const unsigned int count,

    float ecc,
    float ecc2,
    float one_ecc2,

    float scale,
    float x0,
    float y0,

    float lambda0,                                                     
    float rho0,
    float c,
    float n)
{
	int i = get_global_id(0);
	
	float8 x = (xy_in[i].even - x0) / scale;
	float8 y = (xy_in[i].odd - y0) / scale;
    
	float8 lambda, phi;
    
    float8 rho;
        
    y = rho0 - y;
    
    rho = hypot(x, y);
    
    if (n < 0.f) {
        rho = -rho;
        x = -x;
        y = -y;
    }
    
    phi = select(select(-M_PI_2F, M_PI_2F, n > 0.f), pl_phi2(pow(rho / c, (float8)1.f/n), ecc), rho != 0.f);
    lambda = select(0.f, atan2(x, y) / n, rho != 0.f);
    
	xy_out[i].even = degrees(pl_mod_pi(lambda + lambda0));
	xy_out[i].odd = degrees(phi);
}


__kernel void pl_unproject_lambert_conformal_conic_s(
    __global float16 *xy_in,
    __global float16 *xy_out,
    const unsigned int count,

    float scale,
    float x0,
    float y0,

    float lambda0,                                                     
    float rho0,
    float c,
    float n)
{
	int i = get_global_id(0);
	
	float8 x = (xy_in[i].even - x0) / scale;
	float8 y = (xy_in[i].odd - y0) / scale;

	float8 lambda, phi;
    
    float8 rho;
    
    y = rho0 - y;
    
    rho = hypot(x, y);
    
    if (n < 0.f) {
        rho = -rho;
        x = -x;
        y = -y;
    }
    
    phi = select(select(-M_PI_2F, M_PI_2F, n > 0.f), M_PI_2F - 2.f * atan(pow(rho / c, (float8)1.f/n)), rho != 0.f);
    lambda = select(0.f, atan2(x, y) / n, rho != 0.f);
    
	xy_out[i].even = degrees(pl_mod_pi(lambda + lambda0));
	xy_out[i].odd = degrees(phi);
}
