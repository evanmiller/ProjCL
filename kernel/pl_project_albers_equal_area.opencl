
float8 phi1_(float8 qs, float Te, float Tone_es);

float8 phi1_(float8 qs, float Te, float Tone_es) {
	int i;
	float8 Phi, sinphi, cosphi, con, com, dphi;

	Phi = asin(.5f * qs);
	
	i = ALBERS_EQUAL_AREA_N_ITER;
	
	do {
		sinphi = sincos(Phi, &cosphi);
		con = Te * sinphi;
		com = 1.f - con * con;
		dphi = .5f * com * com / cosphi * (qs / Tone_es - sinphi / com - atanh(con) / Te); 
		Phi += dphi;
	} while (any(fabs(dphi) > TOL7) && --i);
	
	return Phi;
}

__kernel void pl_project_albers_equal_area_s(
	__global float16 *xy_in,
	__global float16 *xy_out,
	const unsigned int count,
	
    float scale,
    float x0,
    float y0,

	float lambda0,
	float rho0,
	float c,
	float n
) {
	int i = get_global_id(0);
	
	float8 lambda = radians(xy_in[i].even) - lambda0;
	float8 phi    = radians(xy_in[i].odd);
	
	float8 x, y;
	
	float8 rho, sinLambda, cosLambda;

	rho = sqrt(c - 2 * n * sin(phi));

	sinLambda = sincos(lambda * n, &cosLambda);
	
	x = rho * sinLambda;
	y = rho0 - rho * cosLambda;
	
	xy_out[i].even = x0 + scale * x;
	xy_out[i].odd  = y0 + scale * y;
}

__kernel void pl_unproject_albers_equal_area_s(
	__global float16 *xy_in,
	__global float16 *xy_out,
	const unsigned int count,
	
    float scale,
    float x0,
    float y0,

	float lambda0,
	float rho0,
	float c,
	float n
) {
	int i = get_global_id(0);
	
	float8 x = (xy_in[i].even - x0) / scale;
	float8 y = (xy_in[i].odd - y0) / scale;
	
	float8 lambda, phi;
	
	y = rho0 - y;

    phi = asin(clamp(0.5f * (c - (x * x + y * y)) / n, -1.f, 1.f));
	lambda = atan2(x, y) / n;
	
	xy_out[i].even = degrees(pl_mod_pi(lambda + lambda0));
	xy_out[i].odd = degrees(phi);
}

__kernel void pl_project_albers_equal_area_e(
	__global float16 *xy_in,
	__global float16 *xy_out,
	const unsigned int count,
	
	float ecc,
	float ecc2,
	float one_ecc2,
	
	float ec,
	
    float scale,
    float x0,
    float y0,
	float lambda0,
	float rho0,
	float c,
	float n
) {
	int i = get_global_id(0);
	
	float8 lambda = radians(xy_in[i].even) - lambda0;
	float8 phi    = radians(xy_in[i].odd);
	
	float8 x, y;
	
	float8 rho, sinLambda, cosLambda;

	rho = sqrt(c - n * pl_qsfn(sin(phi), ecc, one_ecc2));

	sinLambda = sincos(lambda * n, &cosLambda);
	
	x = rho * sinLambda;
	y = rho0 - rho * cosLambda;
	
	xy_out[i].even = x0 + scale * x;
	xy_out[i].odd  = y0 + scale * y;
}

__kernel void pl_unproject_albers_equal_area_e(
	__global float16 *xy_in,
	__global float16 *xy_out,
	const unsigned int count,
	
	float ecc,
	float ecc2,
	float one_ecc2,
	
	float ec,
	
    float scale,
    float x0,
    float y0,

	float lambda0,
	float rho0,
	float c,
	float n
) {
	int i = get_global_id(0);
	
	float8 x = (xy_in[i].even - x0) / scale;
	float8 y = (xy_in[i].odd - y0) / scale;
	
	float8 lambda, phi;
	
	y = rho0 - y;

	phi = (c - (x * x + y * y)) / n;
	
	phi = select(copysign(M_PI_2F, phi), phi1_(phi, ecc, one_ecc2), fabs(ec - fabs(phi)) > TOL7);
	lambda = atan2(x, y) / n;
	
	xy_out[i].even = degrees(pl_mod_pi(lambda + lambda0));
	xy_out[i].odd = degrees(phi);
}
