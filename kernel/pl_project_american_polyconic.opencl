__kernel void pl_project_american_polyconic_s(
	__global float16 *xy_in,
	__global float16 *xy_out,
	const unsigned int count,
		
	float scale,
    float x0,
    float y0,

	float phi0,
	float lambda0)
{
	int i = get_global_id(0);

	float8 lambda = radians(xy_in[i].even) - lambda0;
	float8 phi    = radians(xy_in[i].odd);
	
	float8 sinphi, cosphi, cotphi, sinE, x, y;
	
	sinphi = sincos(phi, &cosphi);
	cotphi = cosphi / sinphi;
    sinE = sin(lambda * sinphi);
	x = cotphi * sinE;
	y = phi - phi0 + cotphi * sinE * tan(0.5f * lambda * sinphi); // half-angle formula for 1-cosE
	
	xy_out[i].even = x0 + scale * x;
	xy_out[i].odd = y0 + scale * y;
}

__kernel void pl_unproject_american_polyconic_s(
	__global float16 *xy_in,
	__global float16 *xy_out,
	const unsigned int count,
		
	float scale,
    float x0,
    float y0,

	float phi0,
	float lambda0)
{
	int i = get_global_id(0);
	
	float8 x = (xy_in[i].even - x0) / scale;
	float8 y = (xy_in[i].odd - y0) / scale + phi0;
	
	float8 lambda, phi;
	
    float8 r = y * y + x * x;
	
	float8 dPhi, tanphi;
    int iter = 4;
	
	phi = y;

    do {
		tanphi = tan(phi);
		dPhi = (y * (phi * tanphi + 1.f) - phi - 0.5f * (phi * phi + r) * tanphi) /
			((phi - y) / tanphi - 1.f);
		phi -= dPhi;
	} while (--iter);

	lambda = asin(x * tan(phi)) / sin(phi);
	
	xy_out[i].even = degrees(pl_mod_pi(lambda + lambda0));
	xy_out[i].odd = degrees(phi);
}

__kernel void pl_project_american_polyconic_e(
	__global float16 *xy_in,
	__global float16 *xy_out,
	const unsigned int count,
	
	float ecc,
	float ecc2,
	float one_ecc2,
	
	float scale,
    float x0,
    float y0,

	float phi0,
	float lambda0,
	float ml0,
	float8 en)
{
	int i = get_global_id(0);

	float8 lambda = radians(xy_in[i].even) - lambda0;
	float8 phi    = radians(xy_in[i].odd);
	
	float8 ms, sinphi, cosphi, sinE, x, y;
	
	sinphi = sincos(phi, &cosphi);
	ms = pl_msfn(sinphi, cosphi, ecc2) / sinphi;
    sinE = sin(lambda * sinphi);

	x = ms * sinE;
	y = (pl_mlfn(phi, sinphi, cosphi, en) - ml0) + ms * sinE * tan(0.5f * lambda * sinphi); // = 1.f - cosE;

	xy_out[i].even = x0 + scale * x;
	xy_out[i].odd = y0 + scale * y;
}

__kernel void pl_unproject_american_polyconic_e(
	__global float16 *xy_in,
	__global float16 *xy_out,
	const unsigned int count,
	
	float ecc,
	float ecc2,
	float one_ecc2,
	
	float scale,
    float x0,
    float y0,

	float phi0,
	float lambda0,
	float ml0,
	float8 en)
{
	int i = get_global_id(0);
	
	float8 x = (xy_in[i].even - x0) / scale;
	float8 y = (xy_in[i].odd - y0) / scale + ml0;
	
	float8 lambda, phi;
	
	float8 r = y * y + x * x;
	
	float8 c, sinphi, cosphi, sincosphi, ml, mlb, mlp, dPhi;
    int iter = 4;
		
    phi = y;

    do {
		sinphi = sincos(phi, &cosphi);
		sincosphi = sinphi * cosphi;
		mlp = sqrt(1.f - ecc2 * sinphi * sinphi);
		c = sinphi * mlp / cosphi;
		ml = pl_mlfn(phi, sinphi, cosphi, en);
		mlb = ml * ml + r;
		mlp = one_ecc2 / (mlp * mlp * mlp);
		
		dPhi = (ml + ml + c * mlb - 2.f * y * (c * ml + 1.f)) / (
			ecc2 * sincosphi * (mlb - 2.f * y * ml) / c +
			2.f * (y - ml) * (c * mlp - 1.f / sincosphi) - mlp - mlp);

		phi += dPhi;
	} while (--iter);

	c = sin(phi);
	lambda = asin(x * tan(phi) * sqrt(1.f - ecc2 * c * c)) / c;
	
	xy_out[i].even = degrees(pl_mod_pi(lambda + lambda0));
	xy_out[i].odd = degrees(phi);
}
