
float8 pl_mlfn(float8 phi, float8 sphi, float8 cphi, float8 en);
float8 pl_mlfn1(float8 phi, float8 sphi, float8 cphi, float8 en);

float8 pl_mlfn(float8 phi, float8 sphi, float8 cphi, float8 en) {
	cphi *= sphi;
	sphi *= sphi;
	return(en.s0 * phi - cphi * (en.s1 + sphi*(en.s2 + sphi*(en.s3 + sphi*en.s4))));
}

/* first derivative */
float8 pl_mlfn1(float8 phi, float8 sphi, float8 cphi, float8 en) {
    cphi *= cphi;
    sphi *= sphi;
    return en.s0 - (en.s1*(cphi-sphi) + sphi*(en.s2*(3*cphi-sphi) 
                    + sphi*(en.s3*(5*cphi-sphi) + sphi*(en.s4*(7*cphi-sphi)))));
}


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
	float8 y = (xy_in[i].odd - y0) / scale;
	
	float8 lambda, phi;

    float8 dPhi, dLam, cosPhi, sinPhi;
    float8 sinLSinPhi, cosLSinPhi, cosLSinPhi1;
    float8 f1, f2, df1phi, df2phi, df1lam, df2lam;
    float8 c, invDet;

    int iter = AMERICAN_POLYCONIC_N_ITER;

    phi = (y + phi0); // * cos(.5f * x); need a better initial guess
    sinPhi = sincos(phi, &cosPhi);
    lambda = asin(x * sinPhi / cosPhi) / sinPhi;

    do { // Newton-Raphson w/ full Jacobian matrix 
        sinLSinPhi = sincos(lambda * sinPhi, &cosLSinPhi);
        cosLSinPhi1 = sinLSinPhi * tan(0.5f * lambda * sinPhi); // half-angle formula

        c = lambda * cosPhi * cosPhi / sinPhi;

        f1 = cosPhi * sinLSinPhi / sinPhi - x;
        f2 = phi - phi0 + cosPhi * cosLSinPhi1 / sinPhi - y;

        df1phi = c * cosLSinPhi - sinLSinPhi / sinPhi / sinPhi;
        df2phi = 1.f + c * sinLSinPhi - cosLSinPhi1 / sinPhi / sinPhi;
        df1lam = cosPhi * cosLSinPhi;
        df2lam = cosPhi * sinLSinPhi;

        invDet = 1.f / (df1phi * df2lam - df2phi * df1lam);

        dPhi = (f1 * df2lam - f2 * df1lam) * invDet;
        dLam = (f2 * df1phi - f1 * df2phi) * invDet;

        phi -= dPhi;
        lambda -= dLam;

        sinPhi = sincos(phi, &cosPhi);
	} while (--iter);

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
	ms = cosphi / sinphi / sqrt(1.f - ecc2 * sinphi * sinphi);

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
	float8 y = (xy_in[i].odd - y0) / scale;
	
	float8 lambda, phi;
	
	float8 c1, c2, sinPhi, cosPhi, tanPhi, sinLSinPhi, cosLSinPhi;
    float8 sinL2SinPhi, cosL2SinPhi;
    float8 mlp;

    float8 dLam, dPhi, invDet;
    float8 f1, f2, df1phi, df2phi, df1lam, df2lam;
	
    int iter = AMERICAN_POLYCONIC_N_ITER + 2; // seems to require more than spherical case... maybe a bug?

    phi = y + phi0;
    // 1 iteration of Newton's method to fix initial guess
    phi -= (pl_mlfn(phi, sin(phi), cos(phi), en) - (y + ml0)) / pl_mlfn1(phi, sin(phi), cos(phi), en);

    sinPhi = sincos(phi, &cosPhi);
    mlp = sqrt(1.f - ecc2 * sinPhi * sinPhi);
	lambda = asin(x * sinPhi / cosPhi * mlp) / sinPhi;

    do { // Newton-Raphson
        tanPhi = sinPhi / cosPhi;

        sinLSinPhi = sincos(lambda * sinPhi, &cosLSinPhi);
        sinL2SinPhi = sincos(.5f * lambda * sinPhi, &cosL2SinPhi);

        f1 = sinLSinPhi / tanPhi / mlp - x;
        f2 = pl_mlfn(phi, sinPhi, cosPhi, en) - ml0 + x * sinL2SinPhi / cosL2SinPhi - y;

        df1lam = cosPhi / mlp * cosLSinPhi;
        df2lam = cosPhi / mlp * sinLSinPhi;

        c1 = ecc2 * (1.f + cosPhi * cosPhi) / (mlp * (1.f - ecc2 * sinPhi * sinPhi));
        c2 = 1.f / (sinPhi * sinPhi * mlp * (1.f - ecc2 * sinPhi * sinPhi));

        df1phi = lambda * cosPhi / tanPhi / mlp * cosLSinPhi + sinLSinPhi * (c1 - c2);
        df2phi = pl_mlfn1(phi, sinPhi, cosPhi, en) + 0.5f * lambda * x * cosPhi / cosL2SinPhi / cosL2SinPhi;

        invDet = 1.f / (df1phi * df2lam - df2phi * df1lam);

        dPhi = (f1 * df2lam - f2 * df1lam) * invDet;
        dLam = (f2 * df1phi - f1 * df2phi) * invDet;

        phi -= dPhi;
        lambda -= dLam;

        sinPhi = sincos(phi, &cosPhi);
        mlp = sqrt(1.f - ecc2 * sinPhi * sinPhi);
    } while(--iter);

	xy_out[i].even = degrees(pl_mod_pi(lambda + lambda0));
	xy_out[i].odd = degrees(phi);
}
