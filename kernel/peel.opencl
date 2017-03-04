#define EPS 5e-14f
#define EPS10   1.e-10f
#define EPS8    1.e-8f
#define EPS7    1.e-7f
#define EPS6    1.e-6f
#define ITOL 1.e-12f
#define TOL7 1.e-7f
#define TOL6 1.e-6f
#define TOL5 1.e-5f
#define I_ITER 20

#define ALBERS_EQUAL_AREA_N_ITER 15
#define OBLIQUE_STEREOGRAPHIC_N_ITER 15

#define M_PIF	         3.1415926535897932384626f
#define M_PI_2F          1.570796326794896557999f
#define M_PI_4F			 0.78539816339744833f

/* Robinson */
#define FXC     0.8487f
#define FYC     1.3523f
#define C1      11.45915590261646417544f
#define RC1     0.08726646259971647884f
#define NODES   18
#define ONEEPS  1.000001f

/* Transverse Mercator */
#define FC1 1.f
#define FC2 .5f
#define FC3 .16666666666666666666f
#define FC4 .08333333333333333333f
#define FC5 .05f
#define FC6 .03333333333333333333f
#define FC7 .02380952380952380952f
#define FC8 .01785714285714285714f

float8 pl_mlfn(float8 phi, float8 sphi, float8 cphi, float8 en);
float8 pl_inv_mlfn(float8 argphi, float es, float8 en);
float8 pl_msfn(float8 sinphi, float8 cosphi, float es);
float8 pl_tsfn(float8 phi, float8 sinphi, float e);
float8 pl_qsfn(float8 sinphi, float e, float one_es);
float8 pl_phi2(float8 ts, float8 e);
float8 pl_mod_pi(float8 phi);
float4 pl_interpolate_cubic4(float X, float4 A, float4 B, float4 C, float4 D);

float8 pl_mlfn(float8 phi, float8 sphi, float8 cphi, float8 en) {
	cphi *= sphi;
	sphi *= sphi;
	return(en.s0 * phi - cphi * (en.s1 + sphi*(en.s2
											   + sphi*(en.s3 + sphi*en.s4))));
}

float8 pl_inv_mlfn(float8 argphi, float es, float8 en) {
    float8 phi, sinPhi, cosPhi, t;
    float k = 1.f/(1.f-es);

    phi = argphi;
    int iter = I_ITER;
    do {
        sinPhi = sincos(phi, &cosPhi);
        t = 1.f - es * sinPhi * sinPhi;
        t = (pl_mlfn(phi, sinPhi, cosPhi, en) - argphi) * (t * sqrt(t)) * k;
        phi -= t;
        if (all(fabs(t) < ITOL))
            break;
    } while(--iter);

    return phi;
}

float8 pl_msfn(float8 sinphi, float8 cosphi, float es) {
	return (cosphi / sqrt(1.f - es * sinphi * sinphi));
}

float8 pl_tsfn(float8 phi, float8 sinphi, float e) {
	sinphi *= e;
	return (tan(.5f * (M_PI_2F - phi)) /
		 pow((1.f - sinphi) / (1.f + sinphi), (float8)(.5f * e))
		);
}

float8 pl_qsfn(float8 sinphi, float e, float one_es) {
	float8 con;

	if (e < EPS7)
		return (sinphi + sinphi);

	con = e * sinphi;
	return (one_es * (sinphi / (1.f - con * con) -
			(.5f / e) * log((1.f - con) / (1.f + con))));
}

float8 pl_phi2(float8 ts, float8 e) {
	float8 eccnth, Phi, con, dphi;
	int i;
	
	eccnth = .5f * e;
	Phi = M_PI_2F - 2.f * atan(ts);
	for (i = I_ITER; i; --i) {
		con = e * sin(Phi);
		dphi = M_PI_2F - 2.f * atan(ts * pow((1.f - con) / (1.f + con), eccnth)) - Phi;
		Phi += dphi;
		if (all(fabs(dphi) <= ITOL))
			break;
	}
	
	return Phi;
}

float8 pl_mod_pi(float8 phi) {
	return select(phi, phi - copysign((float8)(2.f*M_PIF), phi), fabs(phi) > M_PIF);
}

float4 pl_interpolate_cubic4(float X, float4 A, float4 B, float4 C, float4 D) {
    return B + 0.5f * X * (C - A + X * (2.f * A - 5.f * B + 4.f * C - D + X * (3.f * (B - C) + D - A)));
}


