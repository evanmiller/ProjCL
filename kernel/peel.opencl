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

#define AMERICAN_POLYCONIC_N_ITER 6
#define ALBERS_EQUAL_AREA_N_ITER  6
#define OBLIQUE_STEREOGRAPHIC_N_ITER 6
#define WINKEL_TRIPEL_N_ITER 4

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

float8 pl_qsfn(float8 sinphi, float e, float one_es);
float8 pl_phi2(float8 log_ts, float e);
float8 pl_mod_pi(float8 phi);
float4 pl_interpolate_cubic4(float X, float4 A, float4 B, float4 C, float4 D);

float8 pl_qsfn(float8 sinphi, float e, float one_es) {
	float8 con = e * sinphi;
	return one_es * (sinphi / (1.f - con * con) - (.5f / e) * (log1p(-con) - log1p(con)));
}

float8 pl_phi2(float8 log_ts, float e) {
	float8 eccnth, Phi, con, dphi;
	int i;
	
	eccnth = .5f * e;
	Phi = -asin(tanh(log_ts));
	for (i = I_ITER; i; --i) {
		con = e * sin(Phi);
		dphi = -asin(tanh(log_ts + eccnth * (log1p(-con)-log1p(con)))) - Phi;
		Phi += dphi;
		if (all(fabs(dphi) <= ITOL))
			break;
	}
	
	return Phi;
}

float8 pl_mod_pi(float8 phi) {
	return select(phi, phi - copysign(2.f*M_PIF, phi), fabs(phi) > M_PIF);
}

float4 pl_interpolate_cubic4(float X, float4 A, float4 B, float4 C, float4 D) {
    return B + 0.5f * X * (C - A + X * (2.f * A - 5.f * B + 4.f * C - D + X * (3.f * (B - C) + D - A)));
}


