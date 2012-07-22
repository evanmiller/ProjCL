
#define V(C,z) (C.s0 + z * (C.s1 + z * (C.s2 + z * C.s3)))
#define DV(C,z) (C.s1 + z * (C.s2 + C.s2 + z * 3. * C.s3))
/* note: following terms based upon 5 deg. intervals in degrees. */
__constant float4 ROBINSON_X[] = {
(float4)(1,      -5.67239e-12,   -7.15511e-05,   3.11028e-06),
(float4)(0.9986, -0.000482241,   -2.4897e-05,    -1.33094e-06),
(float4)(0.9954, -0.000831031,   -4.4861e-05,    -9.86588e-07),
(float4)(0.99,   -0.00135363,    -5.96598e-05,   3.67749e-06),
(float4)(0.9822, -0.00167442,    -4.4975e-06,    -5.72394e-06),
(float4)(0.973,  -0.00214869,    -9.03565e-05,   1.88767e-08),
(float4)(0.96,   -0.00305084,    -9.00732e-05,   1.64869e-06),
(float4)(0.9427, -0.00382792,    -6.53428e-05,   -2.61493e-06),
(float4)(0.9216, -0.00467747,    -0.000104566,   4.8122e-06),
(float4)(0.8962, -0.00536222,    -3.23834e-05,   -5.43445e-06),
(float4)(0.8679, -0.00609364,    -0.0001139,     3.32521e-06),
(float4)(0.835,  -0.00698325,    -6.40219e-05,   9.34582e-07),
(float4)(0.7986, -0.00755337,    -5.00038e-05,   9.35532e-07),
(float4)(0.7597, -0.00798325,    -3.59716e-05,   -2.27604e-06),
(float4)(0.7186, -0.00851366,    -7.0112e-05,    -8.63072e-06),
(float4)(0.6732, -0.00986209,    -0.000199572,   1.91978e-05),
(float4)(0.6213, -0.010418,      8.83948e-05,    6.24031e-06),
(float4)(0.5722, -0.00906601,    0.000181999,    6.24033e-06),
(float4)(0.5322, 0.,0.,0.)  
};

__constant float4 ROBINSON_Y[] = {
(float4)(0,      0.0124, 3.72529e-10,    1.15484e-09),
(float4)(0.062,  0.0124001,      1.76951e-08,    -5.92321e-09),
(float4)(0.124,  0.0123998,      -7.09668e-08,   2.25753e-08),
(float4)(0.186,  0.0124008,      2.66917e-07,    -8.44523e-08),
(float4)(0.248,  0.0123971,      -9.99682e-07,   3.15569e-07),
(float4)(0.31,   0.0124108,      3.73349e-06,    -1.1779e-06),
(float4)(0.372,  0.0123598,      -1.3935e-05,    4.39588e-06),
(float4)(0.434,  0.0125501,      5.20034e-05,    -1.00051e-05),

/* This should be 0.4958, and the other values should be
   updated accordingly. Proj.4 fixed the first column but not
   the others, and so their projection is inconsistent.
   I can't find the source of the other coefficients,
   so for now we use the wrong (but consistent) coefficients. */
(float4)(0.4968, 0.0123198,      -9.80735e-05,   9.22397e-06),
(float4)(0.5571, 0.0120308,      4.02857e-05,    -5.2901e-06),
(float4)(0.6176, 0.0120369,      -3.90662e-05,   7.36117e-07),
(float4)(0.6769, 0.0117015,      -2.80246e-05,   -8.54283e-07),
(float4)(0.7346, 0.0113572,      -4.08389e-05,   -5.18524e-07),
(float4)(0.7903, 0.0109099,      -4.86169e-05,   -1.0718e-06),
(float4)(0.8435, 0.0103433,      -6.46934e-05,   5.36384e-09),
(float4)(0.8936, 0.00969679,     -6.46129e-05,   -8.54894e-06),
(float4)(0.9394, 0.00840949,     -0.000192847,   -4.21023e-06),
(float4)(0.9761, 0.00616525,     -0.000256001,   -4.21021e-06),
(float4)(1., 0.,0.,0)
 };


__kernel void pl_project_robinson_s(
	__global float2 *xy_in,
	__global float2 *xy_out,
	const unsigned int count,

	float scale,
    float x0,
    float y0
) {
	int i = get_global_id(0);
	
	float lambda = radians(xy_in[i].even);
	float phi    = radians(xy_in[i].odd);
	
	int index;
	float x, y;

	float dphi = fabs(phi);
	index = floor(dphi * C1);
	index = select(index, NODES - 1, index >= NODES);
	dphi = degrees(dphi - RC1 * index);
	x = V(ROBINSON_X[index], dphi) * FXC * lambda;
	y = V(ROBINSON_Y[index], dphi) * FYC;
	y = select(y, -y, phi < 0.f);
	
	xy_out[i].even = x0 + scale * x;
	xy_out[i].odd = y0 + scale * y;
}

__kernel void pl_unproject_robinson_s(
	__global float2 *xy_in,
	__global float2 *xy_out,
	const unsigned int count,

	float scale,
    float x0,
    float y0
) {
	int i = get_global_id(0);
	
	float x = (xy_in[i].even - x0) / scale;
	float y = (xy_in[i].odd - y0) / scale;
	
	float lambda, phi;
	
	int index;
	float t, t1;
	float4 T;

	lambda = x / FXC;
	phi    = fabs(y / FYC);
	
	if (phi >= 1.) {
		phi = M_PI_2F;
		lambda /= ROBINSON_X[NODES].s0;
	} else {
		/* general problem */
				/* in Y space, reduce to table interval */
		for (index = floor(phi * NODES);;) {
			if (ROBINSON_Y[index].s0 > phi) --index;
			else if (ROBINSON_Y[index+1].s0 <= phi) ++index;
			else break;
		}
		T = ROBINSON_Y[index];
		/* first guess, linear interp */
		t = 5. * (phi - T.s0)/(ROBINSON_Y[index+1].s0 - T.s0);
		/* make into root */
		T.s0 -= phi;
		for (;;) { /* Newton-Raphson reduction */
			t -= t1 = V(T,t) / DV(T,t);
			if (fabs(t1) < EPS6)
				break;
		}
		phi = radians(5 * index + t);
		lambda /= V(ROBINSON_X[index], t);
	}
	phi = select(phi, -phi, y < 0.f);
	
	xy_out[i].even = degrees(lambda);
	xy_out[i].odd = degrees(phi);
}
