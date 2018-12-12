
#define V(C,z) (C.s0 + z * (C.s1 + z * (C.s2 + z * C.s3)))
#define DV(C,z) (C.s1 + z * (C.s2 + C.s2 + z * 3.f * C.s3))
/* note: following terms based upon 5 deg. intervals in degrees. */
/* Source: http://article.gmane.org/gmane.comp.gis.proj-4.devel/6039 */
__constant float4 ROBINSON_X[] = {
(float4)(1, 2.2199e-17, -7.15515e-05, 3.1103e-06),
(float4)(0.9986, -0.000482243, -2.4897e-05, -1.3309e-06),
(float4)(0.9954, -0.00083103, -4.48605e-05, -9.86701e-07),
(float4)(0.99, -0.00135364, -5.9661e-05, 3.6777e-06),
(float4)(0.9822, -0.00167442, -4.49547e-06, -5.72411e-06),
(float4)(0.973, -0.00214868, -9.03571e-05, 1.8736e-08),
(float4)(0.96, -0.00305085, -9.00761e-05, 1.64917e-06),
(float4)(0.9427, -0.00382792, -6.53386e-05, -2.6154e-06),
(float4)(0.9216, -0.00467746, -0.00010457, 4.81243e-06),
(float4)(0.8962, -0.00536223, -3.23831e-05, -5.43432e-06),
(float4)(0.8679, -0.00609363, -0.000113898, 3.32484e-06),
(float4)(0.835, -0.00698325, -6.40253e-05, 9.34959e-07),
(float4)(0.7986, -0.00755338, -5.00009e-05, 9.35324e-07),
(float4)(0.7597, -0.00798324, -3.5971e-05, -2.27626e-06),
(float4)(0.7186, -0.00851367, -7.01149e-05, -8.6303e-06),
(float4)(0.6732, -0.00986209, -0.000199569, 1.91974e-05),
(float4)(0.6213, -0.010418, 8.83923e-05, 6.24051e-06),
(float4)(0.5722, -0.00906601, 0.000182, 6.24051e-06),
(float4)(0.5322, -0.00677797, 0.000275608, 6.24051e-06)
};

__constant float4 ROBINSON_Y[] = {
(float4)(-5.20417e-18, 0.0124, 1.21431e-18, -8.45284e-11),
(float4)(0.062, 0.0124, -1.26793e-09, 4.22642e-10),
(float4)(0.124, 0.0124, 5.07171e-09, -1.60604e-09),
(float4)(0.186, 0.0123999, -1.90189e-08, 6.00152e-09),
(float4)(0.248, 0.0124002, 7.10039e-08, -2.24e-08),
(float4)(0.31, 0.0123992, -2.64997e-07, 8.35986e-08),
(float4)(0.372, 0.0124029, 9.88983e-07, -3.11994e-07),
(float4)(0.434, 0.0123893, -3.69093e-06, -4.35621e-07),
(float4)(0.4958, 0.0123198, -1.02252e-05, -3.45523e-07),
(float4)(0.5571, 0.0121916, -1.54081e-05, -5.82288e-07),
(float4)(0.6176, 0.0119938, -2.41424e-05, -5.25327e-07),
(float4)(0.6769, 0.011713, -3.20223e-05, -5.16405e-07),
(float4)(0.7346, 0.0113541, -3.97684e-05, -6.09052e-07),
(float4)(0.7903, 0.0109107, -4.89042e-05, -1.04739e-06),
(float4)(0.8435, 0.0103431, -6.4615e-05, -1.40374e-09),
(float4)(0.8936, 0.00969686, -6.4636e-05, -8.547e-06),
(float4)(0.9394, 0.00840947, -0.000192841, -4.2106e-06),
(float4)(0.9761, 0.00616527, -0.000256, -4.2106e-06),
(float4)(1, 0.00328947, -0.000319159, -4.2106e-06)
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
	index = index >= NODES ? NODES - 1 : index;
	dphi = degrees(dphi - RC1 * index);
	x = V(ROBINSON_X[index], dphi) * FXC * lambda;
	y = V(ROBINSON_Y[index], dphi) * FYC;
	y = copysign(y, phi);
	
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
	phi = copysign(phi, y);
	
	xy_out[i].even = degrees(lambda);
	xy_out[i].odd = degrees(phi);
}
