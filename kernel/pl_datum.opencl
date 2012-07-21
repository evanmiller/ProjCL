
#define AD_C       1.0026000f /* Toms region 1 constant */


__kernel void pl_cartesian_apply_affine_transform(
    __global float8 *x_rw,
    __global float8 *y_rw,
    __global float8 *z_rw,
    float16 matrix
) {
    int i = get_global_id(0);

    float8 x = x_rw[i];
    float8 y = y_rw[i];
    float8 z = z_rw[i];
    
    x_rw[i] = matrix.lo.lo.x * x + matrix.lo.lo.y * y + matrix.lo.lo.z * z + matrix.lo.lo.w;
    y_rw[i] = matrix.lo.hi.x * x + matrix.lo.hi.y * y + matrix.lo.hi.z * z + matrix.lo.hi.w;
    z_rw[i] = matrix.hi.lo.x * x + matrix.hi.lo.y * y + matrix.hi.lo.z * z + matrix.hi.lo.w;
}

__kernel void pl_geodesic_to_cartesian(
    __global float16 *lp_in,
    __global float8 *x_out,
    __global float8 *y_out,
    __global float8 *z_out,
                                    
    float ecc,
    float ecc2,
    float one_ecc2,
                                    
    float major_axis,
    float minor_axis
) {
    int i = get_global_id(0);
    
    float8 lam = radians(lp_in[i].even);
    float8 phi = radians(lp_in[i].odd);
    
    float8 x, y, z, r;
    
    float8 sinPhi, cosPhi, sinLambda, cosLambda;
    
    sinPhi = sincos(phi, &cosPhi);
    sinLambda = sincos(lam, &cosLambda);
    
    r = major_axis / sqrt(1.f - ecc2 * sinPhi * sinPhi);
    x = r * cosPhi * cosLambda;
    y = r * cosPhi * sinLambda;
    z = r * one_ecc2 * sinPhi;
    
    x_out[i] = x;
    y_out[i] = y;
    z_out[i] = z;
}

__kernel void pl_cartesian_to_geodesic(
    __global float8 *x_in,
    __global float8 *y_in,
    __global float8 *z_in,
    __global float16 *lp_out,
                                    
    float ecc,
    float ecc2,
    float one_ecc2,

    float major_axis,
    float minor_axis                                    
) {
    int i = get_global_id(0);

    float8 X = x_in[i];
    float8 Y = y_in[i];
    float8 Z = z_in[i];
    /*
     * The method used here is derived from 'An Improved Algorithm for
     * Geocentric to Geodetic Coordinate Conversion', by Ralph Toms, Feb 1996
     */
    
    /* Note: Variable names follow the notation used in Toms, Feb 1996 */
    
    float8 W;        /* distance from Z axis */
    float8 W2;       /* square of distance from Z axis */
    float8 T0;       /* initial estimate of vertical component */
    float8 T1;       /* corrected estimate of vertical component */
    float8 S0;       /* initial estimate of horizontal component */
    float8 S1;       /* corrected estimate of horizontal component */
    float8 Sin_B0;   /* sin(B0), B0 is estimate of Bowring aux variable */
    float8 Sin3_B0;  /* cube of sin(B0) */
    float8 Cos_B0;   /* cos(B0) */
    float8 Sin_p1;   /* sin(phi1), phi1 is estimated latitude */
    float8 Cos_p1;   /* cos(phi1) */
    float8 Sum;      /* numerator of cos(phi1) */
    
    float8 lambda, phi;
   
    lambda = select(select((float8)M_PI_2F, (float8)-M_PI_2F, Y <= 0.f), atan2(Y, X), X != 0.f);

    W2 = X*X + Y*Y;
    W = sqrt(W2);
    T0 = Z * AD_C;
    S0 = sqrt(T0 * T0 + W2);
    Sin_B0 = T0 / S0;
    Cos_B0 = W / S0;
    Sin3_B0 = Sin_B0 * Sin_B0 * Sin_B0;
    T1 = Z + minor_axis * ecc2 / one_ecc2 * Sin3_B0;
    Sum = W - major_axis * ecc2 * Cos_B0 * Cos_B0 * Cos_B0;
    S1 = sqrt(T1*T1 + Sum * Sum);
    Sin_p1 = T1 / S1;
    Cos_p1 = Sum / S1;
    
    phi = atan(Sin_p1 / Cos_p1);
    
    lp_out[i].even = degrees(lambda);
    lp_out[i].odd = degrees(phi);
}
