
__kernel void pl_project_transverse_mercator_s (
    __global float16 *xy_in,
    __global float16 *xy_out,
    const unsigned int count,

    float scale, float x0, float y0,
    float lambda0,
    float8 kruger
) {
    int i = get_global_id(0);
    
    float8 lambda = radians(xy_in[i].even) - lambda0;
    float8 phi    = radians(xy_in[i].odd);
    
    float8 x, y, tau, sinLambda, cosLambda;
    
    sinLambda = sincos(lambda, &cosLambda);
    
    tau = tan(phi);
    y = atan2(tau, cosLambda);
    x = asinh(sinLambda / hypot(tau, cosLambda));
    
    xy_out[i].even = x0 + scale * x;
    xy_out[i].odd = y0 + scale * y;
}

__kernel void pl_unproject_transverse_mercator_s (
    __global float16 *xy_in,
    __global float16 *xy_out,
    const unsigned int count,

    float scale, float x0, float y0,
    float lambda0,
    float8 kruger
) {
    int i = get_global_id(0);

    float8 x = (xy_in[i].even - x0) / scale;
    float8 y = (xy_in[i].odd - y0) / scale;
    
    float8 phi, lambda, sinhX, sinY, cosY;
    
    sinhX = sinh(x);
    sinY = sincos(y, &cosY);

    lambda = atan2(sinhX, cosY);
    phi = atan2(sinY, hypot(sinhX, cosY));
    
    xy_out[i].even = degrees(pl_mod_pi(lambda + lambda0));
    xy_out[i].odd  = degrees(phi);
}

__kernel void pl_project_transverse_mercator_e (
    __global float16 *xy_in,
    __global float16 *xy_out,
    const unsigned int count,

    float ecc,
    float ecc2,
    float one_ecc2,

    float scale, float x0, float y0,
    float lambda0,
    float8 kruger
) {
    int i = get_global_id(0);

    float8 lambda = radians(xy_in[i].even) - lambda0;
    float8 phi    = radians(xy_in[i].odd);

    float8 x, y, xi, eta, tau, tau1, sigma, sinLambda, cosLambda;
    float f, n;

    f = 1.f - sqrt(one_ecc2);
    n = f / (2.f - f);

    sinLambda = sincos(lambda, &cosLambda);
    
    tau = tan(phi);
    sigma = sinh(ecc * atanh(ecc * tau / hypot(1.f, tau)));

    tau1 = tau * hypot(1.f, sigma) - sigma * hypot(1.f, tau);

    xi = atan2(tau1, cosLambda);
    eta = asinh(sinLambda / hypot(tau1, cosLambda));

    y = xi;
    y += kruger.s0 * sin(2.f * xi) * cosh(2.f * eta);
    y += kruger.s1 * sin(4.f * xi) * cosh(4.f * eta);
    y += kruger.s2 * sin(6.f * xi) * cosh(6.f * eta);
    y += kruger.s3 * sin(8.f * xi) * cosh(8.f * eta);

    x = eta;
    x += kruger.s0 * cos(2.f * xi) * sinh(2.f * eta);
    x += kruger.s1 * cos(4.f * xi) * sinh(4.f * eta);
    x += kruger.s2 * cos(6.f * xi) * sinh(6.f * eta);
    x += kruger.s3 * cos(8.f * xi) * sinh(8.f * eta);

    xy_out[i].even = x0 + scale * x;
    xy_out[i].odd  = y0 + scale * y;
}

__kernel void pl_unproject_transverse_mercator_e (
    __global float16 *xy_in,
    __global float16 *xy_out,
    const unsigned int count,

    float ecc,
    float ecc2,
    float one_ecc2,

    float scale, float x0, float y0,
    float lambda0,
    float8 kruger
) {
    int i = get_global_id(0);

    float8 x = (xy_in[i].even - x0) / scale;
    float8 y = (xy_in[i].odd - y0) / scale;

    float8 phi, lambda, sinhX, sinY, cosY;
    float8 xi, eta;
    float8 tau0, sigma0, tau0p, dtau0;

    xi = y;
    xi -= kruger.s4 * sin(2.f * y) * cosh(2.f * x);
    xi -= kruger.s5 * sin(4.f * y) * cosh(4.f * x);
    xi -= kruger.s6 * sin(6.f * y) * cosh(6.f * x);
    xi -= kruger.s7 * sin(8.f * y) * cosh(8.f * x);

    eta = x;
    eta -= kruger.s4 * cos(2.f * y) * sinh(2.f * x);
    eta -= kruger.s5 * cos(4.f * y) * sinh(4.f * x);
    eta -= kruger.s6 * cos(6.f * y) * sinh(6.f * x);
    eta -= kruger.s7 * cos(8.f * y) * sinh(8.f * x);

    sinY = sincos(xi, &cosY);
    sinhX = sinh(eta);

    /* Newton's method (1 iteration) */
    tau0 = sinY / hypot(sinhX, cosY);
    sigma0 = sinh(ecc * tanh(ecc * tau0 / hypot(1.f, tau0)));
    tau0p = tau0 * hypot(1.f, sigma0) - sigma0 * hypot(1.f, tau0);
    dtau0 = (tau0 - tau0p) / hypot(1.f, tau0p) * (1.f + one_ecc2 * tau0 * tau0) / (one_ecc2 * hypot(1.f, tau0));

    lambda = atan2(sinhX, cosY);
    phi = atan(tau0 + dtau0);

    xy_out[i].even = degrees(pl_mod_pi(lambda + lambda0));
    xy_out[i].odd  = degrees(phi);
}
