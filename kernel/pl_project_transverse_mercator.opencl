
/* See Karney, Transverse Mercator with an accuracy of a few nanometers (2011)
 * https://arxiv.org/pdf/1002.1417.pdf */

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

    float8 sin2, sin4, sin6, sin8;
    float8 cos2, cos4, cos6, cos8;
    float8 sinh2, sinh4, sinh6, sinh8;
    float8 cosh2, cosh4, cosh6, cosh8;

    f = 1.f - sqrt(one_ecc2);
    n = f / (2.f - f);

    sinLambda = sincos(lambda, &cosLambda);
    
    tau = tan(phi);
    sigma = sinh(ecc * atanh(ecc * tau / hypot(1.f, tau)));

    tau1 = tau * hypot(1.f, sigma) - sigma * hypot(1.f, tau);

    xi = atan2(tau1, cosLambda);
    eta = asinh(sinLambda / hypot(tau1, cosLambda));

    sin2 = sincos(2.f * xi, &cos2);

    sin4 = 2.f * sin2 * cos2;
    cos4 = 2.f * cos2 * cos2 - 1.f;

    sin6 = sin4 * cos2 + cos4 * sin2;
    cos6 = cos4 * cos2 - sin4 * sin2;

    sin8 = 2.f * sin4 * cos4;
    cos8 = 2.f * cos4 * cos4 - 1.f;

    sinh2 = sinh(2.f * eta);
    cosh2 = cosh(2.f * eta);

    sinh4 = 2.f * sinh2 * cosh2;
    cosh4 = 2.f * cosh2 * cosh2 - 1.f;

    sinh6 = sinh4 * cosh2 + cosh4 * sinh2;
    cosh6 = cosh4 * cosh2 + sinh4 * sinh2;

    sinh8 = 2.f * sinh4 * cosh4;
    cosh8 = 2.f * cosh4 * cosh4 - 1.f;

    y = xi;
    y += kruger.s0 * sin2 * cosh2;
    y += kruger.s1 * sin4 * cosh4;
    y += kruger.s2 * sin6 * cosh6;
    y += kruger.s3 * sin8 * cosh8;

    x = eta;
    x += kruger.s0 * cos2 * sinh2;
    x += kruger.s1 * cos4 * sinh4;
    x += kruger.s2 * cos6 * sinh6;
    x += kruger.s3 * cos8 * sinh8;

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

    float8 sin2, sin4, sin6, sin8;
    float8 cos2, cos4, cos6, cos8;
    float8 sinh2, sinh4, sinh6, sinh8;
    float8 cosh2, cosh4, cosh6, cosh8;

    sin2 = sincos(2.f * y, &cos2);

    sin4 = 2.f * sin2 * cos2;
    cos4 = 2.f * cos2 * cos2 - 1.f;

    sin6 = sin4 * cos2 + cos4 * sin2;
    cos6 = cos4 * cos2 - sin4 * sin2;

    sin8 = 2.f * sin4 * cos4;
    cos8 = 2.f * cos4 * cos4 - 1.f;

    sinh2 = sinh(2.f * x);
    cosh2 = cosh(2.f * x);

    sinh4 = 2.f * sinh2 * cosh2;
    cosh4 = 2.f * cosh2 * cosh2 - 1.f;

    sinh6 = sinh4 * cosh2 + cosh4 * sinh2;
    cosh6 = cosh4 * cosh2 + sinh4 * sinh2;

    sinh8 = 2.f * sinh4 * cosh4;
    cosh8 = 2.f * cosh4 * cosh4 - 1.f;

    xi = y;
    xi -= kruger.s4 * sin2 * cosh2;
    xi -= kruger.s5 * sin4 * cosh4;
    xi -= kruger.s6 * sin6 * cosh6;
    xi -= kruger.s7 * sin8 * cosh8;

    eta = x;
    eta -= kruger.s4 * cos2 * sinh2;
    eta -= kruger.s5 * cos4 * sinh4;
    eta -= kruger.s6 * cos6 * sinh6;
    eta -= kruger.s7 * cos8 * sinh8;

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
