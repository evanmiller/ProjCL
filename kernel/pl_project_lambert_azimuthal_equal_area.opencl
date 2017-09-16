
__kernel void pl_project_lambert_azimuthal_equal_area_e(
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
    float qp,
    float sinB1,
    float cosB1,

    float rq,
    float4 apa,
    float dd,
    float xmf,
    float ymf
) {
    int i = get_global_id(0);

    float8 lambda = radians(xy_in[i].even) - lambda0;
    float8 phi    = radians(xy_in[i].odd);

    float8 x, y;
    float8 b, sinLambda, cosLambda, sinPhi, sinB, cosB;

    sinLambda = sincos(lambda, &cosLambda);
    sinPhi = sin(phi);

    sinB = pl_qsfn(sinPhi, ecc, one_ecc2) / qp;
    cosB = sqrt(1.f - sinB * sinB);

    b = sqrt(2.f / (1.f + sinB1 * sinB + cosB1 * cosB * cosLambda));

    x = xmf * b * cosB * sinLambda;
    y = ymf * b * (cosB1 * sinB - sinB1 * cosB * cosLambda);

    xy_out[i].even = x0 + scale * x;
    xy_out[i].odd  = y0 + scale * y;
}

__kernel void pl_project_lambert_azimuthal_equal_area_s(
    __global float16 *xy_in,
    __global float16 *xy_out,
    const unsigned int count,

    float scale,
    float x0,
    float y0,

    float phi0,
    float lambda0,
    float qp,
    float sinB1,
    float cosB1
) {
    int i = get_global_id(0);

    float8 lambda = radians(xy_in[i].even) - lambda0;
    float8 phi    = radians(xy_in[i].odd);

    float8 x, y;
    float8 b, sinLambda, cosLambda, sinPhi, cosPhi;

    sinLambda = sincos(lambda, &cosLambda);
    sinPhi = sincos(phi, &cosPhi);

    b = sqrt(2.f / (1.f + sinB1 * sinPhi + cosB1 * cosPhi * cosLambda));
    x = b * cosPhi * sinLambda;
    y = b * (cosB1 * sinPhi - sinB1 * cosPhi * cosLambda);

    xy_out[i].even = x0 + scale * x;
    xy_out[i].odd  = y0 + scale * y;
}
 
__kernel void pl_unproject_lambert_azimuthal_equal_area_e(
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
                                                          
    float qp,

    float sinB1,
    float cosB1,

    float rq,
    float4 apa,

    float dd,
    float xmf,
    float ymf
) {
    int i = get_global_id(0);

    float8 x = (xy_in[i].even - x0) / scale;
    float8 y = (xy_in[i].odd - y0) / scale;

    float8 lambda, phi;

    float8 cCe, sCe, rho, beta;
        
    x /= dd;
    y *= dd;

    rho = hypot(x, y);
    sCe = sincos(2.f * asin(0.5f * rho / rq), &cCe);
    x *= sCe;
    beta = asin(cCe * sinB1 + y * sCe * cosB1 / rho);
    y = rho * cosB1 * cCe - y * sinB1 * sCe;

    lambda = atan2(x, y);    
    
    phi = (beta + apa.s0 * sin(2.f * beta) + apa.s1 * sin(4.f * beta) + apa.s2 * sin(6.f * beta));
     
    xy_out[i].even = degrees(pl_mod_pi(lambda + lambda0));
    xy_out[i].odd = degrees(phi);
}


__kernel void pl_unproject_lambert_azimuthal_equal_area_s(
    __global float16 *xy_in,
    __global float16 *xy_out,
    const unsigned int count,

    float scale,
    float x0,
    float y0,

    float phi0,
    float lambda0,
    float qp,
    float sinB1,
    float cosB1
) {
    int i = get_global_id(0);

    float8 x = (xy_in[i].even - x0) / scale;
    float8 y = (xy_in[i].odd - y0) / scale;

    float8 lambda, phi;

    float8 sinZ, cosZ, rho;

    rho = hypot(x, y);
    phi = 2.f * asin(0.5f * rho);

    sinZ = sincos(phi, &cosZ);

    phi = select(asin(cosZ * sinB1 + y * sinZ * cosB1 / rho), 
            phi0, fabs(rho) <= EPS7);

    x *= sinZ * cosB1;
    y = (cosZ - sin(phi) * sinB1) * rho;

    lambda = select(atan2(x, y), 0.f, y == 0.f);

    xy_out[i].even = degrees(pl_mod_pi(lambda + lambda0));
    xy_out[i].odd = degrees(phi);
}
