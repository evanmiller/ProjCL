
__kernel void pl_project_transverse_mercator_s (
    __global float16 *xy_in,
    __global float16 *xy_out,
    const unsigned int count,

    float scale, float x0, float y0,
    float phi0,
    float lambda0,
    float ml0,
    float8 en
) {
    int i = get_global_id(0);
    
    float8 lambda = radians(xy_in[i].even) - lambda0;
    float8 phi    = radians(xy_in[i].odd);
    
    float8 x, y, b, cosPhi, sinLambda, cosLambda;
    
    cosPhi = cos(phi);
    sinLambda = sincos(lambda, &cosLambda);
    
    b = cosPhi * sinLambda;
    x = 0.5f * log((1.f + b) / (1.f - b));
    y = cosPhi * cosLambda / sqrt(1.f - b * b);
    y = select(acos(y), 0.f, fabs(y) >= 1.f);
    y = select(y, -y, phi < 0.f);
    y = y - phi0;
    
    xy_out[i].even = x0 + scale * x;
    xy_out[i].odd = y0 + scale * y;
}

__kernel void pl_unproject_transverse_mercator_s (
    __global float16 *xy_in,
    __global float16 *xy_out,
    const unsigned int count,

    float scale, float x0, float y0,
    float phi0,
    float lambda0,
    float ml0,
    float8 en
) {
    int i = get_global_id(0);

    float8 x = (xy_in[i].even - x0) / scale;
    float8 y = (xy_in[i].odd - y0) / scale;
    
    float8 h, g, phi, lambda;
    
    h = exp(x);
    g = 0.5f * (h - 1.f / h);
    h = cos(phi0 + y);
    phi = asin(sqrt((1.f - h * h) / (1.f + g * g)));
    phi = select(phi, -phi, y < 0.f);
    lambda = select(0.f, atan2(g, h), g != 0.f || h != 0.f);
    
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
    float phi0,
    float lambda0,
    float ml0,
    float8 en
) {
    int i = get_global_id(0);

    float8 lambda = radians(xy_in[i].even) - lambda0;
    float8 phi    = radians(xy_in[i].odd);

    float8 sinPhi, cosPhi, t, al, als, n, x, y;

    sinPhi = sincos(phi, &cosPhi);
    t = select(sinPhi/cosPhi, 0.f, fabs(cosPhi) < EPS7);
    t *= t;
    al = cosPhi * lambda;
    als = al * al;
    al /= sqrt(1.f - ecc2 * sinPhi * sinPhi);
    n = ecc2 / one_ecc2 * cosPhi * cosPhi;

    x = al * (FC1 +
            FC3 * als * (1.f - t + n +
                FC5 * als * (5.f + t * (t - 18.f) + n * (14.f - 58.f * t)
                    + FC7 * als * (61.f + t * ( t * (179.f - t) - 479.f ) )
                    )));
    y = (pl_mlfn(phi, sinPhi, cosPhi, en) - ml0 +
            sinPhi * al * lambda * FC2 * ( 1.f +
                FC4 * als * (5.f - t + n * (9.f + 4.f * n) +
                    FC6 * als * (61.f + t * (t - 58.f) + n * (270.f - 330.f * t)
                        + FC8 * als * (1385.f + t * ( t * (543.f - t) - 3111.f) )
                        ))));

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
    float phi0,
    float lambda0,
    float ml0,
    float8 en
) {
    int i = get_global_id(0);

    float8 x = (xy_in[i].even - x0) / scale;
    float8 y = (xy_in[i].odd - y0) / scale;

    float8 sinPhi, cosPhi, con, t, n, d, ds;

    float8 lambda, phi;

    phi = pl_inv_mlfn(ml0 + y, ecc2, en);

    sinPhi = sincos(phi, &cosPhi);

    t = select(0.f, sinPhi/cosPhi, fabs(cosPhi) > EPS7);
    n = ecc2 / one_ecc2 * cosPhi * cosPhi;
    con = 1.f - ecc2 * sinPhi * sinPhi;
    d = x * sqrt(con);
    con *= t;
    t *= t;
    ds = d * d;
    phi -= (con * ds / one_ecc2) * FC2 * (1.f - 
            ds * FC4 * (5.f + t * (3.f - 9.f * n) + n * (1.f - 4.f * n) -
                ds * FC6 * (61.f + t * (90.f - 252.f * n + 45.f * t) + 46.f * n
                    - ds * FC8 * (1385.f + t * (3633.f + t * (4095.f + 1574.f * t)))
                    )));
    lambda = d * (FC1 -
            ds * FC3 * (1.f + 2.f * t + n -
                ds * FC5 * (5.f + t * (28.f + 24.f * t + 8.f * n) + 6.f * n
                    - ds * FC7 * (61.f + t * (662.f + t * (1320.f + 720.f * t)))
                    ))) / cosPhi;

    xy_out[i].even = degrees(pl_mod_pi(lambda + lambda0));
    xy_out[i].odd  = degrees(phi);
}
