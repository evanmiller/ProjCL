
/* Adapted from FORTRAN code in Cengizhan Ipbuker and I. Ozturg Bildirici, 
 * "Computer Program for the Inverse Transformation of the Winkel Projection"
 * Journal of Survey Engineering, Vol. 131 No. 4 (2005) pp. 125-129
 *
 * Modified to use a better initial guess for longitude.
 * And to fix a stupid bug in df1lm in their program.
 */

__kernel void pl_project_winkel_tripel_s(
    __global float16 *xy_in,
    __global float16 *xy_out,
    const unsigned int count,

    float scale, float x0, float y0,
    float lambda0,                                                   
    float cosphi1)
{
    int i = get_global_id(0);
    
    float8 lambda = radians(xy_in[i].even) - lambda0;
    float8 phi    = radians(xy_in[i].odd);
    
    float8 x, y, lambda2, c, d;
    
    float8 cosPhi, sinPhi, cosLambda2, sinLambda2;
    
    lambda2 = .5f * lambda;
    sinPhi = sincos(phi, &cosPhi);
    sinLambda2 = sincos(lambda2, &cosLambda2);
    
    c = sqrt(1.f - cosPhi * cosPhi * cosLambda2 * cosLambda2);
    d = acos(cosPhi * cosLambda2);

    x = select(0.f, .5f * (2.f * d * cosPhi * sinLambda2 / c + lambda * cosphi1), d != 0.f);
    y = select(0.f, .5f * (d * sinPhi / c + phi), d != 0.f);
    
    xy_out[i].even = x0 + scale * x;
    xy_out[i].odd = y0 + scale * y;
}

__kernel void pl_unproject_winkel_tripel_s(
    __global float16 *xy_in,
    __global float16 *xy_out,
    const unsigned int count,

    float scale, float x0, float y0,
    float lambda0,                                                   
    float cosphi1)
{
    int i = get_global_id(0);
    
    float8 x = (xy_in[i].even - x0) / scale;
    float8 y = (xy_in[i].odd - y0) / scale;
    
    float8 phi = y;
    float8 cosPhi;
    float8 sinPhi = sincos(phi, &cosPhi);
    float8 lambda = 2.f * x / (cosPhi + cosphi1);
    
    float8 f1, f2;
    float8 c, d;
    float8 invC12, invC;
    float8 sinLambda; // float8 cosLambda;
    float8 sinLambda2, cosLambda2; 
    float8 sin2Phi;
    float8 df1phi, df1lam, df2phi, df2lam;
    float8 dPhi, dLam;
    /*
    float8 ddPhi, ddLam;
    float8 invC32, invC2, invC52;
    float8 df1phiphi, df1lamphi, df1lamlam;
    float8 df2phiphi, df2lamphi, df2lamlam;
    float8 vector1, vector2; */
    float8 invDet;
    float8 dInvC32;
    
    int iter = 4;
    do {
        sin2Phi = 2.f * sinPhi * cosPhi;
        sinLambda2 = sincos(.5f * lambda, &cosLambda2);
        sinLambda = 2.f * sinLambda2 * cosLambda2;

        d = acos(cosPhi*cosLambda2);
        c = 1.f - cosPhi * cosPhi * cosLambda2 * cosLambda2;
        invC = 1.f / c;
        invC12 = 1.f / sqrt(c);
        dInvC32 = d * invC * invC12;
        f1 = d * cosPhi * sinLambda2 * invC12 + .5f * lambda * cosphi1 - x;
        f2 = .5f * d * sinPhi * invC12 + .5f * phi - y;
        
        df1phi = .25f * sinLambda * sin2Phi * invC 
            - dInvC32 * sinPhi * sinLambda2;
        df1lam = .5f * (cosPhi * cosPhi * sinLambda2 * sinLambda2 * invC
                        + dInvC32 * cosPhi * cosLambda2 * sinPhi * sinPhi
                        + cosphi1);
        
        df2phi = .5f * (sinPhi * sinPhi * cosLambda2 * invC
                        + dInvC32 * sinLambda2 * sinLambda2 * cosPhi
                        + 1.f);
        df2lam = .125f * (sin2Phi * sinLambda2 * invC
                          - dInvC32 * sinPhi * cosPhi * cosPhi * sinLambda);

        invDet = 1.f / (df1phi * df2lam - df2phi * df1lam);
        
        /* Newton's correction */
        dPhi = -(f1 * df2lam - f2 * df1lam) * invDet;
        dLam = -(f2 * df1phi - f1 * df2phi) * invDet;
        
        
        phi += dPhi;
        lambda += dLam; 
        
        /* Halley's correction... broken? */
        
        
        /*
        df1phiphi = .5f * sinLambda * invC * (1.f - 3.f * sinPhi * sinPhi * invC)
        + dInvC32 * sinLambda2 * cosPhi * (2.f - 3.f * sinLambda2 * sinLambda2 * invC);
        
        df1lamphi = .25f * sin2Phi * invC * (1.f - 3.f * sinLambda2 * sinLambda2 * invC)
        + .5f * dInvC32 * sinPhi * cosLambda2 * (2.f - 3.f * sinPhi * sinPhi * invC);
        
        df1lamlam = .09375f * sin2Phi * sin2Phi * sinLambda * invC * invC 
        + .125f * dInvC32 * sinLambda2 * sinPhi * sin2Phi * (2.f - 3.f * invC);
        
        
        df2phiphi = .375f * sinLambda * sin2Phi * sinLambda2 * invC2
        + .5f * dInvC32 * sinPhi * sinLambda2 * sinLambda2 * (2.f - 3.f * invC);
        
        df2lamphi = .25f * sinLambda2 * invC * (sinPhi * sinPhi + 1.f - 3.f * sinPhi * sinPhi * invC)
        - .125f * dInvC32 * cosPhi * sinLambda * (1.f - 3.f * sinPhi * sinPhi * invC);
        
        df2lamlam = -.0625f * sin2Phi * cosLambda2 * invC * (2.f - 3.f * sinPhi * sinPhi * invC)
        - .125f * dInvC32 * sinPhi * cosPhi * cosPhi * (sinLambda2 * sinLambda2 + 1.f - 3.f * sinLambda2 * sinLambda2 * invC);
        
        vector1 = df1phiphi * dPhi * dPhi + 2.f * df1lamphi * dLam * dPhi + df1lamlam * dLam * dLam;
        vector2 = df2phiphi * dPhi * dPhi + 2.f * df2lamphi * dLam * dPhi + df2lamlam * dLam * dLam;
        
        ddPhi = (vector1 * df2lam - vector2 * df1lam) * invDet;
        ddLam = (vector2 * df1phi - vector1 * df2phi) * invDet;
        
        
        phi += dPhi * dPhi / (dPhi + 0.5f * ddPhi);
        lambda += dLam * dLam / (dLam + 0.5f * ddLam);
         */

        if (!any(fabs(dPhi) > TOL6) && !any(fabs(dLam) > TOL6))
            break;
        
        sinPhi = sincos(phi, &cosPhi);
    } while (--iter);
    
//    printf("Iterations left: %d\n", iter);
        
    xy_out[i].even = degrees(lambda + lambda0);
    xy_out[i].odd = degrees(phi);
}
