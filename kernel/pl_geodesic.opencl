__kernel void pl_sincos(
	__global float8 *phi_in,
	__global float16 *sincos_out
) {
	int i = get_global_id(0);
	
	float8 phi = radians(phi_in[i]);
	float8 sinPhi, cosPhi;
	
	sinPhi = sincos(phi, &cosPhi);
	
	sincos_out[i].even = sinPhi;
	sincos_out[i].odd = cosPhi;
}

__kernel void pl_sincos1(
	__global float16 *phi_in,
	__global float16 *sincos_out
) {
	int i = get_global_id(0);
	
	float8 phi = radians(phi_in[i].odd);
	float8 sinPhi, cosPhi;
	
	sinPhi = sincos(phi, &cosPhi);
	
	sincos_out[i].even = sinPhi;
	sincos_out[i].odd = cosPhi;
}

__kernel void pl_inverse_geodesic_s(
	__global float2 *lp1_in,
	__global float16 *lp2_in,
	__global float8 *dist_out,
	
	float radius
) {
	int i = get_global_id(0);
	int j = get_global_id(1);
	
	int j_size = get_global_size(1);
	
	float lam1 = radians(lp1_in[i].even);
	float phi1 = radians(lp1_in[i].odd);
	
	float8 lam2 = radians(lp2_in[j].even);
	float8 phi2 = radians(lp2_in[j].odd);
	
	float cosPhi1 = cos(phi1);
	float8 cosPhi2 = cos(phi2);
	
    float8 dlam = lam2 - lam1;
    float8 dphi = phi2 - phi1;

    float8 shp2 = sin(0.5f * dphi);
    float8 shl2 = sin(0.5f * dlam);

    dist_out[i*j_size+j] = 2.f * radius * asin(sqrt(shp2 * shp2 + cosPhi1 * cosPhi2 * shl2 * shl2));
}

__kernel void pl_forward_geodesic_fixed_distance_s(
	__global float2 *lp_in,
	__global float2 *phi_sincos,
	__global float16 *az_sincos,
	__global float16 *lp_out,
	
	float distance,
	float sinDistance,
	float cosDistance
) {
	int i = get_global_id(0);
	int j = get_global_id(1);
	
	int j_size = get_global_size(1);
	
	float lam1 = radians(lp_in[i].s0);
	
	float sinPhi = phi_sincos[i].s0;
	float cosPhi = phi_sincos[i].s1;
	
	float8 sinAz = az_sincos[j].even;
	float8 cosAz = az_sincos[j].odd;
	
	float8 lam2, phi2;
	
	phi2 = asin(sinPhi * cosDistance + cosPhi * sinDistance * cosAz);
	lam2 = lam1 + atan2(sinDistance * sinAz, 
		cosPhi * cosDistance - sinPhi * sinDistance * cosAz);

	lp_out[i*j_size+j].even = degrees(pl_mod_pi(lam2));
	lp_out[i*j_size+j].odd = degrees(phi2);
}

__kernel void pl_forward_geodesic_fixed_angle_s(
	float2 lp_in,
	__global float8 *dist,
	__global float16 *lp_out,
	
	float azimuth,
	float sinAz,
	float cosAz
) {
	int i = get_global_id(0);
	
	float lam1 = radians(lp_in.s0);
	float phi1 = radians(lp_in.s1);
	
	float sinPhi, cosPhi;
    sinPhi = sincos(phi1, &cosPhi);
	
	float8 sinDistance, cosDistance;
    sinDistance = sincos(dist[i], &cosDistance);
	
	float8 lam2, phi2;
	
	phi2 = asin(sinPhi * cosDistance + cosPhi * sinDistance * cosAz);
	lam2 = lam1 + atan2(sinDistance * sinAz, 
		cosPhi * cosDistance - sinPhi * sinDistance * cosAz);
	
	lp_out[i].even = degrees(pl_mod_pi(lam2));
	lp_out[i].odd = degrees(phi2);
}

/* *** SOLUTION OF THE GEODETIC DIRECT PROBLEM AFTER T.VINCENTY */
/* *** MODIFIED RAINSFORD'S METHOD WITH HELMERT'S ELLIPTICAL TERMS */
/* *** EFFECTIVE IN ANY AZIMUTH AND AT ANY DISTANCE SHORT OF ANTIPODAL */

/* *** A IS THE SEMI-MAJOR AXIS OF THE REFERENCE ELLIPSOID */
/* *** F IS THE FLATTENING OF THE REFERENCE ELLIPSOID */
/* *** LATITUDES AND LONGITUDES IN RADIANS POSITIVE NORTH AND EAST */
/* *** AZIMUTHS IN RADIANS CLOCKWISE FROM NORTH */
/* *** GEODESIC DISTANCE S ASSUMED IN UNITS OF SEMI-MAJOR AXIS A */

/* *** PROGRAMMED FOR CDC-6600 BY LCDR L.PFEIFER NGS ROCKVILLE MD 20FEB75 */
/* *** MODIFIED FOR SYSTEM 360 BY JOHN G GERGEN NGS ROCKVILLE MD 750608 */

/* *** ...modified for OpenCL by evan miller (CHICAGO IL) */

/* kernel void pl_forward_geodesic_e(
	__global float2 *lp_in,
	__global float2 *phi_sincos,
	__global float16 *az_sincos,
	__global float16 *lp_out,
	
	float distance,
	float sinDistance,
	float cosDistance,
	
	float flattening
) {
	int i = get_global_id(0);
	int j = get_global_id(1);
	
	int j_size = get_global_size(1);
	
	float lam1 = lp_in[i].s0;
	float phi1 = lp_in[i].s1;
	
	float sinPhi = phi_sincos[i].s0;
	float cosPhi = phi_sincos[i].s1;
	
	float8 sinAz = az_sincos[j].even;
	float8 cosAz = az_sincos[j].odd;
	
	float8 az21, lam2, phi2;
	
	float r;

    float8 c, d, e, x, y, cu, cy, cz, sa, su, tu, sy, c2a;



    r = 1.f - flattening;
    tu = r * sinPhi / cosPhi;
	az21 = select(atan2(tu, cosAz) * 2.f, 0.f, cosAz == 0.f);
    cu = 1.f / sqrt(tu * tu + 1.f);
    su = tu * cu;
    sa = cu * sinAz;
    c2a = -sa * sa + 1.f;
    x = sqrt((1.f / r / r - 1.f) * c2a + 1.f) + 1.f;
    x = (x - 2.f) / x;
    c = 1.f - x;
    c = (x * x / 4.f + 1.f) / c;
    d = (x * .375f * x - 1.f) * x;
    tu = distance / r / c;
    y = tu;
    do {
        sy = sincos(y, &cy);
        cz = cos(az21 + y);
        e = cz * cz * 2.f - 1.f;
        c = y;
        x = e * cy;
        y = e + e - 1.f;
        y = (((sy * sy * 4.f - 3.f) * y * cz * d / 6.f + x) * d / 4.f - cz) *
            sy * d + tu;
    } while (any(fabs(y - c) > EPS7));
    az21 = cu * cy * cosAz - su * sy;
    phi2 = atan2(su * cy + cu * sy * cosAz, r * hypot(sa, az21));
    x = atan2(sy * sinAz, cu * cy - su * sy * cosAz);
    c = ((c2a * -3.f + 4.f) * flattening + 4.f) * c2a * flattening / 16.f;
    d = ((e * cy * c + cz) * sy * c + y) * sa;
    lam2 = lam1 + x - (1.f - c) * d * flattening;
	
//    az21 = atan2(sa, az21) + M_PIF;
	
	lp_out[i*j_size+j].even = lam2;
	lp_out[i*j_size+j].odd = phi2;
}
*/

