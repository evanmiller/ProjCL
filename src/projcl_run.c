/*
 *  projcl_run.c
 *  Magic Maps
 *
 *  Created by Evan Miller on 2/12/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include <projcl/projcl.h>
#include "projcl_run.h"
#include "projcl_util.h"
#include "projcl_spheroid.h"

#include <math.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif
#ifdef __linux__
#include "cblas.h"
typedef int __CLPK_integer;
typedef float __CLPK_real;
#endif

#define RAD_TO_DEG	57.29577951308232
#define DEG_TO_RAD	.0174532925199432958

#define EPS7 1.e-7

#define P00 .33333333333333333333
#define P01 .17222222222222222222
#define P02 .10257936507936507936
#define P10 .06388888888888888888
#define P11 .06640211640211640211
#define P20 .01641501294219154443


#define SEC_TO_RAD 4.84813681109535993589914102357e-6

struct pl_datum_info {
    float dx;
    float dy;
    float dz;
    float ex;
    float ey;
    float ez;
    float ppm;
};

/* Source: "WGS 84 Implementation Manual" */
static struct pl_datum_info pl_datum_params[] = {
    /*    Dx       Dy        Dz      Ex        Ey       Ez       m   */
    
    { /* PL_DATUM_WGS_84 */
          0.0,     0.0,    0.0,     0.0,     0.0,     0.0,    0.0     },
    { /* PL_DATUM_WGS_72 */
          0.0,     0.0,    4.5,     0.0,     0.0,    -0.554,  0.22    },
    { /* PL_DATUM_ED_50 */
        -87.0,   -98.0,  -121.0,    0.0,     0.0,     0.0,    0.0     },
    { /* PL_DATUM_ED_79 */
        -86.0,   -98.0,  -119.0,    0.0,     0.0,     0.0,    0.0     },
    { /* PL_DATUM_ED_87 */
        -82.5,   -91.7,  -117.7,    0.1338, -0.0625, -0.047,  0.045   },
    { /* PL_DATUM_AUSTRIA_NS */
        595.6,    87.3,   473.3,    4.7994,  0.0671,  5.7850, 2.555   },
    { /* PL_DATUM_BELGIUM_50 */
        -55.0,    49.0,  -158.0,    0.0,     0.0,     0.0,    0.0     },
    { /* PL_DATUM_BERNE_1873 */
        649.0,     9.0,   376.0,    0.0,     0.0,     0.0,    0.0     },
    { /* PL_DATUM_CH_1903 */
        660.1,    13.1,   369.2,    0.8048,  0.5777,  0.9522, 5.660   },
    { /* PL_DATUM_DANISH_GI_1934 */
        662.0,    18.0,   734.0,    0.0,     0.0,     0.0,    0.0     },
    { /* PL_DATUM_NOUV_TRIG_DE_FRANCE_GREENWICH */
       -168.0,   -60.0,   320.0,    0.0,     0.0,     0.0,    0.0     },
    { /* PL_DATUM_NOUV_TRIG_DE_FRANCE_PARIS */
       -168.0,   -60.0,   320.0,    0.0,     0.0,  8414.03,   0.0     },
    { /* PL_DATUM_POTSDAM */
        587.0,    16.0,   393.0,    0.0,     0.0,     0.0,    0.0     },
    { /* PL_DATUM_GGRS_87 */
        199.6,   -75.1,  -246.3,    0.0202,  0.0034,  0.0135, -0.015  },
    { /* PL_DATUM_HJORSEY_55 */
        -73.0,    46.0,   -86.0,    0.0,     0.0,     0.0,    0.0     },
    { /* PL_DATUM_IRELAND_65 */
        506.0,  -122.0,   611.0,    0.0,     0.0,     0.0,    0.0     },
    { /* PL_DATUM_ITALY_1940 */
       -133.0,   -50.0,    97.0,    0.0,     0.0, 44828.40,   0.0     },
    { /* PL_DATUM_NOUV_TRIG_DE_LUX */
       -262.0,    75.0,    25.0,    0.0,     0.0,     0.0,    0.0     },
    { /* PL_DATUM_NETHERLANDS_1921 */
        719.0,    47.0,   640.0,    0.0,     0.0,     0.0,    0.0     },
    { /* PL_DATUM_OSGB_36 */
        375.0,  -111.0,   431.0,    0.0,     0.0,     0.0,    0.0     },
    { /* PL_DATUM_PORTUGAL_DLX */
        504.1,  -220.9,   563.0,    0.0,     0.0,    -0.554,  0.220   },
    { /* PL_DATUM_PORTUGAL_1973 */
        227.0,    97.5,    35.4,    0.0,     0.0,    -0.554,  0.220   },
    { /* PL_DATUM_RNB_72 */
       -104.0,    80.0,   -75.0,    0.0,     0.0,     0.0,    0.0     },
    { /* PL_DATUM_RT_90 */
        424.3,   -80.5,   613.1,    4.3965, -1.9866,  5.1846, 0.0     },
    { /* PL_DATUM_NAD_27 */
         -8.0,   160.0,   176.0,    0.0,     0.0,     0.0,    0.0     },
    { /* PL_DATUM_NAD_83 */
          0.0,     0.0,     0.0,    0.0,     0.0,     0.0,    0.0     },
    { /* PL_DATUM_ETRS_89 */
          0.0,     0.0,     0.0,    0.0,     0.0,     0.0,    0.0     }
};

struct pl_matrix {
    float  elements[4][4];
};

static struct pl_matrix pl_affine_transform_make(float Rx, float Ry, float Rz, float M, float Dx, float Dy, float Dz) {
    /* store in column-major order */
    struct pl_matrix matrix = {
        .elements = {
            {      M,  M * Rz,  -M * Ry, 0.f },
            {-M * Rz,       M,   M * Rx, 0.f },
            { M * Ry, -M * Rx,        M, 0.f },
            {     Dx,      Dy,       Dz, 1.f }
        }
    };
    return matrix;
}


static float _pl_mlfn(float phi, float sphi, float cphi, float *en) {
	cphi *= sphi;
	sphi *= sphi;
	return(en[0] * phi - cphi * (en[1] + sphi*(en[2]
											   + sphi*(en[3] + sphi*en[4]))));
}

static float _pl_qsfn(float sinphi, float e, float one_es) {
	float con;
	
	if (e < EPS7)
		return (sinphi + sinphi);
	
	con = e * sinphi;
	return (one_es * (sinphi / (1.f - con * con) -
					  (.5f / e) * log((1.f - con) / (1.f + con))));
}

static float _pl_msfn(float sinphi, float cosphi, float es) {
	return (cosphi / sqrt(1.f - es * sinphi * sinphi));
}

static float _pl_tsfn(float phi, float sinphi, float e) {
	sinphi *= e;
	return (tan(.5f * (M_PI_2 - phi)) /
            pow((1.f - sinphi) / (1.f + sinphi), .5f * e)
            );
}

static void _pl_authset(float es, float *APA) {
    float t;
    APA[0] = es * P00;
    t = es * es; 
    APA[0] += t * P01;
    APA[1] = t * P10; 
    t *= es;
    APA[0] += t * P02;
    APA[1] += t * P11;
    APA[2] = t * P20;
}

static cl_int _pl_set_kernel_args(cl_kernel kernel, cl_mem xy_in, cl_mem xy_out, size_t count, 
                                  PLSpheroidInfo *info, int *offset_ptr) {
	cl_int error = 0;
	cl_uint vec_count = ck_padding(count, PL_FLOAT_VECTOR_SIZE) / PL_FLOAT_VECTOR_SIZE;
	int offset = *offset_ptr;
	error |= clSetKernelArg(kernel, offset++, sizeof(cl_mem), &xy_in);
	error |= clSetKernelArg(kernel, offset++, sizeof(cl_mem), &xy_out);
	error |= clSetKernelArg(kernel, offset++, sizeof(cl_uint), &vec_count);
	
	if (!_pl_spheroid_is_spherical(info->tag)) {
		error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &info->ecc);
		error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &info->ecc2);
		error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &info->one_ecc2);
	}
	
	*offset_ptr = offset;
	
	return error;
}

static cl_int _pl_enqueue_kernel_1d(cl_command_queue queue, cl_kernel kernel, size_t dim_count) {
    cl_int error = CL_SUCCESS;
	error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &dim_count, NULL, 0, NULL, NULL);
    return error;
}

cl_int pl_read_buffer(cl_command_queue queue, cl_mem xy_out_buf, float *xy_out, size_t out_count) {
    cl_int error;

    error = clEnqueueReadBuffer(queue, xy_out_buf, CL_TRUE, 0, out_count, xy_out, 0, NULL, NULL);
	if (error != CL_SUCCESS)
		return error;
	
	error = clFinish(queue);
	if (error != CL_SUCCESS)
		return error;
	
	return CL_SUCCESS;
}

cl_int pl_enqueue_kernel_albers_equal_area(cl_kernel kernel, PLContext *pl_ctx, cl_mem xy_in, cl_mem xy_out, size_t count,
									   PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0,
                                       float rlat1, float rlat2) {
	cl_int error = CL_SUCCESS;
	cl_int argc = 0;
	size_t vec_count = ck_padding(count, PL_FLOAT_VECTOR_SIZE) / PL_FLOAT_VECTOR_SIZE;
	PLSpheroidInfo info = _pl_get_spheroid_info(pl_ell);
	error = _pl_set_kernel_args(kernel, xy_in, xy_out, count, &info, &argc);
	
	float phi1 = rlat1 * DEG_TO_RAD;
	float phi2 = rlat2 * DEG_TO_RAD;
	float phi0 = lat0 * DEG_TO_RAD;
	float lam0 = lon0 * DEG_TO_RAD;
    float k0 = scale * info.major_axis;
    
	float c, dd, rho0, n, sinphi, cosphi;
	
	int secant; /* secant cone */
	
	n = sinphi = sin(phi1);
	cosphi = cos(phi1);
	secant = fabs(phi1 - phi2) >= EPS7;
	
	if (_pl_spheroid_is_spherical(pl_ell)) {
		if (secant) {
			n = .5 * (n + sin(phi2));
		}
		c = cosphi * cosphi + 2 * n * sinphi;
		dd = 1. / n;
		rho0 = dd * sqrt(c - 2 * n * sin(phi0));		
	} else {
		float ml1, m1;
		
		m1 = _pl_msfn(sinphi, cosphi, info.ecc2);
		ml1 = _pl_qsfn(sinphi, info.ecc, info.one_ecc2);
		if (secant) { 
			float ml2, m2;
			
			sinphi = sin(phi2);
			cosphi = cos(phi2);
			m2 = _pl_msfn(sinphi, cosphi, info.ecc2);
			ml2 = _pl_qsfn(sinphi, info.ecc, info.one_ecc2);
			n = (m1 * m1 - m2 * m2) / (ml2 - ml1);
		}
        
		c = m1 * m1 + n * ml1;
		dd = 1. / n;
		rho0 = dd * sqrt(c - n * _pl_qsfn(sin(phi0), info.ecc, info.one_ecc2));
	}
	
	if (!_pl_spheroid_is_spherical(pl_ell)) {
		error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &info.ec);
	}
    error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &k0);
    error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &x0);
    error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &y0);
	error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &lam0);
	error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &rho0);
	error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &c);
	error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &dd);
	error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &n);
    
	if (error != CL_SUCCESS)
		return error;
	
	return _pl_enqueue_kernel_1d(pl_ctx->queue, kernel, vec_count);
}

cl_int pl_enqueue_kernel_american_polyconic(cl_kernel kernel, PLContext *pl_ctx, cl_mem xy_in, cl_mem xy_out, size_t count,
									 PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0) {
	cl_int error;
	cl_int offset = 0;
	size_t vec_count = ck_padding(count, PL_FLOAT_VECTOR_SIZE) / PL_FLOAT_VECTOR_SIZE;
	PLSpheroidInfo info = _pl_get_spheroid_info(pl_ell);
	error = _pl_set_kernel_args(kernel, xy_in, xy_out, count, &info, &offset);
	
	float phi0 = lat0 * DEG_TO_RAD;
	float lambda0 = lon0 * DEG_TO_RAD;
	
	float ml0 = (float)_pl_mlfn(phi0, sinf(phi0), cosf(phi0), info.en);
	
	float k0 = scale * info.major_axis;
	
	error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &k0);
    error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &x0);
    error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &y0);
	error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &phi0);
	error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &ml0);
	error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &lambda0);
	error |= clSetKernelArg(kernel, offset++, sizeof(cl_float8), info.en);
	if (error != CL_SUCCESS) {
		return error;
	}
	
	return _pl_enqueue_kernel_1d(pl_ctx->queue, kernel, vec_count);
}

cl_int pl_enqueue_kernel_lambert_azimuthal_equal_area(cl_kernel kernel, PLContext *pl_ctx, 
                                                      cl_mem xy_in, cl_mem xy_out, size_t count,
                                                      PLSpheroid pl_ell, float scale,
                                                      float x0, float y0, float lon0, float lat0) {
    cl_int error;
    cl_int offset = 0;
    size_t vec_count = ck_padding(count, PL_FLOAT_VECTOR_SIZE) / PL_FLOAT_VECTOR_SIZE;
	PLSpheroidInfo info = _pl_get_spheroid_info(pl_ell);
	error = _pl_set_kernel_args(kernel, xy_in, xy_out, count, &info, &offset);
    
    float apa[4];
    
    _pl_authset(info.ecc2, apa);
	
	float phi0 = lat0 * DEG_TO_RAD;
	float lambda0 = lon0 * DEG_TO_RAD;
    float k0 = scale * info.major_axis;
    float qp = _pl_qsfn(1.f, info.ecc, info.one_ecc2);
    float rq = sqrtf(0.5f * qp);
    float sinPhi = sin(phi0);
    float sinB1 = _pl_qsfn(sinPhi, info.ecc, info.one_ecc2) / qp;
    float cosB1 = sqrtf(1.f - sinB1 * sinB1);
    float dd = cos(phi0) / (sqrtf(1.f - info.ecc2 * sinPhi * sinPhi) * rq * cosB1);
    float ymf = rq / dd;
    float xmf = rq * dd;

	error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &k0);
    error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &x0);
    error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &y0);
    error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &phi0);
    error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &lambda0);
	error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &qp);
	error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &sinB1);
    error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &cosB1);
    if (!_pl_spheroid_is_spherical(pl_ell)) {
        error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &rq);
        error |= clSetKernelArg(kernel, offset++, sizeof(cl_float4), apa);

        error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &dd);
        error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &xmf);
        error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &ymf);
    }
    if (error != CL_SUCCESS) {
		return error;
	}
    
    return _pl_enqueue_kernel_1d(pl_ctx->queue, kernel, vec_count);
}

cl_int pl_enqueue_kernel_lambert_conformal_conic(cl_kernel kernel, PLContext *pl_ctx, 
                                                 cl_mem xy_in, cl_mem xy_out, size_t count,
                                                 PLSpheroid pl_ell, float scale, 
                                                 float x0, float y0, 
                                                 float lon0, float lat0, 
                                                 float rlat1, float rlat2) {
    cl_int error;
    cl_int offset = 0;
    
    size_t vec_count = ck_padding(count, PL_FLOAT_VECTOR_SIZE) / PL_FLOAT_VECTOR_SIZE;
    PLSpheroidInfo info = _pl_get_spheroid_info(pl_ell);
    error = _pl_set_kernel_args(kernel, xy_in, xy_out, count, &info, &offset);
    
    float lambda0 = lon0 * DEG_TO_RAD;
    float phi0 = lat0 * DEG_TO_RAD;
    float phi1 = rlat1 * DEG_TO_RAD;
    float phi2 = rlat2 * DEG_TO_RAD;
    
    float k0, rho0, c, n, sinphi1, cosphi1, sinphi2;
    int secant = 0;
    
    k0 = scale * info.major_axis;
    
    sinphi1 = sin(phi1);
    cosphi1 = cos(phi1);
    if (fabs(phi1 - phi2) < 1.e-7) {
        n = sinphi1;
    } else {
        secant = 1;
    }
    
    if (_pl_spheroid_is_spherical(pl_ell)) {
        if (secant)
            n = log(cosphi1 / cos(phi2)) / log(tan(M_PI_4 + .5 * phi2) / tan(M_PI_4 + .5 * phi1));
        c = cosphi1 * pow(tan(M_PI_4 + .5 * phi1), n) / n;
        rho0 = c * pow(tan(M_PI_4 + .5 * phi0), -n);
    } else {
        float m1, ml1;
        
        m1 = _pl_msfn(sinphi1, cosphi1, info.ecc2);
        ml1 = _pl_tsfn(phi1, sinphi1, info.ecc);
        if (secant) {
            sinphi2 = sin(phi2);
            n = log(m1 / _pl_msfn(sinphi2, cos(phi2), info.ecc2));
            n /= log(ml1 / _pl_tsfn(phi2, sinphi2, info.ecc));
        }
        c = m1 * pow(ml1, -n) / n;
        rho0 = c * pow(_pl_tsfn(phi0, sin(phi0), info.ecc), n);
    }
    
    error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &k0);
    error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &x0);
    error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &y0);
    error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &lambda0);
    error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &rho0);
    error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &c);
    error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &n);
    
    if (error != CL_SUCCESS) {
		return error;
	}
    
    return _pl_enqueue_kernel_1d(pl_ctx->queue, kernel, vec_count);
}

cl_int pl_enqueue_kernel_mercator(cl_kernel kernel, PLContext *pl_ctx,
                              cl_mem xy_in, cl_mem xy_out, size_t count,
							  PLSpheroid pl_ell, float scale, float x0, float y0) {
	cl_int error;
	cl_int offset = 0;
	size_t vec_count = ck_padding(count, PL_FLOAT_VECTOR_SIZE) / PL_FLOAT_VECTOR_SIZE;
	
	PLSpheroidInfo info = _pl_get_spheroid_info(pl_ell);
	error = _pl_set_kernel_args(kernel, xy_in, xy_out, count, &info, &offset);
	
	float k0 = scale * info.major_axis;
	
	error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &k0);
    error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &x0);
    error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &y0);
	if (error != CL_SUCCESS) {
		return error;
	}
	return _pl_enqueue_kernel_1d(pl_ctx->queue, kernel, vec_count);
}

cl_int pl_enqueue_kernel_oblique_stereographic(cl_kernel kernel, PLContext *pl_ctx, cl_mem xy_in, cl_mem xy_out, size_t count,
        PLSpheroid pl_ell, float scale, float x0, float y0, float lon0, float lat0) {
	cl_int error = CL_SUCCESS;
	cl_int argc = 0;
	size_t vec_count = ck_padding(count, PL_FLOAT_VECTOR_SIZE) / PL_FLOAT_VECTOR_SIZE;
	PLSpheroidInfo info = _pl_get_spheroid_info(pl_ell);

	error = _pl_set_kernel_args(kernel, xy_in, xy_out, count, &info, &argc);

    float lambda0 = lon0 * DEG_TO_RAD;
    double phi0 = lat0 * DEG_TO_RAD;

    double sinPhi0, cosPhi0;
    sinPhi0 = sin(phi0);
    cosPhi0 = cos(phi0);

    float sinPhiC0, cosPhiC0;
    float scale_r2 = 2.0 * scale * info.major_axis * sqrt(info.one_ecc2) / (1.0 - info.ecc2 * sinPhi0 * sinPhi0);

	error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &scale_r2);
	error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &x0);
	error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &y0);

	if (!_pl_spheroid_is_spherical(pl_ell)) {
        double c0 = sqrt(1.0 + info.ecc2 * cosPhi0 * cosPhi0 * cosPhi0 * cosPhi0 / info.one_ecc2);
        double phiC0 = asin(sinPhi0 / c0);
        sinPhiC0 = sin(phiC0);
        cosPhiC0 = cos(phiC0);

        double k0 = tan(0.5 * phiC0 + M_PI_4) / (
                pow(tan(0.5 * phi0 + M_PI_4), c0) *
                pow((1.-info.ecc * sinPhi0)/(1.+info.ecc*sinPhi0), 0.5 * c0 * info.ecc) );

        float c0_f = c0;
        float k0_f = k0;
		error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &c0_f);
		error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &k0_f);
	} else {
        sinPhiC0 = sinPhi0;
        cosPhiC0 = cosPhi0;
    }

	error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &lambda0);
	error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &sinPhiC0);
	error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &cosPhiC0);
	if (error != CL_SUCCESS)
		return error;
	
	return _pl_enqueue_kernel_1d(pl_ctx->queue, kernel, vec_count);
}

cl_int pl_enqueue_kernel_robinson(cl_kernel kernel, PLContext *pl_ctx, cl_mem xy_in, cl_mem xy_out, size_t count,
        float scale, float x0, float y0) {
	cl_int error = CL_SUCCESS;
	cl_int argc = 0;
	
	PLSpheroidInfo info = _pl_get_spheroid_info(PL_SPHEROID_SPHERE);
    float k0 = scale * info.major_axis;

	error |= clSetKernelArg(kernel, argc++, sizeof(cl_mem), &xy_in);
	error |= clSetKernelArg(kernel, argc++, sizeof(cl_mem), &xy_out);
	error |= clSetKernelArg(kernel, argc++, sizeof(cl_uint), &count);

	error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &k0);
	error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &x0);
	error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &y0);
	if (error != CL_SUCCESS)
		return error;
	
	return _pl_enqueue_kernel_1d(pl_ctx->queue, kernel, count);
}

cl_int pl_enqueue_kernel_transverse_mercator(cl_kernel kernel, PLContext *pl_ctx, cl_mem xy_in, cl_mem xy_out, size_t count,
                                         PLSpheroid pl_ell, float scale, float x0, float y0,
                                         float lon0, float lat0) {
    cl_int error = CL_SUCCESS;
    cl_int argc = 0;
    size_t vec_count = ck_padding(count, PL_FLOAT_VECTOR_SIZE) / PL_FLOAT_VECTOR_SIZE;
    
    PLSpheroidInfo info = _pl_get_spheroid_info(pl_ell);
    error = _pl_set_kernel_args(kernel, xy_in, xy_out, count, &info, &argc);
    
    float k0 = scale * info.major_axis * info.kruger_a;
    float lambda0 = lon0 * DEG_TO_RAD;
    float phi0 = lat0 * DEG_TO_RAD;
    
    // float ml0 = _pl_mlfn(phi0, sin(phi0), cos(phi0), info.en);
    
    error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &k0);
    error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &x0);
    error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &y0);
    error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &phi0);
    error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &lambda0);
    // error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &ml0);
    error |= clSetKernelArg(kernel, argc++, sizeof(cl_float8), info.kruger_coef);
    if (error != CL_SUCCESS)
        return error;
    
    return _pl_enqueue_kernel_1d(pl_ctx->queue, kernel, vec_count);
}

cl_int pl_enqueue_kernel_winkel_tripel(cl_kernel kernel, PLContext *pl_ctx, cl_mem xy_in, cl_mem xy_out, size_t count,
                                       float scale, float x0, float y0, float lon0, float rlat1) {
    cl_int error = CL_SUCCESS;
    int argc = 0;
    
    size_t vec_count = ck_padding(count, PL_FLOAT_VECTOR_SIZE) / PL_FLOAT_VECTOR_SIZE;
    
    PLSpheroidInfo info = _pl_get_spheroid_info(PL_SPHEROID_SPHERE);
    
    float cosphi1 = cosf(rlat1 * DEG_TO_RAD);
    float lambda0 = lon0 * DEG_TO_RAD;
    float k0 = scale * info.major_axis;

    error |= clSetKernelArg(kernel, argc++, sizeof(cl_mem), &xy_in);
	error |= clSetKernelArg(kernel, argc++, sizeof(cl_mem), &xy_out);
	error |= clSetKernelArg(kernel, argc++, sizeof(cl_uint), &vec_count);
    error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &k0);
    error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &x0);
    error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &y0);
    error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &lambda0);
    error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &cosphi1);
    
    if (error != CL_SUCCESS)
        return error;
    
    return _pl_enqueue_kernel_1d(pl_ctx->queue, kernel, vec_count);
}

cl_int pl_run_kernel_inverse_geodesic(cl_kernel inv_kernel, PLContext *pl_ctx, PLInverseGeodesicBuffer *pl_buf,
									  float *dist_out, PLSpheroid pl_ell, float scale) {
	int argc = 0;
	cl_int error = CL_SUCCESS;
	PLSpheroidInfo info = _pl_get_spheroid_info(pl_ell);
	
	size_t xy2VecCount = ck_padding(pl_buf->xy2_count, PL_FLOAT_VECTOR_SIZE) / PL_FLOAT_VECTOR_SIZE;
	
	error |= clSetKernelArg(inv_kernel, argc++, sizeof(cl_mem), &pl_buf->xy1_in);
	error |= clSetKernelArg(inv_kernel, argc++, sizeof(cl_mem), &pl_buf->xy2_in);
	error |= clSetKernelArg(inv_kernel, argc++, sizeof(cl_mem), &pl_buf->dist_out);
	error |= clSetKernelArg(inv_kernel, argc++, sizeof(cl_float), &info.major_axis);
	
	if (error != CL_SUCCESS) {
		return error;
	}
	
	const size_t dim[2] = { pl_buf->xy1_count, xy2VecCount };
	
	error = clEnqueueNDRangeKernel(pl_ctx->queue, inv_kernel, 2, NULL, dim, NULL, 0, NULL, NULL);
	if (error != CL_SUCCESS) {
		return error;
	}
	
	float *dist_pad_out;
	size_t pad_count = 2 * pl_buf->xy1_count * ck_padding(pl_buf->xy2_count, PL_FLOAT_VECTOR_SIZE);
	if ((dist_pad_out = malloc(sizeof(float) * pad_count)) == NULL) {
		return CL_OUT_OF_HOST_MEMORY;
	}
	
	error = clEnqueueReadBuffer(pl_ctx->queue, pl_buf->dist_out, CL_TRUE, 0, pad_count * sizeof(cl_float),
								dist_pad_out, 0, NULL, NULL);
	if (error != CL_SUCCESS) {
		free(dist_pad_out);
		return error;
	}
	
	error = clFinish(pl_ctx->queue);
	if (error != CL_SUCCESS) {
		free(dist_pad_out);
		return error;
	}
	
	int j_size = ck_padding(pl_buf->xy2_count, PL_FLOAT_VECTOR_SIZE);
    int i, j;
	
	for (i=0; i<pl_buf->xy1_count; i++) {
		for (j=0; j<pl_buf->xy2_count; j++) {
			dist_out[i*pl_buf->xy2_count+j] = dist_pad_out[i*j_size+j] * scale;
		}
	}
	
	free(dist_pad_out);
	
	return CL_SUCCESS;
}

cl_int pl_run_kernel_forward_geodesic_fixed_distance(cl_kernel fwd_kernel, PLContext *pl_ctx,
    PLForwardGeodesicFixedDistanceBuffer *pl_buf, float *xy_out, PLSpheroid pl_ell, float distance) {
	PLSpheroidInfo info = _pl_get_spheroid_info(pl_ell);
	
	int argc = 0;
	
	cl_int error = CL_SUCCESS;
	
	float D, sinD, cosD;
	
	D = distance / info.major_axis;
	sinD = sinf(D);
	cosD = cosf(D);
	
	error |= clSetKernelArg(fwd_kernel, argc++, sizeof(cl_mem), &pl_buf->xy_in);
	error |= clSetKernelArg(fwd_kernel, argc++, sizeof(cl_mem), &pl_buf->phi_sincos);
	error |= clSetKernelArg(fwd_kernel, argc++, sizeof(cl_mem), &pl_buf->az_sincos);
	error |= clSetKernelArg(fwd_kernel, argc++, sizeof(cl_mem), &pl_buf->xy_out);
	error |= clSetKernelArg(fwd_kernel, argc++, sizeof(cl_float), &D);
	error |= clSetKernelArg(fwd_kernel, argc++, sizeof(cl_float), &sinD);
	error |= clSetKernelArg(fwd_kernel, argc++, sizeof(cl_float), &cosD);
	if (!_pl_spheroid_is_spherical(pl_ell)) {
		float flattening = 1.f/info.inverse_flattening;
		error |= clSetKernelArg(fwd_kernel, argc++, sizeof(cl_float), &flattening);
	}
	if (error != CL_SUCCESS) {
		return error;
	}
	
	const size_t dim[2] = { pl_buf->xy_count, 
		ck_padding(pl_buf->az_count, PL_FLOAT_VECTOR_SIZE)/PL_FLOAT_VECTOR_SIZE };
	
	error = clEnqueueNDRangeKernel(pl_ctx->queue, fwd_kernel, 2, NULL, 
								   dim, NULL, 0, NULL, NULL);
	if (error != CL_SUCCESS) {
		return error;
	}
	
	error = clEnqueueReadBuffer(pl_ctx->queue, pl_buf->xy_out, CL_TRUE, 0, 
								2 * pl_buf->xy_count * pl_buf->az_count * sizeof(cl_float),
								xy_out, 0, NULL, NULL);
	if (error != CL_SUCCESS) {
		return error;
	}
	
	error = clFinish(pl_ctx->queue);
	if (error != CL_SUCCESS) {
		return error;
	}
	
	return CL_SUCCESS;
}

cl_int pl_run_kernel_forward_geodesic_fixed_angle(cl_kernel fwd_kernel, PLContext *pl_ctx,
    PLForwardGeodesicFixedAngleBuffer *pl_buf, float *xy_in, float *xy_out, PLSpheroid pl_ell, float angle) {
    int argc = 0;
    
    size_t distVecCount = ck_padding(pl_buf->dist_count, PL_FLOAT_VECTOR_SIZE) / PL_FLOAT_VECTOR_SIZE;
    
    cl_int error = CL_SUCCESS;
    
    float sinAzimuth = sinf(angle);
    float cosAzimuth = cosf(angle);
    
    error |= clSetKernelArg(fwd_kernel, argc++, sizeof(cl_float2), xy_in);
    error |= clSetKernelArg(fwd_kernel, argc++, sizeof(cl_mem), &pl_buf->dist_in);
    error |= clSetKernelArg(fwd_kernel, argc++, sizeof(cl_mem), &pl_buf->xy_out);
    error |= clSetKernelArg(fwd_kernel, argc++, sizeof(cl_float), &angle);
    error |= clSetKernelArg(fwd_kernel, argc++, sizeof(cl_float), &sinAzimuth);
    error |= clSetKernelArg(fwd_kernel, argc++, sizeof(cl_float), &cosAzimuth);
    if (error != CL_SUCCESS) {
        return error;
    }
    
    error = clEnqueueNDRangeKernel(pl_ctx->queue, fwd_kernel, 1, NULL, &distVecCount,
                                   NULL, 0, NULL, NULL);
    if (error != CL_SUCCESS) {
        return error;
    }
    
    error = clEnqueueReadBuffer(pl_ctx->queue, pl_buf->xy_out, CL_TRUE, 0, 
                                2 * pl_buf->dist_count * sizeof(cl_float), 
                                xy_out, 0, NULL, NULL);
    if (error != CL_SUCCESS) {
        return error;
    }
    
    error = clFinish(pl_ctx->queue);
    if (error != CL_SUCCESS) {
        return error;
    }
    
    return CL_SUCCESS;
}

cl_int pl_run_kernel_geodesic_to_cartesian(cl_kernel g2c_kernel, PLContext *pl_ctx, PLDatumShiftBuffer *pl_buf,
                               PLSpheroid pl_ell) {
    PLSpheroidInfo info = _pl_get_spheroid_info(pl_ell);

    int argc = 0;
    cl_int error = CL_SUCCESS;
    size_t vec_count = ck_padding(pl_buf->count, PL_FLOAT_VECTOR_SIZE) / PL_FLOAT_VECTOR_SIZE;
        
    error |= clSetKernelArg(g2c_kernel, argc++, sizeof(cl_mem), &pl_buf->xy_in);
    error |= clSetKernelArg(g2c_kernel, argc++, sizeof(cl_mem), &pl_buf->x_rw);
    error |= clSetKernelArg(g2c_kernel, argc++, sizeof(cl_mem), &pl_buf->y_rw);
    error |= clSetKernelArg(g2c_kernel, argc++, sizeof(cl_mem), &pl_buf->z_rw);
    
    error |= clSetKernelArg(g2c_kernel, argc++, sizeof(cl_float), &info.ecc);
    error |= clSetKernelArg(g2c_kernel, argc++, sizeof(cl_float), &info.ecc2);
    error |= clSetKernelArg(g2c_kernel, argc++, sizeof(cl_float), &info.one_ecc2);
    
    error |= clSetKernelArg(g2c_kernel, argc++, sizeof(cl_float), &info.major_axis);
    error |= clSetKernelArg(g2c_kernel, argc++, sizeof(cl_float), &info.minor_axis);
    
    if (error != CL_SUCCESS)
        return error;
    
    error = clEnqueueNDRangeKernel(pl_ctx->queue, g2c_kernel, 1, NULL, &vec_count, NULL, 0, NULL, NULL);
    if (error != CL_SUCCESS)
        return error;
    
    error = clFinish(pl_ctx->queue);
    if (error != CL_SUCCESS)
        return error;
    
    return CL_SUCCESS;
}

/* We cut the computations in half by doing some matrix algebra beforehand.
 * Each datum shift is essentially an affine transformation in 3D cartesian
 * coordinates. So to get from on datum to another, instead of transforming
 * to/from WGS 84, we concatenate the transformation matrix of the source
 * datum to the inverse transformation matrix of the destination matrix, and
 * then just do a single matrix multiplication on each point instead of two
 * in order to transform it.
 */
cl_int pl_run_kernel_transform_cartesian(cl_kernel transform_kernel, PLContext *pl_ctx, PLDatumShiftBuffer *pl_buf,
                                         PLDatum src_datum, PLDatum dst_datum) {
    float Rx1, Ry1, Rz1, M1, Dx1, Dy1, Dz1;
    float Rx2, Ry2, Rz2, M2, Dx2, Dy2, Dz2;
    
    cl_int error = CL_SUCCESS;
    int argc = 0;
    size_t vec_count = ck_padding(pl_buf->count, PL_FLOAT_VECTOR_SIZE) / PL_FLOAT_VECTOR_SIZE;
    
    Dx1 = pl_datum_params[src_datum].dx;
    Dy1 = pl_datum_params[src_datum].dy;
    Dz1 = pl_datum_params[src_datum].dz;
    M1  = pl_datum_params[src_datum].ppm*1.e-6 + 1;
    Rx1 = pl_datum_params[src_datum].ex * SEC_TO_RAD;
    Ry1 = pl_datum_params[src_datum].ey * SEC_TO_RAD;
    Rz1 = pl_datum_params[src_datum].ez * SEC_TO_RAD;
    
    Dx2 = pl_datum_params[dst_datum].dx;
    Dy2 = pl_datum_params[dst_datum].dy;
    Dz2 = pl_datum_params[dst_datum].dz;
    M2  = pl_datum_params[dst_datum].ppm*1.e-6 + 1;
    Rx2 = pl_datum_params[dst_datum].ex * SEC_TO_RAD;
    Ry2 = pl_datum_params[dst_datum].ey * SEC_TO_RAD;
    Rz2 = pl_datum_params[dst_datum].ez * SEC_TO_RAD;
    
    struct pl_matrix matrix1 = pl_affine_transform_make(Rx1, Ry1, Rz1, M1, Dx1, Dy1, Dz1);
    struct pl_matrix matrix2 = pl_affine_transform_make(Rx2, Ry2, Rz2, M2, Dx2, Dy2, Dz2);
    
    /* Invert the destination matrix */
    __CLPK_integer n = 4;
    __CLPK_integer info;
    
    __CLPK_integer ipiv[4];
    __CLPK_real work[4];
    __CLPK_integer work_n = 4;
    
    sgetrf_(&n, &n, &matrix2.elements[0][0], &n, ipiv, &info);
    if (info != 0) {
        return info;
    }
    
    sgetri_(&n, &matrix2.elements[0][0], &n, ipiv, work, &work_n, &info);
    if (info != 0) {
        return info;
    }
    
    __CLPK_real alpha = 1.f;
    __CLPK_real beta = 0.f;
    __CLPK_real result_matrix[4][4];
    
    /* Multiply the source matrix with the inverse destination matrix */
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, alpha,
          &matrix2.elements[0][0], n,
          &matrix1.elements[0][0], n,
          beta, &result_matrix[0][0], n);
    
    /* transpose the result */
    float tmatrix[4][4];
    int i, j;
    for (i=0; i<4; i++) {
        for (j=0; j<4; j++) {
            tmatrix[i][j] = result_matrix[j][i];
        }
    }
    
    error |= clSetKernelArg(transform_kernel, argc++, sizeof(cl_mem), &pl_buf->x_rw);
    error |= clSetKernelArg(transform_kernel, argc++, sizeof(cl_mem), &pl_buf->y_rw);
    error |= clSetKernelArg(transform_kernel, argc++, sizeof(cl_mem), &pl_buf->z_rw);
    error |= clSetKernelArg(transform_kernel, argc++, sizeof(cl_float16), &tmatrix[0][0]);
    if (error != CL_SUCCESS)
        return error;
    
    error = clEnqueueNDRangeKernel(pl_ctx->queue, transform_kernel, 1, NULL, 
                                   &vec_count, NULL, 0, NULL, NULL);
    if (error != CL_SUCCESS)
        return error;
    
    return error;
}

cl_int pl_run_kernel_cartesian_to_geodesic(cl_kernel c2g_kernel, PLContext *pl_ctx, PLDatumShiftBuffer *pl_buf,
                                           float *xy_out, PLSpheroid pl_ell) {
    PLSpheroidInfo info = _pl_get_spheroid_info(pl_ell);
    
    int argc = 0;
    cl_int error = CL_SUCCESS;
    size_t vec_count = ck_padding(pl_buf->count, PL_FLOAT_VECTOR_SIZE) / PL_FLOAT_VECTOR_SIZE;
        
    error |= clSetKernelArg(c2g_kernel, argc++, sizeof(cl_mem), &pl_buf->x_rw);
    error |= clSetKernelArg(c2g_kernel, argc++, sizeof(cl_mem), &pl_buf->y_rw);
    error |= clSetKernelArg(c2g_kernel, argc++, sizeof(cl_mem), &pl_buf->z_rw);
    error |= clSetKernelArg(c2g_kernel, argc++, sizeof(cl_mem), &pl_buf->xy_out);
    
    error |= clSetKernelArg(c2g_kernel, argc++, sizeof(cl_float), &info.ecc);
    error |= clSetKernelArg(c2g_kernel, argc++, sizeof(cl_float), &info.ecc2);
    error |= clSetKernelArg(c2g_kernel, argc++, sizeof(cl_float), &info.one_ecc2);
    
    error |= clSetKernelArg(c2g_kernel, argc++, sizeof(cl_float), &info.major_axis);
    error |= clSetKernelArg(c2g_kernel, argc++, sizeof(cl_float), &info.minor_axis);
    if (error != CL_SUCCESS)
        return error;
    
    error = clEnqueueNDRangeKernel(pl_ctx->queue, c2g_kernel, 1, NULL,
                                   &vec_count, NULL, 0, NULL, NULL);
    if (error != CL_SUCCESS)
        return error;
    
    if (xy_out != NULL) {
        error = clEnqueueReadBuffer(pl_ctx->queue, pl_buf->xy_out, CL_TRUE, 0, 
                                    2 * pl_buf->count * sizeof(cl_float), xy_out, 0, NULL, NULL);
        if (error != CL_SUCCESS)
            return error;
    }
    
    error = clFinish(pl_ctx->queue);
    if (error != CL_SUCCESS)
        return error;
    
    return CL_SUCCESS;
}
