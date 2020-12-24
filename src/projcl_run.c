/*
 *  projcl_run.c
 *  Magic Maps
 *
 *  Created by Evan Miller on 2/12/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include <projcl/projcl.h>
#include <projcl/projcl_warp.h>
#include "projcl_run.h"
#include "projcl_kernel.h"
#include "projcl_util.h"
#include "projcl_spheroid.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif
#ifdef __linux__
#include <cblas.h>
typedef int __CLPK_integer;
typedef double __CLPK_doublereal;
extern int dgetrf_(__CLPK_integer *, __CLPK_integer *, __CLPK_doublereal *, __CLPK_integer *, __CLPK_integer *, __CLPK_integer *);
extern int dgetri_(__CLPK_integer *, __CLPK_doublereal *, __CLPK_integer *, __CLPK_integer *, __CLPK_doublereal *, __CLPK_integer *, __CLPK_integer *);
#endif

#define EPS7 1.e-7

#define SEC_TO_RAD 4.84813681109535993589914102357e-6

static cl_int pl_enqueue_kernel_albers_equal_area(PLContext *pl_ctx, cl_kernel kernel,
        PLProjectionParams *params, cl_mem xy_in, cl_mem xy_out, size_t count);
static cl_int pl_enqueue_kernel_american_polyconic(PLContext *pl_ctx, cl_kernel kernel,
        PLProjectionParams *params, cl_mem xy_in, cl_mem xy_out, size_t count);
static cl_int pl_enqueue_kernel_lambert_azimuthal_equal_area(PLContext *pl_ctx, cl_kernel kernel,
        PLProjectionParams *params, cl_mem xy_in, cl_mem xy_out, size_t count);
static cl_int pl_enqueue_kernel_lambert_conformal_conic(PLContext *pl_ctx, cl_kernel kernel,
        PLProjectionParams *params, cl_mem xy_in, cl_mem xy_out, size_t count);
static cl_int pl_enqueue_kernel_mercator(PLContext *pl_ctx, cl_kernel kernel,
        PLProjectionParams *params, cl_mem xy_in, cl_mem xy_out, size_t count);
static cl_int pl_enqueue_kernel_oblique_stereographic(PLContext *pl_ctx, cl_kernel kernel,
        PLProjectionParams *params, cl_mem xy_in, cl_mem xy_out, size_t count);
static cl_int pl_enqueue_kernel_robinson(PLContext *pl_ctx, cl_kernel kernel,
        PLProjectionParams *params, cl_mem xy_in, cl_mem xy_out, size_t count);
static cl_int pl_enqueue_kernel_transverse_mercator(PLContext *pl_ctx, cl_kernel kernel,
        PLProjectionParams *params, cl_mem xy_in, cl_mem xy_out, size_t count);
static cl_int pl_enqueue_kernel_winkel_tripel(PLContext *pl_ctx, cl_kernel kernel,
        PLProjectionParams *params, cl_mem xy_in, cl_mem xy_out, size_t count);

struct pl_projection_info {
    PLProjection proj;
    char         name[80];
    cl_int (*func)(PLContext *pl_ctx,
                          cl_kernel kernel,
                          PLProjectionParams *params,
                          cl_mem xy_in,
                          cl_mem xy_out,
                          size_t count);
};

static struct pl_projection_info _pl_projection_info[] = {
    {
        .proj = PL_PROJECT_ALBERS_EQUAL_AREA,
        .name = "albers_equal_area",
        .func = &pl_enqueue_kernel_albers_equal_area,
    },
    {
        .proj = PL_PROJECT_AMERICAN_POLYCONIC,
        .name = "american_polyconic",
        .func = &pl_enqueue_kernel_american_polyconic,
    },
    {
        .proj = PL_PROJECT_LAMBERT_AZIMUTHAL_EQUAL_AREA,
        .name = "lambert_azimuthal_equal_area",
        .func = &pl_enqueue_kernel_lambert_azimuthal_equal_area,
    },
    {
        .proj = PL_PROJECT_LAMBERT_CONFORMAL_CONIC,
        .name = "lambert_conformal_conic",
        .func = &pl_enqueue_kernel_lambert_conformal_conic,
    },
    {
        .proj = PL_PROJECT_MERCATOR,
        .name = "mercator",
        .func = &pl_enqueue_kernel_mercator,
    },
    {
        .proj = PL_PROJECT_OBLIQUE_STEREOGRAPHIC,
        .name = "oblique_stereographic",
        .func = &pl_enqueue_kernel_oblique_stereographic,
    },
    {
        .proj = PL_PROJECT_ROBINSON,
        .name = "robinson",
        .func = &pl_enqueue_kernel_robinson,
    },
    {
        .proj = PL_PROJECT_TRANSVERSE_MERCATOR,
        .name = "transverse_mercator",
        .func = &pl_enqueue_kernel_transverse_mercator,
    },
    {
        .proj = PL_PROJECT_WINKEL_TRIPEL,
        .name = "winkel_tripel",
        .func = &pl_enqueue_kernel_winkel_tripel,
    }
};

struct pl_datum_info {
    double dx;
    double dy;
    double dz;
    double ex;
    double ey;
    double ez;
    double ppm;
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
    double elements[4][4];
};

static struct pl_matrix pl_affine_transform_make(
        double Rx, double Ry, double Rz, double M,
        double Dx, double Dy, double Dz) {
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

static const char *_pl_proj_name(PLProjection proj) {
    int i=0;
    const char *name = NULL;
    for (i=0; i<sizeof(_pl_projection_info)/sizeof(_pl_projection_info[0]); i++) {
        if (_pl_projection_info[i].proj == proj) {
            name = _pl_projection_info[i].name;
            break;
        }
    }
    return name;
}

static double _pl_mlfn(double phi, double sphi, double cphi, double *en) {
	cphi *= sphi;
	sphi *= sphi;
	return(en[0] * phi - cphi * (en[1] + sphi*(en[2] + sphi*(en[3] + sphi*en[4]))));
}

static double _pl_qsfn(double sinphi, double e, double one_es) {
	double con = e * sinphi;
	return (one_es * (sinphi / (1.0 - con * con) - (.5 / e) * (log1p(-con) - log1p(con))));
}

static double _pl_msfn(double sinphi, double cosphi, double es) {
	return (cosphi / sqrt(1.0 - es * sinphi * sinphi));
}

static double _pl_tsfn(double phi, double sinphi, double e) {
	double con = e * sinphi;
	return (tan(.5 * (M_PI_2 - phi)) /
            pow((1.0 - con) / (1.0 + con), .5 * e));
}

static cl_int _pl_set_projection_kernel_args(PLContext *ctx, cl_kernel kernel,
        cl_mem xy_in, cl_mem xy_out, size_t count, PLSpheroidInfo *info, int *offset_ptr) {
	cl_int error = 0;
	cl_uint vec_count = ck_padding(count, PL_FLOAT_VECTOR_SIZE) / PL_FLOAT_VECTOR_SIZE;
	int offset = *offset_ptr;

	error |= pl_set_kernel_arg_mem(ctx, kernel, offset++, xy_in);
	error |= pl_set_kernel_arg_mem(ctx, kernel, offset++, xy_out);
	error |= pl_set_kernel_arg_uint(ctx, kernel, offset++, vec_count);
	
	if (!_pl_spheroid_is_spherical(info->tag)) {
        error |= pl_set_kernel_arg_float(ctx, kernel, offset++, info->ecc);
        error |= pl_set_kernel_arg_float(ctx, kernel, offset++, info->ecc2);
        error |= pl_set_kernel_arg_float(ctx, kernel, offset++, info->one_ecc2);
	}
	
	*offset_ptr = offset;
	
	return error;
}

cl_kernel pl_find_projection_kernel(PLContext *pl_ctx, PLProjection proj, int fwd, PLSpheroid ell) {
	char requested_name[128];
	if (fwd) {
		sprintf(requested_name, "pl_project_%s_%c", _pl_proj_name(proj), _pl_spheroid_is_spherical(ell) ? 's' : 'e');
	} else {
		sprintf(requested_name, "pl_unproject_%s_%c", _pl_proj_name(proj), _pl_spheroid_is_spherical(ell) ? 's' : 'e');
	}
	return pl_find_kernel(pl_ctx, requested_name);
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

static cl_int _pl_enqueue_projection_kernel(PLContext *pl_ctx, cl_kernel kernel, PLProjection proj, PLProjectionParams *params,
        cl_mem xy_in, cl_mem xy_out, size_t count) {
    cl_int error = CL_SUCCESS;
    int i;
    for (i=0; i<sizeof(_pl_projection_info)/sizeof(_pl_projection_info[0]); i++) {
        if (_pl_projection_info[i].proj == proj) {
            error = _pl_projection_info[i].func(pl_ctx, kernel, params, xy_in, xy_out, count);
            break;
        }
    }
    return error;
}

cl_int pl_enqueue_projection_kernel_points(PLContext *pl_ctx, cl_kernel kernel, PLProjection proj, PLProjectionParams *params,
        PLProjectionBuffer *pl_buf) {
    return _pl_enqueue_projection_kernel(pl_ctx, kernel, proj, params, pl_buf->xy_in, pl_buf->xy_out, pl_buf->count);
}

cl_int pl_enqueue_projection_kernel_grid(PLContext *pl_ctx, cl_kernel kernel, PLProjection proj, PLProjectionParams *params,
        PLPointGridBuffer *src, PLPointGridBuffer *dst) {
    return _pl_enqueue_projection_kernel(pl_ctx, kernel, proj, params, src->grid, dst->grid, src->width * src->height);
}

cl_int pl_enqueue_kernel_albers_equal_area(PLContext *pl_ctx, cl_kernel kernel,
        PLProjectionParams *params, cl_mem xy_in, cl_mem xy_out, size_t count) {
	cl_int error = CL_SUCCESS;
	cl_int argc = 0;
	size_t vec_count = ck_padding(count, PL_FLOAT_VECTOR_SIZE) / PL_FLOAT_VECTOR_SIZE;
	PLSpheroidInfo info = _pl_get_spheroid_info(params->spheroid);
	error = _pl_set_projection_kernel_args(pl_ctx, kernel, xy_in, xy_out, count, &info, &argc);
	
	double phi1 = params->rlat1 * DEG_TO_RAD;
	double phi2 = params->rlat2 * DEG_TO_RAD;
	double phi0 = params->lat0 * DEG_TO_RAD;
    
	double rho0;
	double c, n;
    double sinphi, cosphi;
	
	n = sinphi = sin(phi1);
	cosphi = cos(phi1);
	
	if (_pl_spheroid_is_spherical(params->spheroid)) {
        n = .5 * (sinphi + sin(phi2));
        c = 1.0 + sin(phi2) * sinphi;
        rho0 = sqrt(c - 2.f * n * sin(phi0));
	} else {
		double ml1, m1;
		
		m1 = _pl_msfn(sinphi, cosphi, info.ecc2);
		ml1 = _pl_qsfn(sinphi, info.ecc, info.one_ecc2);
		if (fabs(phi1 - phi2) >= EPS7) {
			double ml2, m2;
			
			sinphi = sin(phi2);
			cosphi = cos(phi2);
			m2 = _pl_msfn(sinphi, cosphi, info.ecc2);
			ml2 = _pl_qsfn(sinphi, info.ecc, info.one_ecc2);
			n = (m1 * m1 - m2 * m2) / (ml2 - ml1);
		}
        
		c = m1 * m1 + ml1 * n;
		rho0 = sqrt(c - n * _pl_qsfn(sin(phi0), info.ecc, info.one_ecc2));
	}
	
	if (!_pl_spheroid_is_spherical(params->spheroid)) {
		error |= pl_set_kernel_arg_float(pl_ctx, kernel, argc++, info.ec);
	}
    error |= pl_set_kernel_arg_float(pl_ctx, kernel, argc++, params->scale * info.major_axis / n);
    error |= pl_set_kernel_arg_float(pl_ctx, kernel, argc++, params->x0);
    error |= pl_set_kernel_arg_float(pl_ctx, kernel, argc++, params->y0);
	error |= pl_set_kernel_arg_float(pl_ctx, kernel, argc++, params->lon0 * DEG_TO_RAD);
	error |= pl_set_kernel_arg_float(pl_ctx, kernel, argc++, rho0);
	error |= pl_set_kernel_arg_float(pl_ctx, kernel, argc++, c);
	error |= pl_set_kernel_arg_float(pl_ctx, kernel, argc++, n);
    
	if (error != CL_SUCCESS)
		return error;
	
	return _pl_enqueue_kernel_1d(pl_ctx->queue, kernel, vec_count);
}

cl_int pl_enqueue_kernel_american_polyconic(PLContext *pl_ctx,cl_kernel kernel,
        PLProjectionParams *params, cl_mem xy_in, cl_mem xy_out, size_t count) {
	cl_int error;
	cl_int offset = 0;
	size_t vec_count = ck_padding(count, PL_FLOAT_VECTOR_SIZE) / PL_FLOAT_VECTOR_SIZE;
	PLSpheroidInfo info = _pl_get_spheroid_info(params->spheroid);
	error = _pl_set_projection_kernel_args(pl_ctx, kernel, xy_in, xy_out, count, &info, &offset);
	
	double phi0 = params->lat0 * DEG_TO_RAD;
	
	double ml0 = _pl_mlfn(phi0, sin(phi0), cos(phi0), info.en);
	
	error |= pl_set_kernel_arg_float(pl_ctx, kernel, offset++, params->scale * info.major_axis);
    error |= pl_set_kernel_arg_float(pl_ctx, kernel, offset++, params->x0);
    error |= pl_set_kernel_arg_float(pl_ctx, kernel, offset++, params->y0);
	error |= pl_set_kernel_arg_float(pl_ctx, kernel, offset++, phi0);
	error |= pl_set_kernel_arg_float(pl_ctx, kernel, offset++, params->lon0 * DEG_TO_RAD);
    if (!_pl_spheroid_is_spherical(params->spheroid)) {
        error |= pl_set_kernel_arg_float(pl_ctx, kernel, offset++, ml0);
        error |= pl_set_kernel_arg_float8(pl_ctx, kernel, offset++, info.en);
    }
	if (error != CL_SUCCESS) {
		return error;
	}
	
	return _pl_enqueue_kernel_1d(pl_ctx->queue, kernel, vec_count);
}

cl_int pl_enqueue_kernel_lambert_azimuthal_equal_area(PLContext *pl_ctx, cl_kernel kernel,
        PLProjectionParams *params, cl_mem xy_in, cl_mem xy_out, size_t count) {
    cl_int error;
    cl_int offset = 0;
    size_t vec_count = ck_padding(count, PL_FLOAT_VECTOR_SIZE) / PL_FLOAT_VECTOR_SIZE;
	PLSpheroidInfo info = _pl_get_spheroid_info(params->spheroid);
	error = _pl_set_projection_kernel_args(pl_ctx, kernel, xy_in, xy_out, count, &info, &offset);
    
	double phi0 = params->lat0 * DEG_TO_RAD;

	error |= pl_set_kernel_arg_float(pl_ctx, kernel, offset++, params->scale * info.major_axis);
    error |= pl_set_kernel_arg_float(pl_ctx, kernel, offset++, params->x0);
    error |= pl_set_kernel_arg_float(pl_ctx, kernel, offset++, params->y0);
    error |= pl_set_kernel_arg_float(pl_ctx, kernel, offset++, phi0);
    error |= pl_set_kernel_arg_float(pl_ctx, kernel, offset++, params->lon0 * DEG_TO_RAD);
    if (_pl_spheroid_is_spherical(params->spheroid)) {
        error |= pl_set_kernel_arg_float(pl_ctx, kernel, offset++, sin(phi0));
        error |= pl_set_kernel_arg_float(pl_ctx, kernel, offset++, cos(phi0));
    } else {
        double qp = _pl_qsfn(1.0, info.ecc, info.one_ecc2);
        double sinPhi = sin(phi0);
        double sinB1 = _pl_qsfn(sinPhi, info.ecc, info.one_ecc2) / qp;
        double cosB1 = sqrt(1.0 - sinB1 * sinB1);

        double rq = sqrt(0.5 * qp);
        double dd = cos(phi0) / (sqrt(1.0 - info.ecc2 * sinPhi * sinPhi) * rq * cosB1);
        double ymf = rq / dd;
        double xmf = rq * dd;

        error |= pl_set_kernel_arg_float(pl_ctx, kernel, offset++, qp);
        error |= pl_set_kernel_arg_float(pl_ctx, kernel, offset++, sinB1);
        error |= pl_set_kernel_arg_float(pl_ctx, kernel, offset++, cosB1);

        error |= pl_set_kernel_arg_float(pl_ctx, kernel, offset++, rq);
        error |= pl_set_kernel_arg_float4(pl_ctx, kernel, offset++, info.apa);

        error |= pl_set_kernel_arg_float(pl_ctx, kernel, offset++, dd);
        error |= pl_set_kernel_arg_float(pl_ctx, kernel, offset++, xmf);
        error |= pl_set_kernel_arg_float(pl_ctx, kernel, offset++, ymf);
    }
    if (error != CL_SUCCESS) {
		return error;
	}
    
    return _pl_enqueue_kernel_1d(pl_ctx->queue, kernel, vec_count);
}

cl_int pl_enqueue_kernel_lambert_conformal_conic(PLContext *pl_ctx, cl_kernel kernel,
        PLProjectionParams *params, cl_mem xy_in, cl_mem xy_out, size_t count) {
    cl_int error;
    cl_int offset = 0;
    
    size_t vec_count = ck_padding(count, PL_FLOAT_VECTOR_SIZE) / PL_FLOAT_VECTOR_SIZE;
    PLSpheroidInfo info = _pl_get_spheroid_info(params->spheroid);
    error = _pl_set_projection_kernel_args(pl_ctx, kernel, xy_in, xy_out, count, &info, &offset);
    
    double phi0 = params->lat0 * DEG_TO_RAD;
    double phi1 = params->rlat1 * DEG_TO_RAD;
    double phi2 = params->rlat2 * DEG_TO_RAD;
    
    double rho0, c, n, sinphi1, cosphi1, sinphi2;
    int secant = 0;
    
    sinphi1 = sin(phi1);
    cosphi1 = cos(phi1);
    if (fabs(phi1 - phi2) < 1.e-7) {
        n = sinphi1;
    } else {
        secant = 1;
    }
    
    if (_pl_spheroid_is_spherical(params->spheroid)) {
        if (secant)
            n = log(cosphi1 / cos(phi2)) / (asinh(tan(phi2)) - asinh(tan(phi1)));
        c = cosphi1 * pow(tan(M_PI_4 + .5 * phi1), n) / n;
        rho0 = c * pow(tan(M_PI_4 + .5 * phi0), -n);
    } else {
        double m1, ml1;
        
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
    
    error |= pl_set_kernel_arg_float(pl_ctx, kernel, offset++, params->scale * info.major_axis);
    error |= pl_set_kernel_arg_float(pl_ctx, kernel, offset++, params->x0);
    error |= pl_set_kernel_arg_float(pl_ctx, kernel, offset++, params->y0);
    error |= pl_set_kernel_arg_float(pl_ctx, kernel, offset++, params->lon0 * DEG_TO_RAD);
    error |= pl_set_kernel_arg_float(pl_ctx, kernel, offset++, rho0);
    error |= pl_set_kernel_arg_float(pl_ctx, kernel, offset++, c);
    error |= pl_set_kernel_arg_float(pl_ctx, kernel, offset++, n);
    
    if (error != CL_SUCCESS) {
		return error;
	}
    
    return _pl_enqueue_kernel_1d(pl_ctx->queue, kernel, vec_count);
}

cl_int pl_enqueue_kernel_mercator(PLContext *pl_ctx, cl_kernel kernel,
        PLProjectionParams *params, cl_mem xy_in, cl_mem xy_out, size_t count) {
	cl_int error;
	cl_int offset = 0;
	size_t vec_count = ck_padding(count, PL_FLOAT_VECTOR_SIZE) / PL_FLOAT_VECTOR_SIZE;
	
	PLSpheroidInfo info = _pl_get_spheroid_info(params->spheroid);
	error = _pl_set_projection_kernel_args(pl_ctx, kernel, xy_in, xy_out, count, &info, &offset);
	
	error |= pl_set_kernel_arg_float(pl_ctx, kernel, offset++, params->scale * info.major_axis);
    error |= pl_set_kernel_arg_float(pl_ctx, kernel, offset++, params->x0);
    error |= pl_set_kernel_arg_float(pl_ctx, kernel, offset++, params->y0);
	if (error != CL_SUCCESS) {
		return error;
	}
	return _pl_enqueue_kernel_1d(pl_ctx->queue, kernel, vec_count);
}

cl_int pl_enqueue_kernel_oblique_stereographic(PLContext *pl_ctx, cl_kernel kernel,
        PLProjectionParams *params, cl_mem xy_in, cl_mem xy_out, size_t count) {
	cl_int error = CL_SUCCESS;
	cl_int argc = 0;
	size_t vec_count = ck_padding(count, PL_FLOAT_VECTOR_SIZE) / PL_FLOAT_VECTOR_SIZE;
	PLSpheroidInfo info = _pl_get_spheroid_info(params->spheroid);

	error = _pl_set_projection_kernel_args(pl_ctx, kernel, xy_in, xy_out, count, &info, &argc);

    double phi0 = params->lat0 * DEG_TO_RAD;

    double sinPhi0, cosPhi0;
    sinPhi0 = sin(phi0);
    cosPhi0 = cos(phi0);

    double sinPhiC0, cosPhiC0;
    double scale_r2 = 2.0 * params->scale * info.major_axis * sqrt(info.one_ecc2) / (1.0 - info.ecc2 * sinPhi0 * sinPhi0);

	error |= pl_set_kernel_arg_float(pl_ctx, kernel, argc++, scale_r2);
	error |= pl_set_kernel_arg_float(pl_ctx, kernel, argc++, params->x0);
	error |= pl_set_kernel_arg_float(pl_ctx, kernel, argc++, params->y0);

	if (!_pl_spheroid_is_spherical(params->spheroid)) {
        double c0 = sqrt(1.0 + info.ecc2 * cosPhi0 * cosPhi0 * cosPhi0 * cosPhi0 / info.one_ecc2);
        double phiC0 = asin(sinPhi0 / c0);
        sinPhiC0 = sin(phiC0);
        cosPhiC0 = cos(phiC0);

        double k0 = tan(0.5 * phiC0 + M_PI_4) / (
                pow(tan(0.5 * phi0 + M_PI_4), c0) *
                pow((1.-info.ecc * sinPhi0)/(1.+info.ecc*sinPhi0), 0.5 * c0 * info.ecc) );

		error |= pl_set_kernel_arg_float(pl_ctx, kernel, argc++, c0);
		error |= pl_set_kernel_arg_float(pl_ctx, kernel, argc++, log(k0));
	} else {
        sinPhiC0 = sinPhi0;
        cosPhiC0 = cosPhi0;
    }

	error |= pl_set_kernel_arg_float(pl_ctx, kernel, argc++, params->lon0 * DEG_TO_RAD);
	error |= pl_set_kernel_arg_float(pl_ctx, kernel, argc++, sinPhiC0);
	error |= pl_set_kernel_arg_float(pl_ctx, kernel, argc++, cosPhiC0);
	if (error != CL_SUCCESS)
		return error;
	
	return _pl_enqueue_kernel_1d(pl_ctx->queue, kernel, vec_count);
}

cl_int pl_enqueue_kernel_robinson(PLContext *pl_ctx, cl_kernel kernel,
        PLProjectionParams *params, cl_mem xy_in, cl_mem xy_out, size_t count) {
	cl_int error = CL_SUCCESS;
	cl_int argc = 0;
	
	PLSpheroidInfo info = _pl_get_spheroid_info(PL_SPHEROID_SPHERE);

	error |= pl_set_kernel_arg_mem(pl_ctx, kernel, argc++, xy_in);
	error |= pl_set_kernel_arg_mem(pl_ctx, kernel, argc++, xy_out);
	error |= pl_set_kernel_arg_uint(pl_ctx, kernel, argc++, count);

	error |= pl_set_kernel_arg_float(pl_ctx, kernel, argc++, params->scale * info.major_axis);
	error |= pl_set_kernel_arg_float(pl_ctx, kernel, argc++, params->x0);
	error |= pl_set_kernel_arg_float(pl_ctx, kernel, argc++, params->y0);
	if (error != CL_SUCCESS)
		return error;
	
	return _pl_enqueue_kernel_1d(pl_ctx->queue, kernel, count);
}

cl_int pl_enqueue_kernel_transverse_mercator(PLContext *pl_ctx, cl_kernel kernel,
        PLProjectionParams *params, cl_mem xy_in, cl_mem xy_out, size_t count) {
    cl_int error = CL_SUCCESS;
    cl_int argc = 0;
    size_t vec_count = ck_padding(count, PL_FLOAT_VECTOR_SIZE) / PL_FLOAT_VECTOR_SIZE;
    
    PLSpheroidInfo info = _pl_get_spheroid_info(params->spheroid);
    error = _pl_set_projection_kernel_args(pl_ctx, kernel, xy_in, xy_out, count, &info, &argc);
    
    error |= pl_set_kernel_arg_float(pl_ctx, kernel, argc++, params->scale * info.major_axis * info.krueger_A);
    error |= pl_set_kernel_arg_float(pl_ctx, kernel, argc++, params->x0);
    error |= pl_set_kernel_arg_float(pl_ctx, kernel, argc++, params->y0);
    error |= pl_set_kernel_arg_float(pl_ctx, kernel, argc++, params->lon0 * DEG_TO_RAD);
    if (!_pl_spheroid_is_spherical(params->spheroid)) {
        error |= pl_set_kernel_arg_float8(pl_ctx, kernel, argc++, info.krueger_alpha);
        error |= pl_set_kernel_arg_float8(pl_ctx, kernel, argc++, info.krueger_beta);
    }
    if (error != CL_SUCCESS)
        return error;
    
    return _pl_enqueue_kernel_1d(pl_ctx->queue, kernel, vec_count);
}

cl_int pl_enqueue_kernel_winkel_tripel(PLContext *pl_ctx, cl_kernel kernel,
        PLProjectionParams *params, cl_mem xy_in, cl_mem xy_out, size_t count) {
    cl_int error = CL_SUCCESS;
    int argc = 0;
    
    size_t vec_count = ck_padding(count, PL_FLOAT_VECTOR_SIZE) / PL_FLOAT_VECTOR_SIZE;
    
    PLSpheroidInfo info = _pl_get_spheroid_info(PL_SPHEROID_SPHERE);
    
    double cosphi1 = isnan(params->rlat1) ? M_2_PI : cos(params->rlat1 * DEG_TO_RAD);

    error |= pl_set_kernel_arg_mem(pl_ctx, kernel, argc++, xy_in);
	error |= pl_set_kernel_arg_mem(pl_ctx, kernel, argc++, xy_out);
	error |= pl_set_kernel_arg_uint(pl_ctx, kernel, argc++, vec_count);
    error |= pl_set_kernel_arg_float(pl_ctx, kernel, argc++, params->scale * info.major_axis);
    error |= pl_set_kernel_arg_float(pl_ctx, kernel, argc++, params->x0);
    error |= pl_set_kernel_arg_float(pl_ctx, kernel, argc++, params->y0);
    error |= pl_set_kernel_arg_float(pl_ctx, kernel, argc++, params->lon0 * DEG_TO_RAD);
    error |= pl_set_kernel_arg_float(pl_ctx, kernel, argc++, cosphi1);
    
    if (error != CL_SUCCESS)
        return error;
    
    return _pl_enqueue_kernel_1d(pl_ctx->queue, kernel, vec_count);
}

cl_int pl_run_kernel_inverse_geodesic( PLContext *pl_ctx, cl_kernel inv_kernel,
        PLInverseGeodesicBuffer *pl_buf, float *dist_out, PLSpheroid pl_ell, double scale) {
	int argc = 0;
	cl_int error = CL_SUCCESS;
	PLSpheroidInfo info = _pl_get_spheroid_info(pl_ell);
	
	size_t xy2VecCount = ck_padding(pl_buf->xy2_count, PL_FLOAT_VECTOR_SIZE) / PL_FLOAT_VECTOR_SIZE;
	
	error |= pl_set_kernel_arg_mem(pl_ctx, inv_kernel, argc++, pl_buf->xy1_in);
	error |= pl_set_kernel_arg_mem(pl_ctx, inv_kernel, argc++, pl_buf->xy2_in);
	error |= pl_set_kernel_arg_mem(pl_ctx, inv_kernel, argc++, pl_buf->dist_out);
	error |= pl_set_kernel_arg_float(pl_ctx, inv_kernel, argc++, info.major_axis);
	
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

cl_int pl_run_kernel_forward_geodesic_fixed_distance(PLContext *pl_ctx, cl_kernel fwd_kernel, 
    PLForwardGeodesicFixedDistanceBuffer *pl_buf, float *xy_out, PLSpheroid pl_ell, double distance) {
	PLSpheroidInfo info = _pl_get_spheroid_info(pl_ell);
	
	int argc = 0;
	
	cl_int error = CL_SUCCESS;
	
	double D, sinD, cosD;
	
	D = distance / info.major_axis;
	sinD = sin(D);
	cosD = cos(D);
	
	error |= pl_set_kernel_arg_mem(pl_ctx, fwd_kernel, argc++, pl_buf->xy_in);
	error |= pl_set_kernel_arg_mem(pl_ctx, fwd_kernel, argc++, pl_buf->phi_sincos);
	error |= pl_set_kernel_arg_mem(pl_ctx, fwd_kernel, argc++, pl_buf->az_sincos);
	error |= pl_set_kernel_arg_mem(pl_ctx, fwd_kernel, argc++, pl_buf->xy_out);
	error |= pl_set_kernel_arg_float(pl_ctx, fwd_kernel, argc++, D);
	error |= pl_set_kernel_arg_float(pl_ctx, fwd_kernel, argc++, sinD);
	error |= pl_set_kernel_arg_float(pl_ctx, fwd_kernel, argc++, cosD);
	if (!_pl_spheroid_is_spherical(pl_ell)) {
		double flattening = 1.f/info.inverse_flattening;
		error |= pl_set_kernel_arg_float(pl_ctx, fwd_kernel, argc++, flattening);
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

cl_int pl_run_kernel_forward_geodesic_fixed_angle(PLContext *pl_ctx, cl_kernel fwd_kernel, 
    PLForwardGeodesicFixedAngleBuffer *pl_buf, double xy_in[2], float *xy_out, PLSpheroid pl_ell, double angle) {
    int argc = 0;
    
    size_t distVecCount = ck_padding(pl_buf->dist_count, PL_FLOAT_VECTOR_SIZE) / PL_FLOAT_VECTOR_SIZE;
    
    cl_int error = CL_SUCCESS;
    
    double sinAzimuth = sin(angle);
    double cosAzimuth = cos(angle);
    
    error |= pl_set_kernel_arg_float2(pl_ctx, fwd_kernel, argc++, xy_in);
    error |= pl_set_kernel_arg_mem(pl_ctx, fwd_kernel, argc++, pl_buf->dist_in);
    error |= pl_set_kernel_arg_mem(pl_ctx, fwd_kernel, argc++, pl_buf->xy_out);
    error |= pl_set_kernel_arg_float(pl_ctx, fwd_kernel, argc++, angle);
    error |= pl_set_kernel_arg_float(pl_ctx, fwd_kernel, argc++, sinAzimuth);
    error |= pl_set_kernel_arg_float(pl_ctx, fwd_kernel, argc++, cosAzimuth);
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

cl_int pl_run_kernel_geodesic_to_cartesian(PLContext *pl_ctx, cl_kernel g2c_kernel,
        PLDatumShiftBuffer *pl_buf, PLSpheroid pl_ell) {
    PLSpheroidInfo info = _pl_get_spheroid_info(pl_ell);

    int argc = 0;
    cl_int error = CL_SUCCESS;
    size_t vec_count = ck_padding(pl_buf->count, PL_FLOAT_VECTOR_SIZE) / PL_FLOAT_VECTOR_SIZE;
        
    error |= pl_set_kernel_arg_mem(pl_ctx, g2c_kernel, argc++, pl_buf->xy_in);
    error |= pl_set_kernel_arg_mem(pl_ctx, g2c_kernel, argc++, pl_buf->x_rw);
    error |= pl_set_kernel_arg_mem(pl_ctx, g2c_kernel, argc++, pl_buf->y_rw);
    error |= pl_set_kernel_arg_mem(pl_ctx, g2c_kernel, argc++, pl_buf->z_rw);
    
    error |= pl_set_kernel_arg_float(pl_ctx, g2c_kernel, argc++, info.ecc);
    error |= pl_set_kernel_arg_float(pl_ctx, g2c_kernel, argc++, info.ecc2);
    error |= pl_set_kernel_arg_float(pl_ctx, g2c_kernel, argc++, info.one_ecc2);
    
    error |= pl_set_kernel_arg_float(pl_ctx, g2c_kernel, argc++, info.major_axis);
    error |= pl_set_kernel_arg_float(pl_ctx, g2c_kernel, argc++, info.minor_axis);
    
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
cl_int pl_run_kernel_transform_cartesian(PLContext *pl_ctx, cl_kernel transform_kernel, 
        PLDatumShiftBuffer *pl_buf, PLDatum src_datum, PLDatum dst_datum) {
    double Rx1, Ry1, Rz1, M1, Dx1, Dy1, Dz1;
    double Rx2, Ry2, Rz2, M2, Dx2, Dy2, Dz2;
    
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
    __CLPK_doublereal work[4];
    __CLPK_integer work_n = 4;
    
    dgetrf_(&n, &n, &matrix2.elements[0][0], &n, ipiv, &info);
    if (info != 0) {
        return info;
    }
    
    dgetri_(&n, &matrix2.elements[0][0], &n, ipiv, work, &work_n, &info);
    if (info != 0) {
        return info;
    }
    
    __CLPK_doublereal alpha = 1.0;
    __CLPK_doublereal beta = 0.0;
    __CLPK_doublereal result_matrix[4][4];
    
    /* Multiply the source matrix with the inverse destination matrix */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, alpha,
          &matrix2.elements[0][0], n,
          &matrix1.elements[0][0], n,
          beta, &result_matrix[0][0], n);
    
    /* transpose the result and store single-precision */
    double tmatrix[4][4];
    int i, j;
    for (i=0; i<4; i++) {
        for (j=0; j<4; j++) {
            tmatrix[i][j] = result_matrix[j][i];
        }
    }
    
    error |= pl_set_kernel_arg_mem(pl_ctx, transform_kernel, argc++, pl_buf->x_rw);
    error |= pl_set_kernel_arg_mem(pl_ctx, transform_kernel, argc++, pl_buf->y_rw);
    error |= pl_set_kernel_arg_mem(pl_ctx, transform_kernel, argc++, pl_buf->z_rw);
    error |= pl_set_kernel_arg_float16(pl_ctx, transform_kernel, argc++, &tmatrix[0][0]);
    if (error != CL_SUCCESS)
        return error;
    
    error = clEnqueueNDRangeKernel(pl_ctx->queue, transform_kernel, 1, NULL, 
                                   &vec_count, NULL, 0, NULL, NULL);
    if (error != CL_SUCCESS)
        return error;
    
    return error;
}

cl_int pl_run_kernel_cartesian_to_geodesic(PLContext *pl_ctx, cl_kernel c2g_kernel, 
        PLDatumShiftBuffer *pl_buf, float *xy_out, PLSpheroid pl_ell) {
    PLSpheroidInfo info = _pl_get_spheroid_info(pl_ell);
    
    int argc = 0;
    cl_int error = CL_SUCCESS;
    size_t vec_count = ck_padding(pl_buf->count, PL_FLOAT_VECTOR_SIZE) / PL_FLOAT_VECTOR_SIZE;
        
    error |= pl_set_kernel_arg_mem(pl_ctx, c2g_kernel, argc++, pl_buf->x_rw);
    error |= pl_set_kernel_arg_mem(pl_ctx, c2g_kernel, argc++, pl_buf->y_rw);
    error |= pl_set_kernel_arg_mem(pl_ctx, c2g_kernel, argc++, pl_buf->z_rw);
    error |= pl_set_kernel_arg_mem(pl_ctx, c2g_kernel, argc++, pl_buf->xy_out);
    
    error |= pl_set_kernel_arg_float(pl_ctx, c2g_kernel, argc++, info.ecc);
    error |= pl_set_kernel_arg_float(pl_ctx, c2g_kernel, argc++, info.ecc2);
    error |= pl_set_kernel_arg_float(pl_ctx, c2g_kernel, argc++, info.one_ecc2);
    
    error |= pl_set_kernel_arg_float(pl_ctx, c2g_kernel, argc++, info.major_axis);
    error |= pl_set_kernel_arg_float(pl_ctx, c2g_kernel, argc++, info.minor_axis);
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
