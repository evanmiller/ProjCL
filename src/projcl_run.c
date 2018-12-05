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
typedef double __CLPK_doublereal;
#endif

#define EPS7 1.e-7

#define SEC_TO_RAD 4.84813681109535993589914102357e-6

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


static double _pl_mlfn(double phi, double sphi, double cphi, float *en) {
	cphi *= sphi;
	sphi *= sphi;
	return(en[0] * phi - cphi * (en[1] + sphi*(en[2] + sphi*(en[3] + sphi*en[4]))));
}

static double _pl_qsfn(double sinphi, double e, double one_es) {
	double con = e * sinphi;
	return (one_es * (sinphi / (1.0 - con * con) -
					  (.5 / e) * (log1p(-con) - log1p(con))));
}

static double _pl_msfn(double sinphi, double cosphi, double es) {
	return (cosphi / sqrt(1.0 - es * sinphi * sinphi));
}

static double _pl_tsfn(double phi, double sinphi, double e) {
	double con = e * sinphi;
	return (tan(.5 * (M_PI_2 - phi)) /
            pow((1.0 - con) / (1.0 + con), .5 * e));
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

cl_int pl_read_buffer_2(cl_command_queue queue, cl_mem xy_out_buf, float *x_out, float *y_out, size_t out_count_bytes) {
    cl_int error;
    // out_count_bytes is in bytes, but clEnqueueReadBufferRect counts stuff in
    // actual array elements, so...
    size_t out_count = out_count_bytes / sizeof(float);

	size_t buffer_origin[3] = {0,0,0};
	size_t host_origin[3] = {0,0,0};
	size_t bh_region[3] = {sizeof(float), out_count/2, 1};
	// First, the x's
	error = clEnqueueReadBufferRect(queue,
					xy_out_buf,
					CL_TRUE,  // non-blocking
					buffer_origin,
					host_origin,
					bh_region,
					2*sizeof(float), // buffer_row_pitch,
					0,               // buffer_slice_pitch,
					1*sizeof(float), // host_row_pitch,
					0,               // host_slice_pitch,
					x_out,
					0,     // num_events_in_wait_list,
					NULL,  // event_wait_list,
					NULL   // event
					);
	if (error != CL_SUCCESS)
		return error;
	
	// Just to be paranoid, do this operation befor queueing the next
	// one...
	error = clFinish(queue);
	if (error != CL_SUCCESS)
		return error;

	// Then, the y's, which are like the x's with minor offsets
	buffer_origin[0] = 1 * sizeof(float);
	error = clEnqueueReadBufferRect(queue,
					xy_out_buf,
					CL_TRUE,  // non-blocking
					buffer_origin,
					host_origin,
					bh_region,
					2*sizeof(float), // buffer_row_pitch,
					0,               // buffer_slice_pitch,
					1*sizeof(float), // host_row_pitch,
					0,               // host_slice_pitch,
					y_out,
					0,     // num_events_in_wait_list,
					NULL,  // event_wait_list,
					NULL   // event
					);

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
    if (proj == PL_PROJECT_ALBERS_EQUAL_AREA) {
        error = pl_enqueue_kernel_albers_equal_area(pl_ctx, kernel, params, xy_in, xy_out, count);
    } else if (proj == PL_PROJECT_AMERICAN_POLYCONIC) {
        error = pl_enqueue_kernel_american_polyconic(pl_ctx, kernel, params, xy_in, xy_out, count);
    } else if (proj == PL_PROJECT_LAMBERT_CONFORMAL_CONIC) {
        error = pl_enqueue_kernel_lambert_conformal_conic(pl_ctx, kernel, params, xy_in, xy_out, count);
    } else if (proj == PL_PROJECT_LAMBERT_AZIMUTHAL_EQUAL_AREA) {
        error = pl_enqueue_kernel_lambert_azimuthal_equal_area(pl_ctx, kernel, params, xy_in, xy_out, count);
    } else if (proj == PL_PROJECT_MERCATOR) {
        error = pl_enqueue_kernel_mercator(pl_ctx, kernel, params, xy_in, xy_out, count);
    } else if (proj == PL_PROJECT_OBLIQUE_STEREOGRAPHIC) {
        error = pl_enqueue_kernel_oblique_stereographic(pl_ctx, kernel, params, xy_in, xy_out, count);
    } else if (proj == PL_PROJECT_ROBINSON) {
        error = pl_enqueue_kernel_robinson(pl_ctx, kernel, params, xy_in, xy_out, count);
    } else if (proj == PL_PROJECT_TRANSVERSE_MERCATOR) {
        error = pl_enqueue_kernel_transverse_mercator(pl_ctx, kernel, params, xy_in, xy_out, count);
    } else if (proj == PL_PROJECT_WINKEL_TRIPEL) {
        error = pl_enqueue_kernel_winkel_tripel(pl_ctx, kernel, params, xy_in, xy_out, count);
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
	error = _pl_set_kernel_args(kernel, xy_in, xy_out, count, &info, &argc);
	
	float phi1 = params->rlat1 * DEG_TO_RAD;
	float phi2 = params->rlat2 * DEG_TO_RAD;
	float phi0 = params->lat0 * DEG_TO_RAD;
	float lam0 = params->lon0 * DEG_TO_RAD;
    float k0 = params->scale * info.major_axis;
    float x0 = params->x0;
    float y0 = params->y0;
    
	float rho0, cf, nf;
	double c, n;
    double sinphi, cosphi;
	
	int secant; /* secant cone */
	
	n = sinphi = sin(phi1);
	cosphi = cos(phi1);
	secant = fabs(phi1 - phi2) >= EPS7;
	
	if (_pl_spheroid_is_spherical(params->spheroid)) {
		if (secant) {
			n = .5 * (n + sin(phi2));
		}
		c = cosphi * cosphi / n + 2 * sinphi;
		rho0 = sqrt((c - 2 * sin(phi0))/n);
	} else {
		double ml1, m1;
		
		m1 = _pl_msfn(sinphi, cosphi, info.ecc2);
		ml1 = _pl_qsfn(sinphi, info.ecc, info.one_ecc2);
		if (secant) { 
			double ml2, m2;
			
			sinphi = sin(phi2);
			cosphi = cos(phi2);
			m2 = _pl_msfn(sinphi, cosphi, info.ecc2);
			ml2 = _pl_qsfn(sinphi, info.ecc, info.one_ecc2);
			n = (m1 * m1 - m2 * m2) / (ml2 - ml1);
		}
        
		c = m1 * m1 / n + ml1;
		rho0 = sqrt((c - _pl_qsfn(sin(phi0), info.ecc, info.one_ecc2))/n);
	}
    cf = c;
    nf = n;
	
	if (!_pl_spheroid_is_spherical(params->spheroid)) {
		error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &info.ec);
	}
    error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &k0);
    error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &x0);
    error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &y0);
	error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &lam0);
	error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &rho0);
	error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &cf);
	error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &nf);
    
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
	error = _pl_set_kernel_args(kernel, xy_in, xy_out, count, &info, &offset);
	
	float phi0 = params->lat0 * DEG_TO_RAD;
	float lambda0 = params->lon0 * DEG_TO_RAD;
	
	float ml0 = _pl_mlfn(phi0, sin(phi0), cos(phi0), info.en);
	
	float k0 = params->scale * info.major_axis;
    float x0 = params->x0;
    float y0 = params->y0;
	
	error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &k0);
    error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &x0);
    error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &y0);
	error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &phi0);
	error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &lambda0);
    if (!_pl_spheroid_is_spherical(params->spheroid)) {
        error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &ml0);
        error |= clSetKernelArg(kernel, offset++, sizeof(cl_float8), info.en);
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
	error = _pl_set_kernel_args(kernel, xy_in, xy_out, count, &info, &offset);
    
	float phi0 = params->lat0 * DEG_TO_RAD;
	float lambda0 = params->lon0 * DEG_TO_RAD;
    float k0 = params->scale * info.major_axis;
    float x0 = params->x0;
    float y0 = params->y0;
    float qp = _pl_qsfn(1.f, info.ecc, info.one_ecc2);

    double sinPhi = sin(phi0);
    float sinB1 = _pl_qsfn(sinPhi, info.ecc, info.one_ecc2) / qp;
    float cosB1 = sqrt(1.0 - sinB1 * sinB1);

	error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &k0);
    error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &x0);
    error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &y0);
    error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &phi0);
    error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &lambda0);
	error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &qp);
	error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &sinB1);
    error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &cosB1);
    if (!_pl_spheroid_is_spherical(params->spheroid)) {
        float rq = sqrtf(0.5f * qp);
        float dd = cos(phi0) / (sqrt(1.0 - info.ecc2 * sinPhi * sinPhi) * rq * cosB1);
        float ymf = rq / dd;
        float xmf = rq * dd;

        error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &rq);
        error |= clSetKernelArg(kernel, offset++, sizeof(cl_float4), info.apa);

        error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &dd);
        error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &xmf);
        error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &ymf);
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
    error = _pl_set_kernel_args(kernel, xy_in, xy_out, count, &info, &offset);
    
    float lambda0 = params->lon0 * DEG_TO_RAD;
    float phi0 = params->lat0 * DEG_TO_RAD;
    float phi1 = params->rlat1 * DEG_TO_RAD;
    float phi2 = params->rlat2 * DEG_TO_RAD;
    float x0 = params->x0;
    float y0 = params->y0;
    float k0 = params->scale * info.major_axis;
    
    float rho0, c, n, sinphi1, cosphi1, sinphi2;
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
            n = log(cosphi1 / cos(phi2)) / log(tan(M_PI_4 + .5 * phi2) / tan(M_PI_4 + .5 * phi1));
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

cl_int pl_enqueue_kernel_mercator(PLContext *pl_ctx, cl_kernel kernel,
        PLProjectionParams *params, cl_mem xy_in, cl_mem xy_out, size_t count) {
	cl_int error;
	cl_int offset = 0;
	size_t vec_count = ck_padding(count, PL_FLOAT_VECTOR_SIZE) / PL_FLOAT_VECTOR_SIZE;
	
	PLSpheroidInfo info = _pl_get_spheroid_info(params->spheroid);
	error = _pl_set_kernel_args(kernel, xy_in, xy_out, count, &info, &offset);
	
	float k0 = params->scale * info.major_axis;
    float x0 = params->x0;
    float y0 = params->y0;
	
	error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &k0);
    error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &x0);
    error |= clSetKernelArg(kernel, offset++, sizeof(cl_float), &y0);
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

	error = _pl_set_kernel_args(kernel, xy_in, xy_out, count, &info, &argc);

    float lambda0 = params->lon0 * DEG_TO_RAD;
    double phi0 = params->lat0 * DEG_TO_RAD;

    double sinPhi0, cosPhi0;
    sinPhi0 = sin(phi0);
    cosPhi0 = cos(phi0);

    float sinPhiC0, cosPhiC0;
    float scale_r2 = 2.0 * params->scale * info.major_axis * sqrt(info.one_ecc2) / (1.0 - info.ecc2 * sinPhi0 * sinPhi0);
    float x0 = params->x0;
    float y0 = params->y0;

	error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &scale_r2);
	error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &x0);
	error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &y0);

	if (!_pl_spheroid_is_spherical(params->spheroid)) {
        double c0 = sqrt(1.0 + info.ecc2 * cosPhi0 * cosPhi0 * cosPhi0 * cosPhi0 / info.one_ecc2);
        double phiC0 = asin(sinPhi0 / c0);
        sinPhiC0 = sin(phiC0);
        cosPhiC0 = cos(phiC0);

        float c0_f = c0;
        double k0 = tan(0.5 * phiC0 + M_PI_4) / (
                pow(tan(0.5 * phi0 + M_PI_4), c0) *
                pow((1.-info.ecc * sinPhi0)/(1.+info.ecc*sinPhi0), 0.5 * c0 * info.ecc) );

        float log_k0 = log(k0);

		error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &c0_f);
		error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &log_k0);
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

cl_int pl_enqueue_kernel_robinson(PLContext *pl_ctx, cl_kernel kernel,
        PLProjectionParams *params, cl_mem xy_in, cl_mem xy_out, size_t count) {
	cl_int error = CL_SUCCESS;
	cl_int argc = 0;
	
	PLSpheroidInfo info = _pl_get_spheroid_info(PL_SPHEROID_SPHERE);
    float k0 = params->scale * info.major_axis;
    float x0 = params->x0;
    float y0 = params->y0;

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

cl_int pl_enqueue_kernel_transverse_mercator(PLContext *pl_ctx, cl_kernel kernel,
        PLProjectionParams *params, cl_mem xy_in, cl_mem xy_out, size_t count) {
    cl_int error = CL_SUCCESS;
    cl_int argc = 0;
    size_t vec_count = ck_padding(count, PL_FLOAT_VECTOR_SIZE) / PL_FLOAT_VECTOR_SIZE;
    
    PLSpheroidInfo info = _pl_get_spheroid_info(params->spheroid);
    error = _pl_set_kernel_args(kernel, xy_in, xy_out, count, &info, &argc);
    
    float k0 = params->scale * info.major_axis * info.krueger_A;
    float lambda0 = params->lon0 * DEG_TO_RAD;
    float x0 = params->x0;
    float y0 = params->y0;
    
    error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &k0);
    error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &x0);
    error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &y0);
    error |= clSetKernelArg(kernel, argc++, sizeof(cl_float), &lambda0);
    if (!_pl_spheroid_is_spherical(params->spheroid)) {
        error |= clSetKernelArg(kernel, argc++, sizeof(cl_float8), info.krueger_alpha);
        error |= clSetKernelArg(kernel, argc++, sizeof(cl_float8), info.krueger_beta);
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
    
    float cosphi1 = isnan(params->rlat1) ? M_2_PI : cos(params->rlat1 * DEG_TO_RAD);
    float lambda0 = params->lon0 * DEG_TO_RAD;
    float k0 = params->scale * info.major_axis;
    float x0 = params->x0;
    float y0 = params->y0;

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
