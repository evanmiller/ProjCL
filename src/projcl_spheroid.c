
#include <projcl/projcl.h>
#include "projcl_spheroid.h"
#include <math.h>

#define C00 1.
#define C02 .25
#define C04 .046875
#define C06 .01953125
#define C08 .01068115234375
#define C22 .75
#define C44 .46875
#define C46 .01302083333333333333
#define C48 .00712076822916666666
#define C66 .36458333333333333333 
#define C68 .00569661458333333333
#define C88 .3076171875

#define P00 .33333333333333333333
#define P01 .17222222222222222222
#define P02 .10257936507936507936
#define P10 .06388888888888888888
#define P11 .06640211640211640211
#define P20 .01641501294219154443


static struct pl_spheroid_info_s pl_spheroid_params[] = {
    { 6370997.0,   6370997.0 },  /* SPHEROID */ 
    { 6378137.0,   6356752.31424 },  /* WGS 84 */
    { 6378137.0,   6356752.31414 }, /* GRS 80 */
    { 6377563.396, 6356256.910 }, /* Airy 1830 */
    { 6377563.0,   6356256.161 }, /* Airy 1848 */
    { 6377340.189, 6356034.448 }, /* Modified Airy */
    { 6377397.155, 6356078.963 }, /* Bessel 1841 */
    { 6378206.4,   6356583.8 }, /* Clarke 1866 */
    { 6378249.145, 6356514.870 }, /* Clarke 1880 (RGS) */
    { 6378160.0,   6356774.7192 }, /* GRS 1967 */
    { 6378137.0, 6378137.0 }, /* WGS 84 Major Auxiliary */
    { 6378388.0, 6356911.9 } /* Hayford / International 1924 */
};


int _pl_spheroid_is_spherical(PLSpheroid ell) {
    return ell == PL_SPHEROID_SPHERE || ell == PL_SPHEROID_WGS_84_MAJOR_AUXILIARY_SPHERE;
}

PLSpheroidInfo _pl_get_spheroid_info(PLSpheroid pl_ell) {
	PLSpheroidInfo info;
	info.tag = pl_ell;
	info.minor_axis = NAN;
	info.inverse_flattening = HUGE_VAL;
    info.major_axis = pl_spheroid_params[pl_ell].major_axis;
    info.minor_axis = pl_spheroid_params[pl_ell].minor_axis;
    if (info.major_axis > info.minor_axis) {
        info.inverse_flattening = 1 / (1 - info.minor_axis / info.major_axis);
    }
	info.one_ecc2 = (info.minor_axis * info.minor_axis) / (info.major_axis * info.major_axis);
	info.ecc2 = 1. - info.one_ecc2;
	info.ecc = sqrt(info.ecc2);
	info.ec = 1. - .5 * info.one_ecc2 * log((1. - info.ecc) / (1. + info.ecc)) / info.ecc;
	
	double t, es = info.ecc2;
	info.en[0] = C00 - es * (C02 + es * (C04 + es * (C06 + es * C08)));
	info.en[1] = es * (C22 - es * (C04 + es * (C06 + es * C08)));

    info.apa[0] = es * P00;

	info.en[2] = (t = es * es) * (C44 - es * (C46 + es * C48));

    info.apa[0] += t * P01;
    info.apa[1] = t * P10; 

	info.en[3] = (t *= es) * (C66 - es * C68);
	info.en[4] = t * es * C88;

    info.apa[0] += t * P02;
    info.apa[1] += t * P11;
    info.apa[2] = t * P20;

    double n = (info.major_axis - info.minor_axis) / (info.major_axis + info.minor_axis);
    double n2 = n * n;
    double n3 = n2 * n;
    double n4 = n2 * n2;

    info.kruger_a = (1.0 + 0.25 * n2 + n4 / 64.0) / (1.0 + n);

    /* alpha */
    info.kruger_coef[0] = 0.5 * n - 2.0/3.0 * n2 + 0.3125 * n3 + 41.0/180.0 * n4;
    info.kruger_coef[1] = 13.0/48.0 * n2 - 0.6 * n3 + 557.0 / 1440.0 * n4;
    info.kruger_coef[2] = 61.0/240.0 * n3 - 103.0/140.0 * n4;
    info.kruger_coef[3] = 49561.0 / 161280.0 * n4;

    /* beta */
    info.kruger_coef[4] = 0.5 * n - 2.0/3.0 * n2 + 37.0 / 96.0 * n3 - 1.0 / 360.0 * n4;
    info.kruger_coef[5] = 1.0/48.0 * n2 + 1.0 / 15.0 * n3 - 437.0 / 1440.0 * n4;
    info.kruger_coef[6] = 17.0/480.0 * n3 - 37.0/840.0 * n4;
    info.kruger_coef[7] = 4397.0 / 161280.0 * n4;

	return info;
}

