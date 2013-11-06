
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

static struct pl_spheroid_info_s pl_spheroid_params[] = {
    { 6370997.0,   6370997.0 },  /* SPHEROID */ 
    { 6378137.0,   6356752.31424 },  /* WGS 84 */
    { 6378137.0,   6356752.31414 }, /* GRS 80 */
    { 6377563.396, 6356256.910 }, /* Airy 1830 */
    { 6377563.0,   6356256.161 }, /* Airy 1848 */
    { 6377340.189, 6356034.448 }, /* Modified Airy */
    { 6377397.155, 6356078.963 }, /* Bessel 1841 */
    { 6378206.4,   6356583.8 }, /* Clarke 1866 */
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
	
	float t, es = info.ecc2;
	info.en[0] = C00 - es * (C02 + es * (C04 + es * (C06 + es * C08)));
	info.en[1] = es * (C22 - es * (C04 + es * (C06 + es * C08)));
	info.en[2] = (t = es * es) * (C44 - es * (C46 + es * C48));
	info.en[3] = (t *= es) * (C66 - es * C68);
	info.en[4] = t * es * C88;
	
	return info;
}

