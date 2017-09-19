
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
	PLSpheroidInfo info = { 0 };
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
	info.ec = 1. - .5 * info.one_ecc2 * (log1p(-info.ecc) - log1p(info.ecc)) / info.ecc;
	
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

    info.krueger_A = (1.0 + (0.25 + 1./64. * n2) * n2)  / (1.0 + n);

    /*
    info.krueger_alpha[0] = (-127./288. + (7891./37800. +             (72161./387072. - 18975107./50803200.*n) * n) * n) * n;
    info.krueger_alpha[1] = (281./630.  + (-1983433./1935360. +      (12769./28800. + 148003883./174182400.*n) * n) * n) * n;
    info.krueger_alpha[2] = (15061./26880. + (167603./181440. + (-67102379./29030400. + 79682431./79833600.*n) * n) * n) * n;
    info.krueger_alpha[3] = (-179./168. + (6601661./7257600.  + (   97445./49896.+ 40176129013./7664025600.*n) * n) * n) * n;
    */

    info.krueger_alpha[0] = (.5 + (-2./3. + (.3125 +  (41./180. + info.krueger_alpha[0]) * n) * n) * n) * n;
    info.krueger_alpha[1] =       (13./48.+ (-.6   + (557./1440.+ info.krueger_alpha[1]) * n) * n) * n  * n;
    info.krueger_alpha[2] =             (61./240.  - (103./140. + info.krueger_alpha[2]) * n) * n  * n  * n;
    info.krueger_alpha[3] =                      (49561./161280.+ info.krueger_alpha[3]) * n  * n  * n  * n;

    /*
    info.krueger_alpha[4] = (34729./80640. + (-3418889./1995840. + (14644087./9123840. + 2605413599./622702080.*n) * n) * n) * n * n4;
    info.krueger_alpha[5] = (212378941./319334400. + (-30705481./10378368.         + 175214326799./58118860800.*n) * n) * n  * n * n4;
    */

    /*
    info.krueger_beta[0] = (-81./512. + (96199./604800. + (-5406467./38707200. +   7944359./67737600.*n) * n) * n) * n;
    info.krueger_beta[1] = (-46./105. + (-1118711./3870720. + (51841./1209600. + 24749483./348364800.*n) * n) * n) * n;
    info.krueger_beta[2] = (-209./4480. + (5569./90720. + (9261899./58060800.    - 6457463./17740800.*n) * n) * n) * n;
    info.krueger_beta[3] = (-11./504. + (-830251./7257600. + (466511./2494800.+324154477./7664025600.*n) * n) * n) * n;
    */

    info.krueger_beta[0] = (.5 + (-2./3. + (37./96. + (-1./360. + info.krueger_beta[0]) * n) * n) * n) * n;
    info.krueger_beta[1] =       (1./48. + (1./15. + (-437./1440.+info.krueger_beta[1]) * n) * n) * n  * n;
    info.krueger_beta[2] =               (17./480. + (-37./840.  +info.krueger_beta[2]) * n) * n  * n  * n;
    info.krueger_beta[3] =                         (4397./161280.+info.krueger_beta[3]) * n  * n  * n  * n;

    /*
    info.krueger_beta[4] = (4583./161280. + (-108847./3991680. + (-8005831./63866880. + 22894433./124540416.*n) * n) * n) * n * n4;
    info.krueger_beta[5] =        (20648693./638668800. + (-16363163./518918400. - 2204645983./12915302400. *n) * n) * n  * n * n4;
    */

	return info;
}

