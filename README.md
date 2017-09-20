[![Travis CI build status](https://travis-ci.org/evanmiller/ProjCL.svg?branch=master)](https://travis-ci.org/evanmiller/ProjCL)

ProjCL: OpenCL-powered map projection, geodesic, and image-warping library
==

ProjCL is a C interface to [OpenCL](https://en.wikipedia.org/wiki/OpenCL)
routines that perform various geographic computations, including map
projection, geodesic (distance) calculations, datum conversion, and basic image
warping (reprojection). For projection calculations it is often 4-10X faster
than [Proj.4](http://proj4.org) on the CPU, and 15-30X faster on a low-end GPU
with large batches of coordinates. For datum shifts ProjCL is smarter than
Proj.4 because it does some matrix math in advance, and generally faster
because OpenCL can utilize all cores and the CPU's vector capabilities.

Most projection routines were originally adapted from Proj.4 code, with
branches replaced with select() statements and various tweaks implemented along
the way.

All of the routines are single-precision, since that gets you about 1m accuracy,
which is more than what I needed for [Magic
Maps](https://magicmaps.evanmiller.org/). Double-precision should probably be
implemented at some point, but that will be painful as OpenCL compilers tend to
have half-hearted support for double-precision.

The API differs from Proj.4 in that projections are specified using compile-time
constants, and parameters are specified using a dedicated data structure.
Text-based C APIs like Proj.4's are prone to error in my experience.

A test suite covers the projection routines, but not the geodesic calculations,
datum shifts, or image warping. The output is checked for self-consistency, as
well as agreement with Proj.4. The code is tested to work on OS X as well as
Linux; when making a pull request, please ensure all the tests pass with
`./test/projcl_test -CPU`.

ProjCL needs more map projections. In fact, the world needs more map projections.
If you want to try your hand at one, check out "Adding a Map Projection" below.

Available projections:
* Albers Equal Area
* American Polyconic
* Lambert Azimuthal Equal Area
* Lambert Conformal Conic
* Mercator
* Robinson
* Transverse Mercator
* Winkel Tripel

Available datums and spheroids: see [include/projcl/projcl_types.h](https://github.com/evanmiller/ProjCL/blob/master/include/projcl/projcl_types.h)

Available image filters: see [include/projcl/projcl_warp.h](https://github.com/evanmiller/ProjCL/blob/master/include/projcl/projcl_warp.h)

Building
--

ProjCL requires [CMake](http://www.cmake.org) build system. To build the library do:

```
$ cmake CMakeLists.txt
$ make	
```

Setup
--

```{C}
#include <projcl/projcl.h>

cl_int error = CL_SUCCESS;

PLContext *ctx = pl_context_init(CL_DEVICE_TYPE_CPU, &error);

PLCode *code = pl_compile_code(ctx, "/path/to/ProjCL/kernel", 
        PL_MODULE_DATUM 
        | PL_MODULE_GEODESIC 
        | PL_MODULE_PROJECTION
        | PL_MODULE_WARP 
        | PL_MODULE_FILTER);

error = pl_load_code(ctx, code);
```

Teardown
--

```{C}
pl_unload_code(ctx);
pl_release_code(code);
pl_context_free(ctx);
```

Forward projection
--

```{C}
/* get the input data from somewhere */
/* latitude-longitude pairs */
int count = ...;
float *lat_lon_data = ...;

/* load point data */
PLProjectionBuffer *proj_buffer = pl_load_projection_data(ctx, lat_lon_data, count, 1, &error);

/* allocate output buffer */
float *xy_data = malloc(2 * count * sizeof(float));

/* Set some params */
PLProjectionParams *params = pl_params_init();
pl_params_set_scale(1.0);
pl_params_set_spheroid(PL_SPHEROID_WGS_84);
pl_params_set_false_northing(0.0);
pl_params_set_false_easting(0.0);

/* project forwards */
error = pl_project_points_forward(ctx, PL_PROJECT_MERCATOR, params, proj_buffer, xy_data);

/* unload */
pl_unload_projection_data(proj_buffer);
pl_params_free(params);
```

Inverse projection
--

```{C}
/* get the input data from somewhere */
/* X-Y pairs */
int count = ...;
float *xy_data = ...;

PLProjectionBuffer *cartesian_buffer = pl_load_projection_data(ctx, xy_data, count, 1, &error);

float *lat_lon_data = malloc(2 * count * sizeof(float));

PLProjectionParams *params = pl_params_init();
pl_params_set_scale(1.0);
pl_params_set_spheroid(PL_SPHEROID_WGS_84);
pl_params_set_false_northing(0.0);
pl_params_set_false_easting(0.0);

error = pl_project_points_reverse(ctx, PL_PROJECT_MERCATOR, params, cartesian_buffer, lat_lon_data);

pl_unload_projection_data(cartesian_buffer);
```

Datum shift
--

```{C}
/* load lon-lat coordinates */
int count = ...;
float *xy_in = ...;
PLDatumShiftBuffer *buf = pl_load_datum_shift_data(ctx, PL_SPHEROID_WGS_84,
    xy_in, count, &error);

/* allocate space for result */
float *xy_out = malloc(2 * count * sizeof(float));

/* perform the shift */
error = pl_shift_datum(ctx, 
        PL_DATUM_NAD_83,  /* source */
        PL_DATUM_NAD_27,  /* destination */
        PL_SPHEROID_CLARKE_1866, /* destination spheroid */
        buf, xy_out);

pl_unload_datum_shift_data(buf);
```

Image warping
---

```{C}
#include <projcl/projcl.h>
#include <projcl/projcl_warp.h>

void *src_image_data = ...;
void *dst_image_data = malloc(...);

PLProjectionParams *src_params = pl_params_init();
PLProjectionParams *dst_params = pl_params_init();
/* set projection parameters here... */


/* load the source image (raw pixel data) */
PLImageBuffer *image = pl_load_image(pl_ctx,
    CL_RGBA,            /* channel order */
    CL_UNSIGNED_INT8,   /* channel type */
    1024,               /* image width */
    1024,               /* image height */
    0,                  /* row pitch (0 to calculate automatically) */
    src_image_data,
    1,                  /* copy data? */
    &error);

/* load a grid of destination points to back-project to the source */
PLPointGridBuffer *xy_grid = pl_load_grid(ctx,
    0.0,   /* X-origin */
    100.0, /* width in projected coordinate space */
    1024,  /* width in pixels */
    0.0,   /* Y-origin */
    100.0, /* height in projected coordinate space */
    1024,  /* height in pixels */
    &error);

PLPointGridBuffer *geo_grid = pl_load_empty_grid(pl_ctx, 1024, 1024, &error);

/* project destination XY coordinates back to geo */
pl_project_grid_reverse(pl_ctx, PL_PROJECT_WINKEL_TRIPEL, dst_params, xy_grid, geo_grid);

/* project geo to source XY coordinates */
pl_project_grid_forward(pl_ctx, PL_PROJECT_MERCATOR, src_params, geo_grid, xy_grid);

/* now sample the points using your favorite filter */
pl_sample_image(pl_ctx, xy_grid, image, PL_IMAGE_FILTER_BICUBIC, dst_image_data);
```

See
[include/projcl/projcl_warp.h](https://github.com/evanmiller/ProjCL/blob/master/include/projcl/projcl_warp.h)
for a more complete discussion, including how to perform datum conversion on the point grid.

Forward geodesic: Fixed distance, multiple points, multiple angles (blast radii)
--

```{C}
/* get the point data from somewhere */
float *xy_in = ...;
int xy_count = ...;

/* get the angle (azimuth) data from somewhere */
float *az_in = ...;
int az_count = ...;

/* load it up */
PLForwardGeodesicFixedDistanceBuffer *buf = pl_load_forward_geodesic_fixed_distance_data(ctx,
    xy_in, xy_count, az_in, az_count, &error);

/* allocate output buffer */
float *xy_out = malloc(2 * xy_count * az_count * sizeof(float));

/* compute */
error = pl_forward_geodesic_fixed_distance(ctx, buf, xy_out, PL_SPHEROID_SPHERE,
        1000.0 /* distance in meters */
        );

/* unload */
pl_unload_forward_geodesic_fixed_distance_data(buf);
```

Forward geodesic: Fixed angle, single point, multiple distances (great circle)
--

```{C}
int count = ...;
float *dist_in = ...;

PLForwardGeodesicFixedAngleBuffer *buf = pl_load_forward_geodesic_fixed_angle_data(ctx,
    dist_in, count, &error);

float *xy_out = malloc(2 * count * sizeof(float));
float xy_in[2] = ...;

error = pl_forward_geodesic_fixed_angle(ctx, buf, xy_in, xy_out, PL_SPHEROID_SPHERE, 
        M_PI_2 /* angle in radians */
        );

pl_unload_forward_geodesic_fixed_angle_data(buf);
```

Inverse geodesic: Many-to-many (distance table)
--

```{C}
int count1 = ...;
float *xy1_in = ...;

int count2 = ...;
float *xy2_in = ...;

float *dist_out = malloc(count1 * count2 * sizeof(float));

PLInverseGeodesicBuffer *buf = pl_load_inverse_geodesic_data(ctx, 
        xy1_in, count1, 1, xy2_in, count2, &error);

error = pl_inverse_geodesic(ctx, buf, dist_out, PL_SPHEROID_SPHERE, 
        1.0 /* scale */);

pl_unload_inverse_geodesic_data(buf);
```

Adding a Map Projection
--

It's relatively straightforward to add a map projection to ProjCL. You just
need to...

1. Create a file kernel/pl_project_&lt;name&gt;.opencl in kernel/ with the projection routines

2. Create a pl_enqueue_kernel_&lt;name&gt; function in [src/projcl_run.c](https://github.com/evanmiller/ProjCL/blob/master/src/projcl_run.c)

3. Add an entry to the `PLProjection` enum in [include/projcl/projcl_types.h](https://github.com/evanmiller/ProjCL/blob/master/include/projcl/projcl_types.h)

4. Add an entry to the `_pl_projection_info` array in [src/projcl_run.c](https://github.com/evanmiller/ProjCL/blob/master/src/projcl_run.c)

5. Add tests to [test/projcl_test.c](https://github.com/evanmiller/ProjCL/blob/master/test/projcl_test.c)

Some tips on writing OpenCL routines:

* Use float16 arrays of 8 points for input and output
* Use `any` and `all` for break conditions
* Use `select` or the ternary operator for conditional assignments
* Use `sincos` if you need the sine and cosine of the same angle
* Double-angle and half-angle formulas are your friend
* `log(tan(M_PI_4 + 0.5*x)) == asinh(tan(x))`
* `2*atan(exp(x)) - M_PI_2 == asin(tanh(x))`
