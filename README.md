ProjCL: OpenCL-powered map projection and geodesic library
==

ProjCL is a C interface to OpenCL routines that perform various geographic
computations, including map projection, geodesic (distance) calculations, and
datum conversion. For projection calculations it is several times faster than
Proj.4 on the CPU, and could be even faster on a GPU for large batches but I
haven't actually done much GPU performance testing. For datum shifts ProjCL
is smarter than Proj.4 because it does some matrix math in advance, and generally
faster because OpenCL can utilize all cores and the CPU's vector capabilities.

Most projection routines were originally adapted from Proj.4 code, with
branches replaced with select() statements and various tweaks implemented along
the way. Unlike Proj.4, or any other project for that matter, ProjCL includes a
functioning Winkel Tripel inverse projection.

All of the routines are single-precision, since that gets you about 1m accuracy,
which is more than what I needed for Magic Maps. Double-precision should
probably be implemented at some point, but that will be painful as OpenCL
compilers tend to have half-assed support for double-precision.

The API differs from Proj.4 in that each projection gets its own pair of
functions (one forward, one inverse) with arguments only for the parameters
that apply to that projection. Text-based APIs like Proj.4's are stupid and
error-prone, and anyone who implements one thinking it's "clever" and
"flexible" should have their C compiler taken away.

A test suite covers the projection routines, and if you run it you will notice
that there are flaws in the Transverse Mercator algorithms. Charles Karney has
recently published new TM algorithms that would be nice to use here. It would
also be nice to use his new geodesic algorithms, since at the moment ProjCL can
only perform spherical distance calculations.

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

Available datums and spheroids: see src/projcl_types.h

Setup
--

    #include "projcl.h"

    cl_int error = CL_SUCCESS;

    PLContext *ctx = pl_context_init(CL_DEVICE_TYPE_CPU, &error);

    PLCode *code = pl_compile_code(ctx, "/path/to/ProjCL/kernel", 
            PL_MODULE_DATUM | PL_MODULE_GEODESIC | PL_MODULE_PROJECTION);

    error = pl_load_code(ctx, code);

Teardown
--
    pl_unload_code(ctx);
    pl_release_code(code);
    pl_context_free(ctx);

Forward projection
--

    /* get the input data from somewhere */
    /* latitude-longitude pairs */
    int count = ...;
    float *lat_lon_data = ...;

    /* load point data */
    PLProjectionBuffer *proj_buffer = pl_load_projection_data(ctx, lat_lon_data, count, 1, &error);

    /* allocate output buffer */
    float *xy_data = malloc(2 * count * sizeof(float));

    /* project forwards */
    error = pl_project_mercator(ctx, proj_buffer, xy_data, PL_SPHEROID_WGS_84, 
            1.0,  /* scale */
            0.0,  /* false northing */
            0.0); /* false easting */

    /* unload */
    pl_unload_projection_data(proj_buffer);

Inverse projection
--

    /* get the input data from somewhere */
    /* X-Y pairs */
    int count = ...;
    float *xy_data = ...;

    PLProjectionBuffer *cartesian_buffer = pl_load_projection_data(ctx, xy_data, count, 1, &error);

    float *lat_lon_data = malloc(2 * count * sizeof(float));

    error = pl_unproject_mercator(ctx, cartesian_buffer, lat_lon_data, PL_SPHEROID_WGS_84,
            1.0, 0.0, 0.0);

    pl_unload_projection_data(cartesian_buffer);

Forward geodesic: Fixed distance, multiple points, multiple angles (blast radii)
--

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

Forward geodesic: Fixed angle, single point, multiple distances (great circle)
--

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

Inverse geodesic: Many-to-many (distance table)
--

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

Datum shift
--

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

Adding a Map Projection
--

It's relatively straightforward to add a map projection to ProjCL. You just
need to...

1. Create a file kernel/pl_project_<name>.opencl in kernel/ with the projection routines

2. Create a pl_enqueue_kernel_<name> function in src/projcl_run.c

3. Create pl_project_<name> and pl_unproject_<name> functions in src/projcl_project.c

4. Add pl_project_<name> and pl_unproject_<name> prototypes to src/projcl.h

5. Add tests to test/projcl_test.c

Some tips on writing OpenCL routines:

* Use float16 arrays of 8 points for input and output
* Use "any" and "all" for break conditions
* Use "select" or the ternary operator for conditional assignments
* Use "sincos" if you need the sine and cosine of the same angle
* If you're on a Mac, get used to bisecting your code to find compilation
  errors. Apple's OpenCL implementation is a low point in the history of
  compilers.
