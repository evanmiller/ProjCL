
typedef enum PLImageFilter {
    PL_IMAGE_FILTER_NEAREST_NEIGHBOR,
    PL_IMAGE_FILTER_BILINEAR,
    PL_IMAGE_FILTER_BICUBIC,
    PL_IMAGE_FILTER_QUASI_BICUBIC
} PLImageFilter;

#define PL_MODULE_NEAREST_NEIGHBOR              (1 << 24)
#define PL_MODULE_BILINEAR                      (1 << 25)
#define PL_MODULE_BICUBIC                       (1 << 26)
#define PL_MODULE_QUASI_BICUBIC                 (1 << 27)

#define PL_MODULE_FILTER                        (0xF << 24)

typedef struct PLImageBuffer_s {
    cl_mem      image;
    cl_image_format image_format;
    cl_image_desc   image_desc;
} PLImageBuffer;

typedef struct PLImageArrayBuffer_s {
    cl_mem      image;
    cl_image_format image_format;
    cl_image_desc   image_desc;
    size_t      tiles_across;
    size_t      tiles_down;
} PLImageArrayBuffer;

/* The basic strategy for image-warping (reprojecting an image from one
 * coordinate system to another) is transform a grid of points from the
 * destination image's coordinate space back to the source image's coordinate
 * space, and then sample the source image at those individual points. Because
 * the transformed points are unlikely to fall exactly in the middle of a
 * source pixel, a "filter" can be used for interpolating the destination pixel
 * value (nearest neighbor, linear interpolation, bicubic, etc).
 *
 * The algorithm thus requires only a set of operations for each destination
 * pixel, so it works especially well for creating destination images that
 * are smaller than the source. One disadvantage is that if the destination
 * projection is skewed relative to the source projection (e.g. a conic
 * projection when the source is Mercator), performance suffers a bit on CPU
 * devices due to poor cache characteristics.
 *
 * Here's a typical sequence of steps for doing a warp:
 *
 * 1. Create a PLImageBuffer or PLImageArrayBuffer containing the raw data
 *    of the source image, in a format described by cl_channel_order and
 *    cl_channel_type. (See the OpenCL docs for valid values; note that an
 *    error will be returned if the combination is not supported by your
 *    hardware.) An array buffer should consist of tiles in row-first order.
 *    The array format is often necessary given hardware limitations on
 *    image size (e.g. 4096 x 4096)
 *
 * 2. Create a grid of points in X/Y image coordinates using pl_load_grid.
 *    The grid should have the height and width of the destination image.
 *
 * 3. Optionally, shift the grid of (destination) points to the projected
 *    coordinate space using pl_transform_grid.
 *
 * 4. Create an empty grid of the same size to store the geographic coordinates
 *    using pl_load_empty_grid.
 *
 * 5. Perform a reverse projection of the grid using pl_project_grid_reverse,
 *    storing the result in the empty grid you just created.
 *
 * 6. Optionally, perform a datum shift on the geographic coordinates using
 *    pl_shift_grid_datum.
 *
 * 7. Project the datum-shifted geographic coordinates forward into the source
 *    image's projection space using pl_project_grid_forward.
 *
 * 8. Finally, read out the image data at the destination-projected points using
 *    pl_sample_image or pl_sample_image_array. The data will be in the same
 *    format (channel order/type) as the original image.
 *
 * Most of the operations are non-destructive so you can manage (and re-use)
 * buffers as you deem appropriate.
 *
 * If the source and destination images will use the same projection, you can
 * skip steps 4-7 and just use an appropriate affine transform in step 3.
 */

PLImageBuffer *pl_load_image(PLContext *pl_ctx,
        cl_channel_order channel_order, cl_channel_type channel_type,
        size_t width, size_t height, size_t row_pitch,
        const void* pvData, cl_bool copy, cl_int *outError);
void pl_unload_image(PLImageBuffer *buf);

PLImageArrayBuffer *pl_load_image_array(PLContext *pl_ctx,
        cl_channel_order channel_order, cl_channel_type channel_type,
        size_t width, size_t height, size_t row_pitch,
        size_t slice_pitch, size_t tiles_across, size_t tiles_down,
        const void *pvData, cl_bool copy, cl_int *outError);
void pl_unload_image_array(PLImageArrayBuffer *buf);

cl_int pl_sample_image(PLContext *pl_ctx, PLPointGridBuffer *grid, PLImageBuffer *img,
        PLImageFilter filter, unsigned char *outData);
cl_int pl_sample_image_array(PLContext *pl_ctx, PLPointGridBuffer *grid, PLImageArrayBuffer *bufs,
        PLImageFilter filter, unsigned char *outData);

PLPointGridBuffer *pl_load_empty_grid(PLContext *pl_ctx, size_t count_x, size_t count_y, int *outError);
PLPointGridBuffer *pl_load_grid(PLContext *pl_ctx,
        double origin_x, double width, size_t count_x,
        double origin_y, double height, size_t count_y,
        int *outError);
void pl_unload_grid(PLPointGridBuffer *grid);

cl_int pl_transform_grid(PLContext *pl_ctx, PLPointGridBuffer *src, PLPointGridBuffer *dst,
        double sx, double sy, double tx, double ty);
cl_int pl_shift_grid_datum(PLContext *pl_ctx,
        PLPointGridBuffer *src, PLDatum src_datum, PLSpheroid src_spheroid,
        PLPointGridBuffer *dst, PLDatum dst_datum, PLSpheroid dst_spheroid);

cl_int pl_project_grid_forward(PLContext *pl_ctx, PLProjection proj, PLProjectionParams *params,
        PLPointGridBuffer *src, PLPointGridBuffer *dst);
cl_int pl_project_grid_reverse(PLContext *pl_ctx, PLProjection proj, PLProjectionParams *params,
        PLPointGridBuffer *src, PLPointGridBuffer *dst);
