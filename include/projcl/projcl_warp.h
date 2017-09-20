
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

PLImageBuffer *pl_load_image(PLContext *pl_ctx,
        cl_channel_order channel_order, cl_channel_type channel_type,
        size_t width, size_t height, size_t row_pitch,
        const void* pvData, int copy, int *outError);
PLImageArrayBuffer *pl_load_image_array(PLContext *pl_ctx,
        cl_channel_order channel_order, cl_channel_type channel_type,
        size_t width, size_t height, size_t row_pitch,
        size_t slice_pitch, size_t tiles_across, size_t tiles_down,
        const void *pvData, int do_copy, cl_int *outError);
void pl_unload_image(PLImageBuffer *buf);
void pl_unload_image_array(PLImageArrayBuffer *buf);

cl_int pl_sample_image(PLContext *pl_ctx, PLPointGridBuffer *grid, PLImageBuffer *img, PLImageFilter filter, unsigned char *outData);
cl_int pl_sample_image_array(PLContext *pl_ctx, PLPointGridBuffer *grid, PLImageArrayBuffer *bufs, PLImageFilter filter,
                             unsigned char *outData);

PLPointGridBuffer *pl_load_empty_grid(PLContext *pl_ctx, int count_x, int count_y, int *outError);
PLPointGridBuffer *pl_load_grid(PLContext *pl_ctx,
                                float origin_x, float width, int count_x,
                                float origin_y, float height, int count_y,
                                int *outError);
void pl_unload_grid(PLPointGridBuffer *grid);
cl_int pl_transform_grid(PLContext *pl_ctx, PLPointGridBuffer *src, PLPointGridBuffer *dst, float sx, float sy, float tx, float ty);
cl_int pl_shift_grid_datum(PLContext *pl_ctx, PLPointGridBuffer *src, PLDatum src_datum, PLSpheroid src_spheroid,
                           PLPointGridBuffer *dst, PLDatum dst_datum, PLSpheroid dst_spheroid);

cl_int pl_project_grid_forward(PLContext *pl_ctx, PLProjection proj, PLProjectionParams *params,
        PLPointGridBuffer *src, PLPointGridBuffer *dst);
cl_int pl_project_grid_reverse(PLContext *pl_ctx, PLProjection proj, PLProjectionParams *params,
        PLPointGridBuffer *src, PLPointGridBuffer *dst);
