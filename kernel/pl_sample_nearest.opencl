
__kernel void pl_sample_image_nearest(
                                      __global float2     *xy_grid,
                                      __read_only image2d_t  image_in,
                                      __write_only image2d_t image_out) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    int j_size = get_global_size(1);
    
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
    
    float2 coords = xy_grid[i*j_size+j];
    
    float4 pixel = read_imagef(image_in, sampler, (float2)(coords.x + 0.5f, coords.y + 0.5f));
    
    write_imagef(image_out, (int2)(j, i), pixel);
}

__kernel void pl_sample_image_array_nearest(
                                            __global float2 *xy_grid,
                                            __read_only image3d_t image_in,
                                            int tile_cols_count,
                                            __write_only image2d_t image_out) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    int j_size = get_global_size(1);
    
    int image_in_width = get_image_width(image_in);
    int image_in_height = get_image_height(image_in);
    
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
    
    float2 coords = xy_grid[i*j_size+j];
    
    int tile_row = (int)(coords.y + 0.5f) / image_in_height;
    int tile_col = (int)(coords.x + 0.5f) / image_in_width;
    int index = tile_col + tile_row * tile_cols_count;
    float4 pixel = read_imagef(image_in, sampler, 
                               (float4)(coords.x + 0.5f - tile_col * image_in_width, 
                                        coords.y + 0.5f - tile_row * image_in_height,
                                        index, 0.f));
    write_imagef(image_out, (int2)(j, i), pixel);
}

