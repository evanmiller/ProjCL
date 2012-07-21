
__kernel void pl_sample_image_linear(
                                     __global float2     *xy_grid,
                                     __read_only image2d_t  image_in,
                                     __write_only image2d_t image_out) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    int j_size = get_global_size(1);
    
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;
    
    float2 coords = xy_grid[i*j_size+j];
    
    float4 pixel = read_imagef(image_in, sampler, (float2)(coords.x + 0.5f, coords.y + 0.5f));
        
    write_imagef(image_out, (int2)(j, i), pixel);
}

__kernel void pl_sample_image_array_linear(
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
    
    int rowATile = ((int)coords.y) / image_in_height;
    int rowA = ((int)coords.y) - rowATile * image_in_height;
    int rowBTile = ((int)ceil(coords.y)) / image_in_height;
    int rowB = ((int)ceil(coords.y)) - rowBTile * image_in_height;
    
    int colATile = ((int)coords.x) / image_in_width;
    int colA = ((int)coords.x) - colATile * image_in_width;
    int colBTile = ((int)ceil(coords.x)) / image_in_width;
    int colB = ((int)ceil(coords.x)) - colBTile * image_in_width;
    
    float4 pixelAA = read_imagef(image_in, sampler, (int4)(colA, rowA, colATile + rowATile * tile_cols_count, 0));
    float4 pixelAB = read_imagef(image_in, sampler, (int4)(colB, rowA, colBTile + rowATile * tile_cols_count, 0));
    float4 pixelBA = read_imagef(image_in, sampler, (int4)(colA, rowB, colATile + rowBTile * tile_cols_count, 0));
    float4 pixelBB = read_imagef(image_in, sampler, (int4)(colB, rowB, colBTile + rowBTile * tile_cols_count, 0));
    
    float2 pos = (float2)(coords.x - (int)coords.x, coords.y - (int)coords.y);
    
    float4 pixel = mix(mix(pixelAA, pixelAB, pos.x), mix(pixelBA, pixelBB, pos.x), pos.y);
    
    write_imagef(image_out, (int2)(j, i), pixel);
}

