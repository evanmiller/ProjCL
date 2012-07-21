
__kernel void pl_sample_image_quasi_bicubic(
                                            __global float2     *xy_grid,
                                            __read_only image2d_t  image_in,
                                            __write_only image2d_t image_out
                                            ) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    int j_size = get_global_size(1);
    
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    float2 coords = xy_grid[i*j_size+j];
    
    int rowA = floor(coords.y) - 1;
    int rowB = rowA + 1;
    int rowC = rowB + 1;
    int rowD = rowC + 1;
    
    int colA = floor(coords.x) - 1;
    int colB = colA + 1;
    int colC = colB + 1;
    int colD = colC + 1;
    
    float4 pixelAB = read_imagef(image_in, sampler, (int2)(colB, rowA));
    float4 pixelAC = read_imagef(image_in, sampler, (int2)(colC, rowA));
    
    float4 pixelBA = read_imagef(image_in, sampler, (int2)(colA, rowB));
    float4 pixelBB = read_imagef(image_in, sampler, (int2)(colB, rowB));
    float4 pixelBC = read_imagef(image_in, sampler, (int2)(colC, rowB));
    float4 pixelBD = read_imagef(image_in, sampler, (int2)(colD, rowB));
    
    float4 pixelCA = read_imagef(image_in, sampler, (int2)(colA, rowC));
    float4 pixelCB = read_imagef(image_in, sampler, (int2)(colB, rowC));
    float4 pixelCC = read_imagef(image_in, sampler, (int2)(colC, rowC));
    float4 pixelCD = read_imagef(image_in, sampler, (int2)(colD, rowC));
    
    float4 pixelDB = read_imagef(image_in, sampler, (int2)(colB, rowD));
    float4 pixelDC = read_imagef(image_in, sampler, (int2)(colC, rowD));
    
    float2 pos = (float2)(coords.x - colB, coords.y - rowB);
    
    float4 pixel = clamp(pl_interpolate_cubic4(pos.y, 
                                            mix(pixelAB, pixelAC, pos.x),
                                            pl_interpolate_cubic4(pos.x, pixelBA, pixelBB, pixelBC, pixelBD),
                                            pl_interpolate_cubic4(pos.x, pixelCA, pixelCB, pixelCC, pixelCD),
                                            mix(pixelDB, pixelDC, pos.x)
                                            ), (float4)(0.f), (float4)(255.f));
    write_imagef(image_out, (int2)(j, i), pixel);
}

__kernel void pl_sample_image_array_quasi_bicubic(
                                                  __global float2     *xy_grid,
                                                  __read_only image3d_t  image_in,
                                                  int tile_cols_count,
                                                  __write_only image2d_t image_out
                                                  ) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    int j_size = get_global_size(1);
    
    int image_in_width = get_image_width(image_in);
    int image_in_height = get_image_height(image_in);
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    float2 coords = xy_grid[i*j_size+j];
    
    int rowATile = ((int)coords.y - 1) / image_in_height;
    int rowA = ((int)coords.y - 1) - rowATile * image_in_height;
    int rowBTile = ((int)coords.y) / image_in_height;
    int rowB = ((int)coords.y) - rowBTile * image_in_height;
    int rowCTile = ((int)coords.y + 1) / image_in_height;
    int rowC = ((int)coords.y + 1) - rowCTile * image_in_height;
    int rowDTile = ((int)coords.y + 2) / image_in_height;
    int rowD = ((int)coords.y + 2) - rowDTile * image_in_height;
    
    int colATile = ((int)coords.x - 1) / image_in_width;
    int colA = ((int)coords.x - 1) - colATile * image_in_width;
    int colBTile = ((int)coords.x) / image_in_width;
    int colB = ((int)coords.x) - colBTile * image_in_width;
    int colCTile = ((int)coords.x + 1) / image_in_width;
    int colC = ((int)coords.x + 1) - colCTile * image_in_width;
    int colDTile = ((int)coords.x + 2) / image_in_width;
    int colD = ((int)coords.x + 2) - colDTile * image_in_width;
    
    float4 pixelAB = read_imagef(image_in, sampler, (int4)(colB, rowA, colBTile + rowATile * tile_cols_count, 0));
    float4 pixelAC = read_imagef(image_in, sampler, (int4)(colC, rowA, colCTile + rowATile * tile_cols_count, 0));
    
    float4 pixelBA = read_imagef(image_in, sampler, (int4)(colA, rowB, colATile + rowBTile * tile_cols_count, 0));
    float4 pixelBB = read_imagef(image_in, sampler, (int4)(colB, rowB, colBTile + rowBTile * tile_cols_count, 0));
    float4 pixelBC = read_imagef(image_in, sampler, (int4)(colC, rowB, colCTile + rowBTile * tile_cols_count, 0));
    float4 pixelBD = read_imagef(image_in, sampler, (int4)(colD, rowB, colDTile + rowBTile * tile_cols_count, 0));
    
    float4 pixelCA = read_imagef(image_in, sampler, (int4)(colA, rowC, colATile + rowCTile * tile_cols_count, 0));
    float4 pixelCB = read_imagef(image_in, sampler, (int4)(colB, rowC, colBTile + rowCTile * tile_cols_count, 0));
    float4 pixelCC = read_imagef(image_in, sampler, (int4)(colC, rowC, colCTile + rowCTile * tile_cols_count, 0));
    float4 pixelCD = read_imagef(image_in, sampler, (int4)(colD, rowC, colDTile + rowCTile * tile_cols_count, 0));
    
    float4 pixelDB = read_imagef(image_in, sampler, (int4)(colB, rowD, colBTile + rowDTile * tile_cols_count, 0));
    float4 pixelDC = read_imagef(image_in, sampler, (int4)(colC, rowD, colCTile + rowDTile * tile_cols_count, 0));
    
    float2 pos = (float2)(coords.x - (int)coords.x, coords.y - (int)coords.y);
    
    float4 pixel = clamp(pl_interpolate_cubic4(pos.y,
                                               mix(pixelAB, pixelAC, pos.x),
                                               pl_interpolate_cubic4(pos.x, pixelBA, pixelBB, pixelBC, pixelBD),
                                               pl_interpolate_cubic4(pos.x, pixelCA, pixelCB, pixelCC, pixelCD),
                                               mix(pixelDB, pixelDC, pos.x)
                                               ), (float4)(0.f), (float4)(255.f));
    write_imagef(image_out, (int2)(j, i), pixel);
}
