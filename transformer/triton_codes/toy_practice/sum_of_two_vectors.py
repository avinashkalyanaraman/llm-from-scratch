import torch
import triton
import triton.language as tl

'''
Takes two vectors -- x and y
[][][][][][]
[][][][][][]

t:  t1. |  t2  |  t3
x: [][] | [][] | [][]
y: [][] | [][] | [][]

and writes
t:  t1. |  t2  |  t3
o: [][] | [][] | [][]

'''

@triton.jit #Converts the below function into a GPU kernel
def sum (x_ptr, y_ptr, output_ptr, D, D_TILE_SIZE : tl.constexpr):
    
    thread_id = tl.program_id(0)  #Which thread am I? Defines where I do my work!
    
    #Create the block pointers! These block pointers define granularity of work!
    x_block_ptr = tl.make_block_ptr (x_ptr, 
                                     shape = (D,),
                                     strides = (1,),
                                     offsets = (thread_id * D_TILE_SIZE,),
                                     block_shape = (D_TILE_SIZE,),
                                     order = (0,) )


    y_block_ptr = tl.make_block_ptr (y_ptr, 
                                     shape = (D,),
                                     strides = (1,),
                                     offsets = (thread_id * D_TILE_SIZE,),
                                     block_shape = (D_TILE_SIZE,),
                                     order = (0,) )


    output_block_ptr = tl.make_block_ptr (output_ptr, 
                                     shape = (D,),
                                     strides = (1,),
                                     offsets = (thread_id * D_TILE_SIZE,),
                                     block_shape = (D_TILE_SIZE,),
                                     order = (0,) )

    #Each thread loads its section of the data 
    my_x = tl.load (x_block_ptr, boundary_check = (0,), padding_option="zero") #zero padding prevents garbage/OOM access
    my_y = tl.load (y_block_ptr, boundary_check = (0,), padding_option="zero") #zero padding prevents garbage/OOM access

    #Sum computation by each thread on its chunk of data
    my_output = my_x + my_y #Compute my sum

    #Store the sum back
    tl.store(output_block_ptr, my_output, boundary_check = (0,))

    return

if __name__ == '__main__':
    dim_size = 6
    tile_size = 3; tile_size = triton.next_power_of_2 (tile_size) #Loads/stores should be powers of 2
    num_threads = triton.cdiv (dim_size, tile_size)

    x = torch.randn (dim_size, dtype = torch.float32, device = "cuda")
    y = torch.randn (dim_size, dtype = torch.float32, device = "cuda")
    output = torch.empty(dim_size, dtype = torch.float32, device = "cuda")

    grid = (num_threads,) #How many threads to invoke?
    sum [grid] (x, y, output, dim_size, tile_size)
    
    print (f"x : {x}")
    print (f"y : {y}")
    print (f"output : {output}")