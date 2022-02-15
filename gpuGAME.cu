#define NBDKIT_API_VERSION 2
#include <nbdkit-plugin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <unordered_map>


#define THREAD_MODEL NBDKIT_THREAD_MODEL_SERIALIZE_CONNECTIONS
#define BLOCK_SZ 4096UL               // 4KB
#define SOFT_LIM (1024UL*1024*1024*4) // 4GB
#define HARD_LIM (1024UL*1024*1024)   // 1GB
#define AREA_SZ  (BLOCK_PER_AREA*BLOCK_SZ)

int err_no;
int num_gpus = -1;
int key_shift = -1;
uint64_t GAME_SIZE;
uint64_t BLOCK_PER_AREA;
cudaStream_t *read_stream;
cudaStream_t *write_stream;

std::unordered_map<uint64_t, struct area_desc*> hash_table;

struct area_desc{
    uint8_t  GPU_ID;
    uint64_t start_offset;
    uint64_t last_offset;
    uint64_t *vram_ptr;
    bool *PBV;
};


uint64_t MIN( uint64_t A, uint64_t B ){
    if( A <= B )
        return A;
    return B;
}

struct area_desc* find_area( uint64_t offset )
{
	uint64_t key = offset >> key_shift;
	std::unordered_map<uint64_t, struct area_desc*>::iterator hash_it;
	
	hash_it = hash_table.find( key );
	if( hash_it == hash_table.end() )
		return NULL;
	return hash_it->second;
}

struct area_desc* make_area( uint64_t offset )
{
	int gpu_id;
	size_t free_mem = 0, tot_mem = 0;
	uint64_t *vram_ptr = NULL;
	uint64_t key = offset >> key_shift;
	struct area_desc *new_area;
	
	for( gpu_id = 0; gpu_id < num_gpus; gpu_id++ )
	{
		err_no = cudaSetDevice( gpu_id );
        assert( err_no == cudaSuccess );
		
		cudaMemGetInfo( &free_mem, &tot_mem );
		if( free_mem > SOFT_LIM )
		{
			err_no = cudaMalloc( &vram_ptr, AREA_SZ );
			assert( err_no == cudaSuccess );
			assert( vram_ptr != NULL );
			
			new_area = (struct area_desc*)malloc(sizeof(struct area_desc));
			new_area->GPU_ID		= gpu_id;
			new_area->start_offset	= (offset/AREA_SZ) * AREA_SZ;
			new_area->last_offset	= new_area->start_offset + AREA_SZ;
			new_area->vram_ptr		= vram_ptr;
			new_area->PBV			= NULL;
			
			hash_table[key] = new_area;
			return hash_table[key];
		}
	}
	return NULL;
}

static void *
gpuGAME_open (int readonly){
    /* create a handle ... */
    
    err_no = cudaGetDeviceCount( &num_gpus );
    printf("\tGAME >> %s: Blocks per Area %lu, Size %lu\n", __FUNCTION__, BLOCK_PER_AREA, GAME_SIZE );
    printf("\tGAME >> %s: CUDA devices: %d\n", __FUNCTION__, num_gpus );
	if( err_no != cudaSuccess )
		printf("\tGAME >> %s: CUDA err_no %d\n", __FUNCTION__, err_no );
	
	read_stream  = (cudaStream_t *)malloc(sizeof(cudaStream_t)*num_gpus);
	write_stream = (cudaStream_t *)malloc(sizeof(cudaStream_t)*num_gpus);
    
    for( int gpu_id = 0; gpu_id < num_gpus; gpu_id++ ) 
    {
        size_t free_mem = 0, tot_mem = 0;
        cudaDeviceProp dev_prop;
        
        err_no = cudaSetDevice( gpu_id );
        assert( err_no == cudaSuccess );
		err_no = cudaStreamCreate( &read_stream[gpu_id]  );
		assert( err_no == cudaSuccess );
		err_no = cudaStreamCreate( &write_stream[gpu_id] );
		assert( err_no == cudaSuccess );


		cudaMemGetInfo( &free_mem, &tot_mem );
		cudaGetDeviceProperties( &dev_prop, gpu_id );
		printf("\tGAME >> %s: (%d) %s\n", __FUNCTION__, gpu_id, dev_prop.name);
		printf("\tGAME >> %s: \t\tFree %.2lf Total %.2lf (GB)\n", __FUNCTION__, (double)free_mem/(1024*1024*1024), (double)tot_mem/(1024*1024*1024) );
    }
    
    return NBDKIT_HANDLE_NOT_NEEDED;
}

static int64_t 
gpuGAME_get_size (void *handle){
    return GAME_SIZE;
}

static int 
gpuGAME_config(const char *key, const char *value)
{
    if( strcmp(key, "size")==0 )
    {
        GAME_SIZE = nbdkit_parse_size( value );
    }
    else if( strcmp(key, "blk_per_area")==0 )
    {
        nbdkit_parse_uint64_t( "blk_per_area", value, &BLOCK_PER_AREA );
		if( BLOCK_PER_AREA & (BLOCK_PER_AREA-1) != 0 ){
			printf("\tGAME >> %s: BLOCK_PER_AREA must be power of two!\n", __FUNCTION__ );
			return -1;
		}
		key_shift = (int)log2(BLOCK_PER_AREA) + 12;
    }
    else{
        printf("\tGAME >> %s: Unknow argument\n", __FUNCTION__ );
        return -1;
    }
    return 0;
}

static int 
gpuGAME_pread (void *handle, void *buf, uint32_t count, uint64_t offset, uint32_t flags)
{
    struct area_desc *curr;
    uint64_t *vram_addr;
    uint64_t curr_offset = offset;
    uint64_t end_offset;
    uint32_t left_count = count;
    uint32_t curr_count;
    
    while( curr_offset < offset + count )
    {
        curr = find_area( curr_offset );
        if( curr == NULL )
            curr = make_area( curr_offset );
		assert( curr != NULL );
            
        err_no = cudaSetDevice( curr->GPU_ID );
        assert( err_no == cudaSuccess );
        
        end_offset = MIN( curr_offset + left_count, curr->last_offset );
        curr_count = end_offset - curr_offset;
        vram_addr  = (uint64_t*)((uint64_t)curr->vram_ptr + (uint64_t)(curr_offset % AREA_SZ));
        
        //printf("\tGAME >> %s: ADDR %p count %u\n", __FUNCTION__, vram_addr, curr_count );
        err_no = cudaMemcpyAsync( buf, vram_addr, curr_count, cudaMemcpyDeviceToHost, read_stream[curr->GPU_ID] );
        assert( err_no == cudaSuccess );
        
        curr_offset = end_offset;
        left_count -= curr_count;
    }
    assert( left_count == 0 );
    return 0;
}

static int 
gpuGAME_pwrite (void *handle, const void *buf, uint32_t count, uint64_t offset, uint32_t flags)
{
    struct area_desc *curr;
    uint64_t *vram_addr;
    uint64_t curr_offset = offset;
    uint64_t end_offset;
    uint32_t left_count = count;
    uint32_t curr_count;
    
    while( curr_offset < offset + count )
    {
        curr = find_area( curr_offset );
        if( curr == NULL )
            curr = make_area( curr_offset );
		assert( curr != NULL );
            
        err_no = cudaSetDevice( curr->GPU_ID );
        assert( err_no == cudaSuccess );
        
        end_offset = MIN( curr_offset + left_count, curr->last_offset );
        curr_count = end_offset - curr_offset;
        vram_addr  = (uint64_t*)((uint64_t)curr->vram_ptr + (uint64_t)(curr_offset % AREA_SZ));
        
        //printf("\tGAME >> %s: ADDR %p count %u\n", __FUNCTION__, vram_addr, curr_count );
        err_no = cudaMemcpyAsync( vram_addr, buf, curr_count, cudaMemcpyHostToDevice, write_stream[curr->GPU_ID] );
        assert( err_no == cudaSuccess );
        
        curr_offset = end_offset;
        left_count -= curr_count;
    }
    assert( left_count == 0 );
    return 0;
}

static struct nbdkit_plugin plugin = {
  .name           = "gpuGAME",
  .config         = gpuGAME_config,
  .open           = gpuGAME_open,
  .get_size       = gpuGAME_get_size,
  .pread          = gpuGAME_pread,
  .pwrite         = gpuGAME_pwrite,
  /* etc */
};
NBDKIT_REGISTER_PLUGIN(plugin)
