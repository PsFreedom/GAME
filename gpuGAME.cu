#define NBDKIT_API_VERSION 2
#include <nbdkit-plugin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define THREAD_MODEL NBDKIT_THREAD_MODEL_SERIALIZE_CONNECTIONS
#define BLOCK_SZ 4096UL               // 4KB
#define SOFT_LIM (1024UL*1024*1024*3) // 4GB
#define HARD_LIM (1024UL*1024*1024)   // 1GB
#define AREA_SZ  (BLOCK_PER_AREA*BLOCK_SZ)

int err_no;
uint64_t GAME_SIZE;
uint64_t BLOCK_PER_AREA;
cudaStream_t read_stream;
cudaStream_t write_stream;

struct area_desc{
    uint8_t  GPU_ID;
    uint64_t start_offset;
    uint64_t last_offset;
    uint64_t *vram_ptr;
    bool *PBV;
    struct area_desc *next;
};

struct area_header{
    struct area_header *next;
    struct area_desc   *first;
    uint8_t  GPU_ID;
    uint64_t count;
};

int num_gpus = -1;
struct area_header *root;

uint64_t MIN( uint64_t A, uint64_t B ){
    if( A <= B )
        return A;
    return B;
}

struct area_desc* find_area( uint64_t offset )
{
    struct area_header *head = root;
    struct area_desc   *curr;
    
    while( head != NULL ){
        curr = head->first;
        while( curr != NULL ){
            if( offset >= curr->start_offset && offset < curr->last_offset )
                return curr;
            curr = curr->next;
        }
        head = head->next;
    }
    return NULL;
}

struct area_desc* make_area( uint64_t offset )
{
    uint64_t *vram_ptr = NULL;
    struct area_header *head = root;
    struct area_desc   *curr;
    
    while( head != NULL ){
        size_t free_mem = 0, tot_mem = 0;
        
        err_no = cudaSetDevice( head->GPU_ID );
        assert( err_no == cudaSuccess);
        
        cudaMemGetInfo( &free_mem, &tot_mem );
        if( free_mem > SOFT_LIM )
        {
            printf("\tGAME >> %s: make an area for %p\n", __FUNCTION__, (void*)offset );
            printf("\tGAME >> %s: Device %d - Free %zd Total %zd\n", __FUNCTION__, head->GPU_ID, free_mem, tot_mem );
            err_no = cudaMalloc( &vram_ptr, AREA_SZ );
            assert( err_no == cudaSuccess );
            assert( vram_ptr != NULL );
            
            curr = (struct area_desc*)malloc(sizeof(struct area_desc));
            curr->GPU_ID       = head->GPU_ID;
            curr->start_offset = (offset/AREA_SZ)   * AREA_SZ;
            curr->last_offset  = curr->start_offset + AREA_SZ;
            curr->vram_ptr     = vram_ptr;
            curr->PBV          = NULL;
            curr->next         = head->first;
            head->first        = curr;
            head->count++;
            
            printf("\tGAME >> %s: GPU %d\n", __FUNCTION__, curr->GPU_ID );
            printf("\tGAME >> %s: offset %p - %p\n", __FUNCTION__, (void*)curr->start_offset, (void*)curr->last_offset );
            printf("\tGAME >> %s: ptr %p\n", __FUNCTION__, curr->vram_ptr );
            return curr;
        }
        head = head->next;
    }
    return NULL;
}

static void *
gpuGAME_open (int readonly){
    /* create a handle ... */
    struct area_header *curr;
    
    cudaGetDeviceCount( &num_gpus );
    printf("\tGAME >> %s: Blocks per Area %lu, Size %lu\n", __FUNCTION__, BLOCK_PER_AREA, GAME_SIZE );
    printf("\tGAME >> %s: CUDA devices: %d\n", __FUNCTION__, num_gpus );
    
    for( int gpu_id = 0; gpu_id < num_gpus; gpu_id++ ) 
    {
        size_t free_mem = 0, tot_mem = 0;
        
        err_no = cudaSetDevice( gpu_id );
        assert( err_no == cudaSuccess);
        
        curr = (struct area_header*)malloc(sizeof(struct area_header));
        curr->GPU_ID = gpu_id;
        curr->next   = root;
        root = curr;

        cudaMemGetInfo( &free_mem, &tot_mem );        
        printf("\tGAME >> %s: Device %d - Free %zd Total %zd\n", __FUNCTION__, gpu_id, free_mem, tot_mem );
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
        cudaStreamCreate( &read_stream );
        cudaStreamCreate( &write_stream );
    }
    else if( strcmp(key, "blk_per_area")==0 )
    {
        nbdkit_parse_uint64_t( "blk_per_area", value, &BLOCK_PER_AREA );
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
            
        err_no = cudaSetDevice( curr->GPU_ID );
        assert( err_no == cudaSuccess);
        
        end_offset = MIN( curr_offset + left_count, curr->last_offset );
        curr_count = end_offset - curr_offset;
        vram_addr  = (uint64_t*)((uint64_t)curr->vram_ptr + (uint64_t)(curr_offset % AREA_SZ));
        
        //printf("\tGAME >> %s: ADDR %p count %u\n", __FUNCTION__, vram_addr, curr_count );
        err_no = cudaMemcpyAsync( buf, vram_addr, curr_count, cudaMemcpyDeviceToHost, read_stream );
        assert( err_no == cudaSuccess);
        
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
            
        err_no = cudaSetDevice( curr->GPU_ID );
        assert( err_no == cudaSuccess);
        
        end_offset = MIN( curr_offset + left_count, curr->last_offset );
        curr_count = end_offset - curr_offset;
        vram_addr  = (uint64_t*)((uint64_t)curr->vram_ptr + (uint64_t)(curr_offset % AREA_SZ));
        
        //printf("\tGAME >> %s: ADDR %p count %u\n", __FUNCTION__, vram_addr, curr_count );
        err_no = cudaMemcpyAsync( vram_addr, buf, curr_count, cudaMemcpyHostToDevice, write_stream );
        assert( err_no == cudaSuccess);
        
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
