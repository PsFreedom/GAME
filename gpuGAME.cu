#define NBDKIT_API_VERSION 2
#include <nbdkit-plugin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define THREAD_MODEL NBDKIT_THREAD_MODEL_SERIALIZE_CONNECTIONS

uint64_t GAME_SIZE;
char    *GAME_PTR;
cudaStream_t read_stream;
cudaStream_t write_stream;

int err_no;
int num_gpus = -1;

static void *
gpuGAME_open (int readonly){
    /* create a handle ... */
    
    err_no = cudaGetDeviceCount( &num_gpus );
    assert( err_no == cudaSuccess);
    printf("\n\tGAME >> %s: Available CUDA capable devices: %d\n", __FUNCTION__, num_gpus );
    
    for( int gpu_id = 0; gpu_id < num_gpus; gpu_id++ ) 
    {
        size_t free_mem, tot_mem;
        
        err_no = cudaSetDevice( gpu_id );
        assert( err_no == cudaSuccess);
        
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
        err_no    = cudaMalloc( &GAME_PTR, GAME_SIZE );
        assert( err_no == cudaSuccess );
        
        cudaStreamCreate( &read_stream );
        cudaStreamCreate( &write_stream );
    }
    else{
        printf("%s: Unknow argument\n", __FUNCTION__ );
        return -1;
    }
    return 0;
}

static int 
gpuGAME_pread (void *handle, void *buf, uint32_t count, uint64_t offset, uint32_t flags)
{
    err_no = cudaMemcpyAsync( buf, (GAME_PTR + offset), count, cudaMemcpyDeviceToHost, read_stream );
//  printf("%s: err_no %d\n", __FUNCTION__, err_no );
    assert( err_no == cudaSuccess);
    return 0;
}

static int 
gpuGAME_pwrite (void *handle, const void *buf, uint32_t count, uint64_t offset, uint32_t flags)
{
    err_no = cudaMemcpyAsync( (GAME_PTR + offset), buf, count, cudaMemcpyHostToDevice, write_stream );
//  printf("%s: err_no %d\n", __FUNCTION__, err_no );
    assert( err_no == cudaSuccess);
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
