#define NBDKIT_API_VERSION 2
#include <nbdkit-plugin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "async_ctrl.h"
#define THREAD_MODEL NBDKIT_THREAD_MODEL_SERIALIZE_REQUESTS

uint64_t gpuGAME_SIZE;
char    *gpuGAME_PTR;

static void 
gpuGAME_load (void){
    printf("\tgpuGAME >> gpuGAME is loaded\n");
}

static void *
gpuGAME_open (int readonly){
    /* create a handle ... */
    printf("\tgpuGAME >> gpuGAME is opened\n");
    return NBDKIT_HANDLE_NOT_NEEDED;
}

static int64_t 
gpuGAME_get_size (void *handle){
    return gpuGAME_SIZE;
}

static int 
gpuGAME_config(const char *key, const char *value)
{
    if( strcmp(key, "size")==0 )
    {
        gpuGAME_SIZE = nbdkit_parse_size(value);
        printf("\tgpuGAME >> SIZE %s = %ld\n", value, gpuGAME_SIZE);
		
		assert( cudaMalloc(&gpuGAME_PTR, gpuGAME_SIZE) == cudaSuccess );
        printf("\tgpuGAME >> GPU memory allocated - %p\n", gpuGAME_PTR);
    }
    else if( strcmp(key, "buff")==0 )
    {
        async_list_init( atoi(value) );
    }
    else{
        printf("\tgpuGAME >> do not recognize this config\n");
		return -1;
    }
	return 0;
}

static int 
gpuGAME_pread (void *handle, void *buf, uint32_t count, uint64_t offset, uint32_t flags)
{
    int i = async_list_add(gpuGAME_PTR + offset, count);
	assert(cudaMemcpyAsync(buf, (gpuGAME_PTR + offset), count, cudaMemcpyDeviceToHost, async_list[i].cpySteam) == cudaSuccess);
	return 0;
}

static int 
gpuGAME_pwrite (void *handle, const void *buf, uint32_t count, uint64_t offset, uint32_t flags)
{
    int i = async_list_add(gpuGAME_PTR + offset, count);
	assert(cudaMemcpyAsync((gpuGAME_PTR + offset), buf, count, cudaMemcpyHostToDevice, async_list[i].cpySteam) == cudaSuccess);
	return 0;
}

static struct nbdkit_plugin plugin = {
  .name     = "gpuGAME",
  .load     = gpuGAME_load,
  .config   = gpuGAME_config,
  .open     = gpuGAME_open,
  .get_size = gpuGAME_get_size,
  .pread    = gpuGAME_pread,
  .pwrite   = gpuGAME_pwrite,
  /* etc */
};
NBDKIT_REGISTER_PLUGIN(plugin)
