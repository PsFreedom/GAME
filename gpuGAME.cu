#define NBDKIT_API_VERSION 2
#include <nbdkit-plugin.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define THREAD_MODEL NBDKIT_THREAD_MODEL_SERIALIZE_CONNECTIONS

uint64_t gpuGAME_SIZE;
char    *gpuGAME_PTR;

static void *
gpuGAME_open (int readonly){
    /* create a handle ... */
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
		assert( cudaMalloc(&gpuGAME_PTR, gpuGAME_SIZE) == cudaSuccess );
    }
    else{
		return -1;
    }
	return 0;
}

static int 
gpuGAME_pread (void *handle, void *buf, uint32_t count, uint64_t offset, uint32_t flags)
{
    assert(cudaMemcpy(buf, (gpuGAME_PTR + offset), count, cudaMemcpyDeviceToHost) == cudaSuccess);
	return 0;
}

static int 
gpuGAME_pwrite (void *handle, const void *buf, uint32_t count, uint64_t offset, uint32_t flags)
{
    assert(cudaMemcpy((gpuGAME_PTR + offset), buf, count, cudaMemcpyHostToDevice) == cudaSuccess);
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
