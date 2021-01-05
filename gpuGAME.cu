#define NBDKIT_API_VERSION 2
#include <nbdkit-plugin.h>
#include <stdio.h>
#include <string.h>

#define THREAD_MODEL NBDKIT_THREAD_MODEL_SERIALIZE_CONNECTIONS

int64_t gpuGAME_SIZE;
int64_t *gpuGAME_PTR;

static void 
gpuGAME_load (void)
{
    printf("gpuGAME: gpuGAME is loaded\n");
}

static void *
gpuGAME_open (int readonly)
{
    /* create a handle ... */
    printf("gpuGAME: gpuGAME is opened\n");
    return NBDKIT_HANDLE_NOT_NEEDED;
}

static int 
gpuGAME_config(const char *key, const char *value)
{
    if( strcmp(key, "size")==0 ){
        gpuGAME_SIZE = nbdkit_parse_size(value);
        printf("gpuGAME: gpuGAME_SIZE (%s) = %ld\n", value, gpuGAME_SIZE);
		
		if( cudaMalloc(&gpuGAME_PTR, gpuGAME_SIZE) == cudaSuccess ){
			printf("gpuGAME: GPU memory allocated!\n");
		}
		else{
			printf("gpuGAME: GPU memory allocation failed!\n");
			return -1;
		}
    }
    else{
        printf("gpuGAME: do not recognize this config\n");
		return -1;
    }
	return 0;
}

static int64_t 
gpuGAME_get_size (void *handle)
{
    return gpuGAME_SIZE;
}

static int 
gpuGAME_pread (void *handle, void *buf, uint32_t count, uint64_t offset, uint32_t flags)
{
    cudaMemcpy( buf, gpuGAME_PTR + offset, count, cudaMemcpyDeviceToHost );
    return 0;
}

static int 
gpuGAME_pwrite (void *handle, const void *buf, uint32_t count, uint64_t offset, uint32_t flags)
{
    cudaMemcpy( gpuGAME_PTR + offset, buf, count, cudaMemcpyHostToDevice );
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
