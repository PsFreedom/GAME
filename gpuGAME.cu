#define NBDKIT_API_VERSION 2
#include <nbdkit-plugin.h>
#include <stdio.h>
#include <string.h>

#define THREAD_MODEL NBDKIT_THREAD_MODEL_SERIALIZE_CONNECTIONS

uint64_t gpuGAME_SIZE;
uint64_t *gpuGAME_PTR;

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
	cudaError_t cuda_errno;
    if( strcmp(key, "size")==0 ){
        gpuGAME_SIZE = nbdkit_parse_size(value);
        printf("\tgpuGAME >> SIZE %s = %ld\n", value, gpuGAME_SIZE);
		printf("\tgpuGAME >> GPU memory pre-alloc - %p\n", gpuGAME_PTR);
		
		cuda_errno = cudaMalloc(&gpuGAME_PTR, gpuGAME_SIZE);
		if( cuda_errno == cudaSuccess ){
			printf("\tgpuGAME >> GPU memory allocated - %p\n", gpuGAME_PTR);
		}
		else{
			printf("\tgpuGAME >> GPU memory allocation failed! (cuda_errno %u)\n", cuda_errno);
			return -1;
		}
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
	cudaError_t cuda_errno;
	cuda_errno = cudaMemcpy(buf, (gpuGAME_PTR + offset/sizeof(uint64_t)), count, cudaMemcpyDeviceToHost);
	if(cuda_errno != cudaSuccess){
		printf("\tgpuGAME >> cudaMemcpy ERROR %d\n", cuda_errno);
		return -1;
	}
	return 0;
}

static int 
gpuGAME_pwrite (void *handle, const void *buf, uint32_t count, uint64_t offset, uint32_t flags)
{
	cudaError_t cuda_errno;
    cuda_errno = cudaMemcpy((gpuGAME_PTR + offset/sizeof(uint64_t)), buf, count, cudaMemcpyHostToDevice);
	if(cuda_errno != cudaSuccess){
		printf("\tgpuGAME >> cudaMemcpy ERROR %d\n", cuda_errno);
		return -1;
	}
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
