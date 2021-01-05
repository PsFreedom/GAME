gpuGAME: gpuGAME.cu
	nvcc -shared -Xcompiler -fPIC -o gpuGAME.so gpuGAME.cu
