gpuGAME: gpuGAME.cu
	nvcc -shared -Xcompiler -fPIC -o gpuGAME.so gpuGAME.cu

clean:
	rm gpuGAME.so
