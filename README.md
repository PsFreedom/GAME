# GAME: GPU-Assisted Memory Expansion
This guide explains how to prepare and setup GAME on a Linux system. GAME works on any Linux distro with Nvidia CUDA and nbdkit support since GAME is a nbdkit plugin (**N**etwork **B**lock **D**evice) that employs Nvidia CUDA library for data trasferring between system memory (RAM) and GPU memory (VRAM). We use a freshly-installed Ubuntu 20.04 LTS (Focal Fossa) for our baseline testbed.

## Updating the system and installing required packages
Dependencies:
 - [Nvidia CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 
   - We tested on CUDA 11.2, 11.3, and 11.5 (older or newer versions should work fine as well)
   - The default CUDA toolkit package come with Ubuntu repository may not work, please follow Nvidia installation guide
 - [nbdkit](http://manpages.ubuntu.com/manpages/focal/man1/nbdkit.1.html) 
   - We tested on nbdkit 1.16 (should work with other versions as well)

Command:
```
sudo apt -y install make gcc g++ git 
sudo apt -y install nbd-server nbd-client nbdkit nbdkit-plugin-dev 
sudo apt -y update 
sudo apt -y upgrade
```
Installing Nvidia CUDA following Nvidia guideline:
```
https://developer.nvidia.com/cuda-downloads
```
Updating PATH environment variable to include CUDA toolkit (~/.profile or ~/.bashrc):
```
export PATH=$PATH:/usr/local/cuda/bin
```
After installation, cleanup and reboot the system:
```
sudo apt -y autoremove
sudo reboot
```

## Getting and building GAME
Getting GAME from GitHub:
```
 git clone https://github.com/PsFreedom/GAME.git
```
Building GAME nbdkit plugin (inside GAME directory):
```
make
```
If sucessfully built, ```gpuGAME.so``` will appear in the directory (shared object for nbdkit).

## Creating nbd server using nbdkit and GAME plugin
 - Make sure that the network block device module (nbd) is added to the system properly
   - ```sudo modprobe nbd``` is to confirm and add nbd module support, which only needs to be executed once
 - **size**, the total memory to allocate from GPUs (cannot be larger than total memory of all GPUs in the system combined)
 - **blk_per_area**, the total memory block per area 
 - Please make sure that the path to **gpuGAME.so** is correct (or it is in your working directory)
 - Learn more about [nbdkit parameters](http://manpages.ubuntu.com/manpages/focal/man1/nbdkit.1.html)
```
sudo modprobe nbd
sudo nbdkit -fv gpuGAME.so size=4G blk_per_area=8192
```
The example command above, nbdkit will allocate total 4GB memory from GPUs.

## Mounting nbd device locally
Using ```nbd-client```, we connect to our nbdkit server (in the previous step) which resides in the same machine (localhost). GAME device is now up on ```/dev/nbd0```. Learn more about [nbd-client parameters](http://manpages.ubuntu.com/manpages/xenial/man8/nbd-client.8.html).
```
sudo nbd-client -b 4096 localhost /dev/nbd0
```

## GPU-Assisted Memory Expansion
GAME can support memory expansion in swap mode by formatting GAME device as a swap partition.
```
sudo mkswap /dev/nbd0
sudo swapon /dev/nbd0
```
