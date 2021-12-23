# GAME: GPU-Assisted Memory Expansion
This guide explains how to prepare and setup GAME on a Linux system. GAME works on any Linux distro with Nvidia CUDA and nbdkit support since GAME is a nbdkit plugin that employs Nvidia CUDA library for data trasferring between system memory (RAM) and GPU memory (VRAM). We use a freshly-installed Ubuntu 20.04 LTS for our baseline system.

## Update the system and install required packages
Dependencies:
 - [Nvidia CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (we tested on CUDA 10.1, 11.2, and 11.3 but older or newer versions should work fine as well)
 - [nbdkit](http://manpages.ubuntu.com/manpages/focal/man1/nbdkit.1.html) (we tested on nbdkit 1.16)

Command:
```
sudo apt -y install make gcc g++
sudo apt -y install nbd-server nbd-client nbdkit nbdkit-plugin-dev
sudo apt -y install nvidia-cuda-toolkit-gcc
sudo apt -y update
sudo apt -y upgrade
sudo apt -y autoremove 
sudo reboot
```
This command will install everything, cleanup, then reboot the system.
