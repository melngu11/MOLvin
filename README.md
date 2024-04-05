# MOLvin
JT-VAE, GAN and reinforcement learning integration for drug discovery

*PIP: Create script/dockerfile of following commands for easy assembly*
- WSL Commands Docker
	- Build image with Dockerfile:
		- ```docker build -t <new_image> .```
			- ensure in correct working dir.
	- Create a container 
		- ```docker run -d --gpus all -it --name <your_container_name> -v <path_to_host_dir>:/<container_dest_dir> <your_image>```
