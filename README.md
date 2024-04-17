# MOLvin
JT-VAE, GAN and reinforcement learning integration for drug discovery

*PIP: Create script/dockerfile of following commands for easy assembly*
- WSL Commands Docker
	- Build image with Dockerfile:
		- ```docker build -t <new_image> .```
			- ensure in correct working dir.
	- Create a container 
		- ```docker run -d --gpus all -it --name <your_container_name> -v <path_to_host_dir>:/<container_dest_dir> <your_image>```
        - detached terminal 
    - Attach terminal to container
        - ```docker exec -it <container_name/ID> /bin/bash```
        

- Dependencies
	- ```pip install deepchem rdkit numpy tensorflow matplotlib seaborn pandas jax jaxlib torch-geometric torch torchvision torchaudio```
		- Reference PyTorch for machine dependent install: https://pytorch.org/get-started/locally/
---


*Updated explore.py should address issues in the current error log*