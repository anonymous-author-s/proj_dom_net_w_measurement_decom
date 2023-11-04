# Enhanced Sparse-View CT Reconstruction with Projection-Domain CNN: Leveraging Hierarchical Measurement Decomposition and Fourier Constraints

## Requirements
___PLEASE INSTALL !!! DOCKER !!! TO CREATE CONTAINER.___

https://docs.docker.com/engine/install/ubuntu/

## Getting Started

1. Clone this repository
   ```
   git clone git@github.com:anonymous-author-s/projection-domain-network-with-measurement-decomposition.git
   ```

2. Create, Start, and Run Docker Container
   ```
   docker run -it -v [HOST_DIR]:/workspace/[CONTAINER_DIR] --gpus all --name [DOCKER_NAME] pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel /bin/bash
   ```

   ex) 
   ```
   docker run -it -v ./PDNET-W-MD:/workspace/PDNET-W-MD --gpus all --name mpi pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel /bin/bash
   ```

3. Install basic libraries
   ```
   apt-get update
   ```

   ```
   apt-get install git vim wget unzip -y
   ```

4. Move to the repository
   ```
   cd PDNET-W-MD
   ```

5. Install the requirements
   ```
   pip install -r requirements.txt
   ```

6. Download pre-trained models
   ```
   wget https://www.dropbox.com/s/88vx8wfulls6gli/checkpoints.zip
   ```
   
   ```
   unzip checkpoints.zip
   ```

   ```
   rm checkpoints.zip
   ```

7. Reproduce Figure 1-2 (a-b) in Supplementary Material
   ```
    python demo_sup_fig1_a.py
   ```

    ```
    python demo_sup_fig1_b.py
   ```

   ```
    python demo_sup_fig2_a.py
   ```

   ```
    python demo_sup_fig2_b.py
   ```

## Prepare the datasets
Before training the image- and projection-domain network, 

___YOU MUST PREPARE THE DATASET ACCORING TO DECOMPOSITION LEVEL.___

___PLEASE SEE THE ALGORITHM 1 TO PERFORM HIERARCHICAL DECOMPOSITION.___

## Train and Test the model
### Train mode
   1. image-domain CNN trained with 96 view and level 1\
      (*level 1 is equal to nstage=0)
      ```python
      python main_for_img.py \
            --mode train \
            --lr_type_img residual \
            --nstage 0 \
            --downsample 8 \
            --num_upscale 3 \
            --factor_upscale 2 2 2
      ```

   2. projection-domain CNN trained with 96 view and level 4\
      (*level 4 is equal to nstage=3)
      ```python
      python main_for_prj.py \
            --mode train \
            --lr_type_prj residual \
            --nstage 3 \
            --downsample 8 \
            --num_upscale 3 \
            --factor_upscale 2 2 2
      ```

### Test mode
   1. image-domain CNN trained with 96 view and level 1\
      (*level 1 is equal to nstage=0)
      ```python
      python main_for_img.py \
            --mode test \
            --lr_type_img residual \
            --nstage 0 \
            --downsample 8 \
            --num_upscale 3 \
            --factor_upscale 2 2 2
      ```

   2. projection-domain CNN trained with 96 view and level 4\
      (*level 4 is equal to nstage=3)
      ```python
      python main_for_prj.py \
            --mode test \
            --lr_type_prj residual \
            --nstage 3 \
            --downsample 8 \
            --num_upscale 3 \
            --factor_upscale 2 2 2
      ```