#!bin/bash

# image2image-domain

python main_for_img2img.py --scope ct_img2img --mode train --nstage 0 --num_epoch 100 --batch_size 4 --loss_type_img img --lr_type_img plain

#python main_for_img2img.py --scope ct_img2img --mode train --nstage 0 --num_epoch 100 --batch_size 4 --loss_type_img img --lr_type_img residual
#python main_for_img2img.py --scope ct_img2img --mode train --nstage 1 --num_epoch 50  --batch_size 4 --loss_type_img img --lr_type_img residual
#python main_for_img2img.py --scope ct_img2img --mode train --nstage 2 --num_epoch 50  --batch_size 4 --loss_type_img img --lr_type_img residual
#python main_for_img2img.py --scope ct_img2img --mode train --nstage 3 --num_epoch 50  --batch_size 4 --loss_type_img img --lr_type_img residual
#python main_for_img2img.py --scope ct_img2img --mode train --nstage 4 --num_epoch 50  --batch_size 4 --loss_type_img img --lr_type_img residual

#python main_for_img2img.py --scope ct_img2img --mode test --nstage 0 --num_epoch 100 --batch_size 1 --loss_type_img img --lr_type_img residual --downsample 2
#python main_for_img2img.py --scope ct_img2img --mode test --nstage 1 --num_epoch 50  --batch_size 1 --loss_type_img img --lr_type_img residual --downsample 2
#python main_for_img2img.py --scope ct_img2img --mode test --nstage 2 --num_epoch 50  --batch_size 1 --loss_type_img img --lr_type_img residual --downsample 2
#python main_for_img2img.py --scope ct_img2img --mode test --nstage 3 --num_epoch 50  --batch_size 1 --loss_type_img img --lr_type_img residual --downsample 2
#python main_for_img2img.py --scope ct_img2img --mode test --nstage 4 --num_epoch 50  --batch_size 1 --loss_type_img img --lr_type_img residual --downsample 2
#
#python main_for_img2img.py --scope ct_img2img --mode test --nstage 0 --num_epoch 100 --batch_size 1 --loss_type_img img --lr_type_img residual --downsample 3
#python main_for_img2img.py --scope ct_img2img --mode test --nstage 1 --num_epoch 50  --batch_size 1 --loss_type_img img --lr_type_img residual --downsample 3
#python main_for_img2img.py --scope ct_img2img --mode test --nstage 2 --num_epoch 50  --batch_size 1 --loss_type_img img --lr_type_img residual --downsample 3
#python main_for_img2img.py --scope ct_img2img --mode test --nstage 3 --num_epoch 50  --batch_size 1 --loss_type_img img --lr_type_img residual --downsample 3
#python main_for_img2img.py --scope ct_img2img --mode test --nstage 4 --num_epoch 50  --batch_size 1 --loss_type_img img --lr_type_img residual --downsample 3
#
#python main_for_img2img.py --scope ct_img2img --mode test --nstage 0 --num_epoch 100 --batch_size 1 --loss_type_img img --lr_type_img residual --downsample 4
#python main_for_img2img.py --scope ct_img2img --mode test --nstage 1 --num_epoch 50  --batch_size 1 --loss_type_img img --lr_type_img residual --downsample 4
#python main_for_img2img.py --scope ct_img2img --mode test --nstage 2 --num_epoch 50  --batch_size 1 --loss_type_img img --lr_type_img residual --downsample 4
#python main_for_img2img.py --scope ct_img2img --mode test --nstage 3 --num_epoch 50  --batch_size 1 --loss_type_img img --lr_type_img residual --downsample 4
#python main_for_img2img.py --scope ct_img2img --mode test --nstage 4 --num_epoch 50  --batch_size 1 --loss_type_img img --lr_type_img residual --downsample 4
#
#python main_for_img2img.py --scope ct_img2img --mode test --nstage 0 --num_epoch 100 --batch_size 1 --loss_type_img img --lr_type_img residual --downsample 6
#python main_for_img2img.py --scope ct_img2img --mode test --nstage 1 --num_epoch 50  --batch_size 1 --loss_type_img img --lr_type_img residual --downsample 6
#python main_for_img2img.py --scope ct_img2img --mode test --nstage 2 --num_epoch 50  --batch_size 1 --loss_type_img img --lr_type_img residual --downsample 6
#python main_for_img2img.py --scope ct_img2img --mode test --nstage 3 --num_epoch 50  --batch_size 1 --loss_type_img img --lr_type_img residual --downsample 6
#python main_for_img2img.py --scope ct_img2img --mode test --nstage 4 --num_epoch 50  --batch_size 1 --loss_type_img img --lr_type_img residual --downsample 6
#
#python main_for_img2img.py --scope ct_img2img --mode test --nstage 0 --num_epoch 100 --batch_size 1 --loss_type_img img --lr_type_img residual --downsample 8
#python main_for_img2img.py --scope ct_img2img --mode test --nstage 1 --num_epoch 50  --batch_size 1 --loss_type_img img --lr_type_img residual --downsample 8
#python main_for_img2img.py --scope ct_img2img --mode test --nstage 2 --num_epoch 50  --batch_size 1 --loss_type_img img --lr_type_img residual --downsample 8
#python main_for_img2img.py --scope ct_img2img --mode test --nstage 3 --num_epoch 50  --batch_size 1 --loss_type_img img --lr_type_img residual --downsample 8
#python main_for_img2img.py --scope ct_img2img --mode test --nstage 4 --num_epoch 50  --batch_size 1 --loss_type_img img --lr_type_img residual --downsample 8
#
#python main_for_img2img.py --scope ct_img2img --mode test --nstage 0 --num_epoch 100 --batch_size 1 --loss_type_img img --lr_type_img residual --downsample 12
#python main_for_img2img.py --scope ct_img2img --mode test --nstage 1 --num_epoch 50  --batch_size 1 --loss_type_img img --lr_type_img residual --downsample 12
#python main_for_img2img.py --scope ct_img2img --mode test --nstage 2 --num_epoch 50  --batch_size 1 --loss_type_img img --lr_type_img residual --downsample 12
#python main_for_img2img.py --scope ct_img2img --mode test --nstage 3 --num_epoch 50  --batch_size 1 --loss_type_img img --lr_type_img residual --downsample 12
#python main_for_img2img.py --scope ct_img2img --mode test --nstage 4 --num_epoch 50  --batch_size 1 --loss_type_img img --lr_type_img residual --downsample 12
