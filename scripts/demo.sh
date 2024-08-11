CUDA_VISIBLE_DEVICES=1 python demo.py --config configs/stable-diffusion/v1-inference_demo.yaml \
                                              --ckpt ckpt/eoc-sd-v1-4-coco.ckpt \
                                              --json examples/cat.json \
                                              --outdir outputs/demo \
                                              --seed 24 \
                                              --plms 
