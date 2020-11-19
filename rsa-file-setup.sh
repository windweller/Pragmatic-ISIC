#!/usr/bin/env bash

mkdir checkpoints

wget https://cub-qud-rsa.s3-us-west-2.amazonaws.com/gve-cub-D2020-03-14-T18-04-04-G0-best-ckpt.pth
mv gve-cub-D2020-03-14-T18-04-04-G0-best-ckpt.pth ./checkpoints/

wget https://cub-qud-rsa.s3-us-west-2.amazonaws.com/cub_data.zip
unzip cub_data.zip