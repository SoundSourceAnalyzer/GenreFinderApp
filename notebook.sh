#!/usr/bin/env bash
set -euvx

curdir=`pwd`
docker run -p 8888:8888 -v ${curdir}/neuralnet:/app -it --rm humblehound/genrefinder:latest /bin/sh -c "jupyter notebook --port=8888 --no-browser --ip=0.0.0.0"
