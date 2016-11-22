#!/usr/bin/env bash
set -euvx
curdir=`pwd`
docker run -v ${curdir}:/app -it --rm humblehound/genrefinder:latest /bin/sh -c "python neuralnet/load_model.py"
