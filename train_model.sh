#!/usr/bin/env bash
set -eu
curdir=`pwd`
docker run -v ${curdir}:/app -it --rm humblehound/genrefinder:latest /bin/sh -c "python neuralnet/model.py"
