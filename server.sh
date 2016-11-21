#!/usr/bin/env bash
set -euvx
curdir=`pwd`
docker run -p 5000:5000 -v ${curdir}:/app -it --rm humblehound/genrefinder:latest /bin/sh -c "python webapp/run.py"
