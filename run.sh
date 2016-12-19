#!/usr/bin/env bash
curdir=`pwd`
for i in "$@"
do
   case $i in
        "--notebook" )
          set -euvx
          docker run -p 8888:8888 -v ${curdir}/:/app -it --rm humblehound/genrefinder:latest /bin/sh -c "jupyter notebook --port=8888 --no-browser --ip=0.0.0.0" ;;
        "--fetch" )
          set -euvx
          wget http://opihi.cs.uvic.ca/sound/genres.tar.gz -P neuralnet/data/
          cd neuralnet/data && tar -xzf genres.tar.gz;;
        "--extract" )
          set -euvx
          docker run -v ${curdir}/:/app -it --rm humblehound/genrefinder:latest /bin/sh -c "python neuralnet/extract.py" ;;
        "--train" )
          set -euvx
          docker run -v ${curdir}/:/app -it --rm humblehound/genrefinder:latest /bin/sh -c "python neuralnet/train.py" ;;
        "--predict" )
          set -euvx
          docker run -v ${curdir}/:/app -it --rm humblehound/genrefinder:latest /bin/sh -c "python neuralnet/predict.py" ;;
        "--shell" )
          set -euvx
          docker run -v ${curdir}/:/app -it --rm humblehound/genrefinder:latest /bin/sh ;;
        "--web" )
          set -euvx
          docker run -p 5000:5000 -v ${curdir}:/app -it --rm humblehound/genrefinder:latest /bin/sh -c "python webapp/run.py" ;;
        *)
          set -euvx
          docker run -p 5000:5000 -v ${curdir}:/app -it --rm humblehound/genrefinder:latest /bin/sh -c "python webapp/run.py" ;;
    esac
done