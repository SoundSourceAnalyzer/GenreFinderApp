# GenreFinderApp

## Overview
A deep neural network project used for sound genre classification. The project consist of several packages: 
### audio_parser
 package with code used to extract data from sound files. Sources for this operations should be placed in data folder. Results should be stored in neural_net/data package.
### dockerfile
 contains the Dockerfile used to build the runtime for this app
### neuralnet 
packge with neural network code
### webapp 
the flask web server used for serving the neural net functionality
    
## Usage
Root of the project contains shell scripts for running particular operations in the solution. 
Current available options:
- start the web server: ```server.sh```
- train the model with GZTAN data: ```train_model.sh```
- start Jupyter Notebook - ```notebook.sh```