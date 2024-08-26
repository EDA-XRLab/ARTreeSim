# Tree Detection App 


## Data Link
weights: 
https://drive.google.com/drive/folders/117cigIbn4oGJx-ANIjr-Az8owto7Jaf6?usp=sharing

## Step 0
Install conda, docker, and docker compose

## Step 1
Download weights, copy contents to /src/perceptree/weights/

## Step 2

create environment `conda env create -f environment.yml`
activate `conda activate timbertrack`

## Step 3
start backend microservice on jetson platform

`docker compose up`

or on x86-64

`docker compose -f docker-compose.yaml -f docker-compose.x86_override.yaml up`

to view swagger schema, go to http://localhost:8000/docs

endpoints include functionality to query/download predictions

Note that `docker compose up` may take 30+ minutes to run

## Step 4

To interface with the app from a python program, you can import `RedisHandler`
after appending the absolute location of /src to your script.

See /src/io.py for details


## Endpoints

/detections/
    start/ # starts the detection service
    stop/ # stops the detection service
    status/ # shows the the number of processed detections by the app
    get/ # query/download processed detections
/images/
    add/ # add images via rest api (maybe ideal for local network interface)



# Trouble shooting:

### "Error starting userland proxy: listen tcp4 127.0.0.1:6379: bind: address already in use"
- stop redis daemon:
    `sudo /etc/init.d/redis-server stop`

- run docker compose:
    `docker compose up`

- restart redis daemon
    `sudo /etc/init.d/redis-server start`
