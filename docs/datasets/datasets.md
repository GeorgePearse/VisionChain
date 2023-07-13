## Dataset Management

s5cmd is the fastest way to retrieve a dataset for training from s3, example command: 

(will pull the docker container for you)
```
docker run --rm -v $(pwd):/aws -v ~/.aws:/root/.aws peakcom/s5cmd sync s3://binit-split-datasets/municipals/data/* .
```
