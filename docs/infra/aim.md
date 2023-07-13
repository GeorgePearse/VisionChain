
To setup, ssh into instance, 

Run:

```
mkdir environments
cd environments
python3 -m pip install virtualenv
python3 -m virtualenv aim 
source environments/aim/bin/activate
pip install aim
sudo fuser -k 43800/tcp
sudo fuser -k 53800/tcp
aim server --repo . & aim up --repo . -h 0.0.0.0
```

Can upload the logs to S3 as backup (should schedule this)

s3://binit-aim-experiments

Available at <go.binit.ai/aim>

NB: have had some problems getting it to work reliably with multi-gpu training
which has become a necessity. 
