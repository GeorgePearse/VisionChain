
Currently default to a p3.8xlarge instance for actual model training, (4 x V100). This allows for 4 images at (1920 x 1080).
A few posts have suggested that the g5 range is more cost effective than the p3 range.

<https://aws.amazon.com/ec2/instance-types/g5/>

* Excellent post on AWS GPU instances (though quite old) - <https://towardsdatascience.com/choosing-the-right-gpu-for-deep-learning-on-aws-d69c157d8c86>

"G5 instances offer up to 15% lower cost-to-train than Amazon EC2 P3 instances. They also deliver up to 3.3x higher performance for ML training compared to G4dn instances."


## How to launch an instance from the command line 

```
aws ec2 run-instances --image-id ami-09dace87985ec779d --instance-type p3.2xlarge --region us-east-1
```

* Very hacky approach to executing commands on a machine <https://blog.ruanbekker.com/blog/2020/11/02/running-ssh-commands-on-aws-ec2-instances-with-python/#:~:text=In%20this%20quick%20post%20I,commands%20on%20the%20target%20instance.>
