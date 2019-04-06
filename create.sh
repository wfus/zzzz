#!/bin/bash

KEYPAIR=$HOME/.ssh/cs205.pem
MASTER_INSTANCE=m3.xlarge
WORKER_INSTANCE=m3.xlarge

aws emr create-cluster \
--applications Name=MXNet Name=Spark \
--release-label emr-5.10.0 \
--service-role EMR_DefaultRole \
--ec2-attributes InstanceProfile=EMR_EC2_DefaultRole,KeyName=$KEYPAIR \
--instance-groups InstanceGroupType=MASTER,InstanceCount=1,InstanceType=$MASTER_INSTANCE \
InstanceGroupType=CORE,InstanceCount=4,InstanceType=$WORKER_INSTANCE \
--bootstrap-actions Name='install-pillow-boto3',Path=s3://aws-dl-emr-bootstrap/mxnet-spark-demo-bootstrap.sh \
--region us-east-1 \
--name "NACHO"


# Things that you could add but I don't want to
# --log-uri 's3n://<YOUR-S3-BUCKET-FOR-EMR-LOGS>/' \
