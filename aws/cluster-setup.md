# How to set up and use AWS cluster

## Configure the cluster
The cluster name is *masters-cluster*

    pcluster create-cluster --cluster-name <CLUSTER-NAME> \
        --cluster-configuration <CLUSTER-NAME>.yaml

## Logging in

    pcluster ssh --cluster-name <CLUSTER-NAME> -i <PATH/TO/KEY>.pem



Reference: https://docs.aws.amazon.com/parallelcluster/latest/ug/tutorials-running-your-first-job-on-version-3.html




