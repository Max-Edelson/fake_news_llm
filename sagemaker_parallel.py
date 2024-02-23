from sagemaker.pytorch import PyTorch

sagemaker_session = sagemaker.Session()
role = "your-sagemaker-role"

train_data = "s3://your-bucket/train_data"

estimator = PyTorch(
    role=role,
    instance_count=2,
    instance_type="ml.c5.large",
    # Activate distributed training with SMDDP
    py_version="py3",
    sagemaker_session=sagemaker_session,
    distribution={ "pytorchddp": { "enabled": True } }  # mpirun, activates SMDDP AllReduce OR AllGather
    # distribution={ "torch_distributed": { "enabled": True } }  # torchrun, activates SMDDP AllGather
    # distribution={ "smdistributed": { "dataparallel": { "enabled": True } } }  # mpirun, activates SMDDP AllReduce OR AllGather
)

estimator.fit({"train": train_data})