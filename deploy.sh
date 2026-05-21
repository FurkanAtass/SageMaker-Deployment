model_name=cloth_detection
account=$(aws sts get-caller-identity --query Account --output text)
region=$AWS_DEFAULT_REGION

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${model_name}:latest"
aws ecr describe-repositories --repository-names "${model_name}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${model_name}" > /dev/null
fi

aws ecr get-login-password --region ${region}|docker login --username AWS --password-stdin ${fullname}

docker buildx build \
    --platform linux/amd64 \
    --provenance=false \
    --output type=docker \
    -t ${model_name} \
    -f src/services/cloth_detection/Dockerfile .

docker tag ${model_name} ${fullname}
docker push ${fullname}