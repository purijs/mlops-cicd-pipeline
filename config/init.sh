#!/bin/bash
# init.sh

## Helper Commands
#kubectl delete all --all --all-namespaces && kind delete cluster --name kind-cluster
# sudo lsof -i:<port>
#helm ls -a --all-namespaces | awk 'NR > 1 { print  "-n "$2, $1}' | xargs -L1 helm delete
#docker system prune -a -f

set -e

rm -rf outputs/*
CLUSTER_NAME="kind-cluster"

if kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
  echo "KIND cluster '${CLUSTER_NAME}' already exists. Skipping creation."
else
  echo "Creating KIND cluster '${CLUSTER_NAME}'..."
  kind create cluster --name "${CLUSTER_NAME}" --config config/kind-config.yaml
fi

# Set kubectl context
kubectl cluster-info --context kind-${CLUSTER_NAME}

# Namespaces
kubectl apply -f manifests/namespaces.yaml

## -------------------- : -------------------- ##

## CONFIGURE: Kube Dash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.7.0/aio/deploy/recommended.yaml -n kubernetes-dashboard
kubectl wait --namespace kubernetes-dashboard --for=condition=ready pod --selector=k8s-app=kubernetes-dashboard --timeout=120s
kubectl apply -f config/admin-user.yaml -n kubernetes-dashboard
echo "KUBE DASHBOARD TOKEN: " $(kubectl -n kubernetes-dashboard create token admin-user) >> outputs/credentials.txt
echo "KUBE DASHBOARD: http://localhost:8001" >> outputs/urls.txt
kubectl proxy &
## CONFIGURE: Kube Dash

## -------------------- ##

## HELM Repos Setup ##
helm repo add argo https://argoproj.github.io/argo-helm
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo add minio https://charts.min.io/
helm repo add community-charts https://community-charts.github.io/helm-charts
helm repo add bitnami https://charts.bitnami.com/bitnami

helm repo update
## HELM Repos Setup ##

## -------------------- ##

## CONFIGURE: Docker Containers (nginx & fastapi)
home_dir=${PWD}
cd ${home_dir}/docker/fastapi/
docker build -t fastapi:latest .
cd ${home_dir}/docker/frontend/
docker build -t frontend:latest .

cd ${home_dir}

kind load docker-image fastapi:latest --name ${CLUSTER_NAME}
kind load docker-image frontend:latest --name ${CLUSTER_NAME}

kubectl create secret generic minio-credentials \
  --from-literal=accessKey=virtualminds \
  --from-literal=secretKey=virtualminds \
  -n fastapi

kubectl create secret generic minio-credentials \
  --from-literal=accessKey=virtualminds \
  --from-literal=secretKey=virtualminds \
  -n mlflow

## CONFIGURE: Docker Containers

## -------------------- ##

## CONFIGURE: ArgoCD
helm install argocd argo/argo-cd --namespace argocd -f helm-values/argocd/argocd-helm-values.yaml
sleep 60
echo "ArgoCD DASHBOARD: http://localhost:8080" >> outputs/urls.txt
echo "Argo Pass: " $(kubectl get secret argocd-initial-admin-secret -n argocd -o jsonpath="{.data.password}" | base64 --decode) >> outputs/credentials.txt
echo "Argo User: admin" >> outputs/credentials.txt
kubectl port-forward svc/argocd-server -n argocd 8080:443 &
## CONFIGURE: ArgoCD

## -------------------- ##

## CONFIGURE: MLFlow
helm install mlflow community-charts/mlflow --version 0.7.19 --namespace mlflow -f helm-values/mlflow/mlflow.yaml
sleep 30

kubectl set env deployment/mlflow -n mlflow \         
  MLFLOW_S3_ENDPOINT_URL=http://minio.minio.svc.cluster.local:9000 \
  AWS_ACCESS_KEY_ID=virtualminds \
  AWS_SECRET_ACCESS_KEY=virtualminds \
  MLFLOW_S3_BUCKET=mlops \
  MLFLOW_S3_IGNORE_TLS=true \
  MLFLOW_ARTIFACTS_DESTINATION=s3://mlops/models

sleep 30
kubectl rollout status deployment/mlflow -n mlflow
kubectl port-forward svc/mlflow -n mlflow 5555:5000 &
echo "MLFlow DASHBOARD: http://localhost:5555" >> outputs/urls.txt
## CONFIGURE: MLFlow

## -------------------- ##

## CONFIGURE: nginx
kubectl apply -f helm-values/nginx/frontend-deployment.yaml
sleep 30
kubectl port-forward svc/frontend-service -n frontend 5001:5001 &
echo "Frontend App: http://localhost:5001" >> outputs/urls.txt
## CONFIGURE: nginx

## -------------------- ##

## CONFIGURE: Ray
helm install kuberay-operator kuberay/kuberay-operator --version 1.1.1 -n kuberay
sleep 30
helm install raycluster kuberay/ray-cluster --version 1.1.1 -n kuberay -f helm-values/ray/kuberay-helm-values.yaml
sleep 60
kubectl port-forward svc/raycluster-kuberay-head-svc -n kuberay 8265:8265 &
echo "Ray DASHBOARD: http://localhost:8265" >> outputs/urls.txt
## CONFIGURE: Ray

## -------------------- ##

## CONFIGURE: Redis
helm install redis bitnami/redis -f helm-values/redis/redis-values.yaml -n db
sleep 60
kubectl port-forward -n db svc/redis-master 6379:6379 &
echo "Redis Endpoint: http://localhost:6379" >> outputs/urls.txt
echo "Redis Password: redis" >> outputs/credentials.txt
## CONFIGURE: Redis

## -------------------- ##

## CONFIGURE: Minio
kubectl create secret generic minio-secret -n minio \
  --from-literal=rootUser=virtualminds \
  --from-literal=rootPassword=virtualminds

kubectl create secret generic minio-user-secret -n minio \
  --from-literal=accesskey0=virtualminds \
  --from-literal=secretkey0=virtualminds

helm install minio minio/minio -n minio -f helm-values/s3/minio-values.yaml
sleep 30
kubectl port-forward svc/minio -n minio 9000:9000 &
kubectl port-forward svc/minio-console -n minio 9001:9001 &

echo "Minio CLI: http://localhost:9000" >> outputs/urls.txt
echo "Minio UI: http://localhost:9001" >> outputs/urls.txt
echo "Minio UI: user/pass: virtualminds/virtualminds" >> outputs/credentials.txt
## CONFIGURE: Minio

## -------------------- : -------------------- ##
# Deploy FastAPI App Using CI/CD
kubectl apply -f helm-values/argocd/argocd-deploy-app.yaml -n argocd
sleep 15
kubectl port-forward svc/fastapi-service 8888:80 -n fastapi &
echo "FastAPI Swagger Endpoints: http://localhost:8888/docs" >> outputs/urls.txt

## Deploy ML Model Using Ray Serve (Run after training at least one model)
#kubectl exec -it fastapi-app-pod-name  -n fastapi -- python /var/task/fastapi/serve.py