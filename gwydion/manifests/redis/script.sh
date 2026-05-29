#!bin/bash
set -euo pipefail

minikube start -p redis --cpus=4 --memory=8192 --driver=docker --addons=metrics-server

kubectl apply -f redis.yaml

helm install prom prometheus-community/kube-prometheus-stack --namespace redis -f values.yaml
helm install redis-leader-exporter prometheus-community/prometheus-redis-exporter -n redis --set redisAddress=redis://redis-leader:6379 --set serviceMonitor.enabled=true --set serviceMonitor.namespace=redis
helm install redis-follower-exporter prometheus-community/prometheus-redis-exporter -n redis --set redisAddress=redis://redis-follower:6379 --set serviceMonitor.enabled=true --set serviceMonitor.namespace=redis

kubectl delete serviceaccount gwydion -n redis 2>/dev/null || true
kubectl create serviceaccount gwydion -n redis
kubectl create clusterrolebinding gwydion-admin --clusterrole=cluster-admin --serviceaccount=redis:gwydion 2>/dev/null || true
kubectl create token gwydion -n redis --duration=999h

kubectl cluster-info
