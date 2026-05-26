#!/bin/bash
set -euo pipefail

minikube start -p onlineboutique --cpus=4 --memory=8192 --driver=docker --addons=metrics-server

kubectl apply -f onlineboutique.yaml

helm install prom prometheus-community/kube-prometheus-stack --namespace onlineboutique -f values.yaml

kubectl create serviceaccount gwydion -n onlineboutique
kubectl create clusterrolebinding gwydion-admin --clusterrole=cluster-admin --serviceaccount=onlineboutique:gwydion
kubectl create token gwydion -n onlineboutique --duration=999h

kubectl cluster-info
