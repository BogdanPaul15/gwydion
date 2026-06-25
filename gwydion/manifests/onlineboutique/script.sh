#!/bin/bash
set -euo pipefail

minikube start -p onlineboutique --cpus=4 --memory=8192 --driver=docker --addons=metrics-server

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install prom prometheus-community/kube-prometheus-stack --namespace onlineboutique --create-namespace -f values.yaml

kubectl wait --for condition=established --timeout=120s crd/servicemonitors.monitoring.coreos.com

kubectl apply -f grafana-dashboard.yaml

kubectl create serviceaccount gwydion -n onlineboutique
kubectl create clusterrolebinding gwydion-admin --clusterrole=cluster-admin --serviceaccount=onlineboutique:gwydion
kubectl create token gwydion -n onlineboutique --duration=999h
kubectl cluster-info

kubectl apply -f onlineboutique.yaml

kubectl -n onlineboutique port-forward svc/prom-grafana 3000:80 &
kubectl -n onlineboutique port-forward svc/prom-kube-prometheus-stack-prometheus 9090:9090 &
kubectl -n onlineboutique port-forward svc/frontend 8080:80 &

python run.py --phase test --use-case onlineboutique --goal smooth_cost --alg maskable_ppo --model "arena_results/MaskablePPO_OnlineBoutique/train_SmoothCostStrategy_steps=250000,label=default_20260607_085211_843393/model_final.zip" --stats "arena_results/MaskablePPO_OnlineBoutique/train_SmoothCostStrategy_steps=250000,label=default_20260607_085211_843393/vecnormalize.pkl" --n-episodes 1000 --live-metrics --record-step-obs
