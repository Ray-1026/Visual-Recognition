# NYCU Visual Recognition using Deep Learning HW4

Student ID: 110550093

Name: 蔡師睿

## Introduction
This task addresses image restoration under adverse weather conditions, specifically rain and snow. The training dataset comprises 1,600 degraded images for each condition (3,200 images in total), and the test set includes 100 degraded samples. The constraint is that only the PromptIR model may be used, and all training must be trained from scratch. Our goal is to produce high-quality restorations, with Peak Signal-to-Noise Ratio (PSNR) serving as the evaluation criterion.

## How to install

```bash
conda create -n hw4 python=3.10
conda activate hw4
pip install -r requirements.txt
```

## Usage

```bash
./run.sh
```

## Performance Snapshot

![image](./assets/score.png)