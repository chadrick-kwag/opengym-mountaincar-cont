# Mountain Car Continuous OpenGym Actor-Critic Model


## Environment

- Ubuntu 20.04.3
- pytorch 1.9.0
- cuda 10.1
- python 3.8.10

checkout `requirements.txt` for used python packages

## How to Train 


```
$ python train.py -gpu 0 \
-actor_lr 0.0001 \ 
-critic_lr 0.0001 \
-suffix this_is_a_testrun \
```

more options: 
```
  -suffix SUFFIX        train session save string to use as suffix
  -actor_lr ACTOR_LR    actor learning rate
  -critic_lr CRITIC_LR  critic learning rate
  -gpu GPU              if using gpu, then provide gpu number
  -epinum EPINUM        number of episodes to train
  -patient_epinum PATIENT_EPINUM
                        episode count to wait for first success
  -gamma GAMMA          reward diminish gamma value
```

## Inference

```
$ cd infer 
$ python run_with_ckpt.py -actor_ckpt /path/to/actor.pt \
-critic_ckpt /path/to/critic.pt \
-scaler_file /path/to/scaler_param
```

running inference script like this will run the continuous mountain car for 10 times with the given actor/critic ckpts.

## Trained Checkpoints

Here are the files that I have trained. Download all files in this shared dir

https://drive.google.com/drive/folders/1CGdpC8SYuQPpYU2uc4rnjM3WSwrX8g5z?usp=sharing