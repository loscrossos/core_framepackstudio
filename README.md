
# core_framepackStudio

This project does not aim at more functionality. *It hardens the core*.

## Features
- **one-in-all**. Includes latest version of: Framepack, Framepack-F1, Framepack-Studio
- **CrossOS**: works on MacOS, Windows and Linux
- **Fully Accelerated** comes with built-in:
    - MPS acceleration (MacOS Silicon): uses Metal native implementation where available.
    - CUDA fully enabled (Win/Lin) for all accelerators. 
    - All accelerators: xFormers, FlashAttention and SageAttention included for Win/Lin
    - Full support for Nvidia 50xx series GPUs (Blackwell) with custom built libraries and code fixes
    - Benchmarked speed for efficient setup
- **Portable flexibility**:
    - Can re-use your existing/downloaded models (no re-download needed)
    - Can use models stored anywhere: other Framepack install, separate drive, USB drive, etc. 
    - Portable model downloader included if you dont have models yet
- **Efficient & Easy Setup**:
    - install from zero to ready in 5 copy paste commands. 
    - Configuration without touching python code
    - Easy update for future versions
- **Full CrossOS support**: Optimized for dual-boot (actually x-multi-boot)
    - install&update using *the same* standard python commands across all OS 
    - the same installation (e.g. on a shared drive) can be run from MacWinLin simultaneously. 
- **Improved robustness**:
    - Documentation of critical parameters that can lead to system crashes (in Docu and GUI)
    - GUI and console system messages improved for understandability
    - Adds improvement for race conditions on CrossOS code
- **Extra features**: 
    - Free of in-app advertisement
    - fps control for video generation
    - reduced command line output to the most relevant messages
    - Mac: Monitoring tool for disk usage monitoring

*Project currently does NOT support AMD GPUs (ROCm) or pure CPU setups*

Installation Tutorial step-by-step (same information as here): coming up


### Contents 


[Installation](#installation)  
[Usage](#usage)  
[Benchmark](#benchmark)  
[Known Issues](#known-issues)  
[Credits](#credits)

You might be wondering how we got here... some days ago i got my hands into Framepack and tried to install. Noticed that the libraries provided, didnt support my Blackwell card and setting it up was not straight forward... tried to see about Mac and it got even worse. Before i knew it, i was knee deep in code, chain-sawing my way through macOS in "hurt me plenty"-mode... specially macOS... let me share my findings.

This project is the documentation that i didn't manage to find on the internet. Lets start with the installation...








# Installation

The installation in general consists of:
- Pre-Requisites: Check that your system can actually run the model
- Project Installation: setup the core_framepackstudio code
- Let the software download the models or optionally: 
    - setup to re-use your already downloaded models 
    - setup the models for multi-boot setups

In order to install we will use the console. There is plenty of caveats you should know before doing using AI models in general.We need to have control over what is happening. Framepack moves along the edge of technology and, as we will later see, can perform poorly or even crash your system (see on MacOS only) if not setup properly. To be as efficient as possible, we will use standard python mechanisms. The procedures we use, will be also useful for other AI projects across MacOS, Windows and Linux you will encounter in the future. 

Remember: all the risks and requirements i list here are **not mine**. They come from the original Framepack/Studio and, in general, are present in all AI projects nowadays. In this project, i am just documenting my findings based on testing and code analysis.

## TLDR Installation

These are the summarized commands to install and run core_framepackStudio. Detailed steps follow below.

**Mac**
```
git clone https://github.com/loscrossos/core_framepackstudio
cd core_framepackstudio

python3.12 -m venv .env_mac
. ./.env_mac/bin/activate

pip install -r requirements.txt
```


***Windows***
```
git clone https://github.com/loscrossos/core_framepackstudio
cd core_framepackstudio

py -3.12 -m venv .env_win
.env_win\Scripts\activate

pip install -r requirements.txt
```




**Linux**
```
git clone https://github.com/loscrossos/core_framepackstudio
cd core_framepackstudio

python3.12 -m venv .env_lin
. ./.env_lin/bin/activate

pip install -r requirements.txt
```

**All OSes**
You can use one of these optional steps (detailed steps below):
- **Option 1**: automatic model download: just go to the next step and start the app!

- **Option 2**: Manual triggered Model Donwload: enter the `models` dir and use the `maclin_get_models.sh` or `win_get_models.bat`

- **Option 3a**: reuse your models: Place your model directories or `hf_download` folder in the `models`folder. Check that it worked with: `python appframepack.py --checkmodels`
- **Option 3b**: reuse your models without changing their paths: run  `python appframepack.py --checkmodels` after install to generate `configmodels.txt` and edit the paths within the file. run the command again to verify it worked.

**Run the app**

start the apps with any of these:

- Original Framepack:   `python appframepack.py --inbrowser`
- Framepack F1:         `python appf1framepack.py --inbrowser`
- Framepack Studio:     `python appstudio.py --inbrowser`

Stop the app pressing `ctrl + c` on the terminal




## Pre-Requisites

In general you should have your PC setup for AI development when trying out AI models, LLMs and the likes. If you have some experience in this area, you likely already fulfill most if not all of these items. Framepack has some steep requirements that will push the edge on some systems.


### Hardware requirements

There are separate requirements to "install" and to "run" the software.

**Installation requirements**

This seem the minimum hardware requirements:


Hardware    | **Mac**                                   | **Win/Lin**
---         | ---                                       | ---
CPU         | Mac Silicon (M1,M2....)                   | Will not be used much. So any modern CPU should do
VRAM        | Tested on 16GB. Less will not likely work | Original author says you need 6GB VRAM
RAM         | see VRAM                                  | The more the merrier.
Disk Space  | 90 GB for the models     




**Run requirements**

CrossOS: you need 70 GB of *free storage* between VRAM, RAM and Disk when you actually run the apps.

Normally an *app* would crash. *This one* can crash *your system*. 


This requirement is the *minimal* i saw during minimal settings runs of 10Seconds videos with low resolution. This might be slightly higher with higher settings. Just be mindful of watching your system parameters.

In order to start the software you should have 70GB **free** memory (between VRAM, RAM and Disk) *after*:
- installing this repository 
- storing models (90GB) (you can save models to external storage or USB drive) 
- OS, Drivers and running apps needs. MacOS needs 9,5GB running a framepack browser and nothing else. Windows some 5GB)
- Disk minimum free Quota: on MacOS around 30GB and in general 10% of your disk. I dont have an official source for this. This is what i saw during my tests. System would crash if under 30GB (at least my logger stopped there). An in general SSDs get sluggish if free storage falls under 10%.

Example:

lets say on Windows:
- you have a GPU with 16GB VRAM and a system with 32GB RAM. 
- you need: 7GB for Windows, drivers and your browser. Also your disk should not run under 15GB. 

Then you should have (70 -16 -32 +7 +15 ) = 44GB free on your disk to run Framepack. 

On a Mac with 16GB unified RAM:

70(Framepack) + 9.5 (OS + Browser) -16(RAM) +30 (System Disk requirement) = 93 GB free Disk storage.

Mind you: all this *after* storing the 90GB models.


**Be careful**:
The model will aggressively claim storage during generation and free it again after quitting. Your system will crash/freeze if the required storage is not available.



### Software requirements

You should have the following setup to run this project:

- Python 3.12
- latest GPU drivers
- latest cuda-toolkit 12.8+ (for nvidia 50 series support)
- Git

I am not using Conda but the original Free Open Source Python. This guide assumes you use that.

If you want an automated, beginner friendly but efficient way to setup a software development environment for AI and Python, you can use my other project: CrossOS_Setup, which setups your Mac, Windows or Linux PC automatically to a full fledged AI Software Development station. It includes a system checker to assess how well installed your current setup is, before you install anything:

https://github.com/loscrossos/crossos_setup

Thats what i use for all my development across all my systems. Its also fully free and open source. No strings attached!



## Project Installation

If you setup your development environment using my `Crossos_Setup` project, you can do this from a normal non-admin account (which you should actually be doing for your own security).

Hint: "CrossOS" means the commands are valid on MacWinLin

 ---

Lets install core_framepackstudio in 5 Lines on all OSes, shall we? Just open a terminal and enter the commands.



1. Clone the repo (CrossOS): 
```
git clone https://github.com/loscrossos/core_framepackstudio
cd core_framepackstudio
```

2. Create and activate a python virtual environment  

task       | Mac                         | Windows                   | Linux
---        | ---                         | ---                       | ---
create venv|`python3.12 -m venv .env_mac`|`py -3.12 -m venv .env_win`|`python3.12 -m venv .env_lin`
activate it|`. ./.env_mac/bin/activate`  |`.env_win\Scripts\activate`|`. ./.env_lin/bin/activate`

At this point you should see at the left of your prompt the name of your environment (e.g. `(.env_mac)`)


3. Install the libraries (CrossOS):
```
pip install -r requirements.txt
```

Thats it.

---

At this point you *could* just start the apps and start generating away... but it would first automatically download the models (all 90GB of them). If you dont have the models yet thats ok. But if you have already downloaded them OR if you have a dual/trial/multiboot machine and want to make them portable, read on...


## Model Installation

The needed models are about 90GB in total. You can get them in 3 ways:
- **Automatic Download** as huggingface cache (easiest way)
- **Manually triggered model download** (reccomended way. second easiest)
- **Re-use existing models**: hf_download or manual

to see the status of the model recognition start any app with the parameter `--checkmodels`

e.g. `python appstudio.py --checkmodels`
The app will report the models it sees and quit without downloading or loading anything.


### Automatic download
just start the app. 

Missing models will be downloaded. This is for when you never had framepack installed before. The models will be downloaded to a huggingface-type folder in the "models" directory. This is ok if you want the most easy solution and dont care about portability (which is ok!). This is not reccomended as its not very reusable for software developers: e.g. if you want to do coding against the models from another project or want to store the models later. This supports multi-boot.

### Manually triggered automatic download


This is the CrossOS reccomended way. change to the the "models" directory (`cd models`) and start the downloader file:

task     | Mac Linux              | Windows   
---      | ---                    | ---       
manual dl|`./maclin_get_models.sh`|`win_get_models.bat`


Models will be downloaded from hugging face. This will take some time as its 90GB of models. let it work.


### Re-use existing models


You can re-use your existing models by configuring the path in the configuration file `modelconfig.txt`.
This file is created when you first start any app. Just call e.g. `python appstudio.py --checkmodels` to create it.
Now open it with any text editor and put in the path of the directory that points to your models. 
You can use absolute or relative paths. If you have a multiboot-Setup (e.g. dualboot Windows/Linux) you should use relative paths with forward slashes e.g. `../mydir/example`

There are 2 types of model downloads: the hugginface (hf) cache and manual model download.

**Re-Use Framepack/Framepack-Studio files**

If you used Framepack/Framepack-Studio you should have a folder called `hf_download` in your app folder. You can do one of these:

- move that folder to the `core_framepackstudio/models` folder and it will be used automatically OR
- replace the set path with the one of the existing folder in the line with `HF_HOME`. Make sure that the line points only to the single 'hf_download' folder. The app will crawl the sub-directories on its own. You dont need to change any other lines as these will be ignored.

If you only used Framepack or Framepack_F1 then the other model will be missing and downloaded when needed. If you used Framepack-Studio then you should have both models already.


**Re-Use manually downloaded models**

If you downloaded the single models directly from huggingface (git cloning them) you can enter the path of each directory in the single lines of the config file.
You dont need to set the `HF_HOME` line as it will be ignored if the other paths are set correctly.


**Checking that the models are correctly configured**

You can easily check that the app sees the models by starting any of the demos with the parameter `--checkmodels` and checking the last line.

e.g. `python appstudio.py --checkmodels`

```
[MISSING]: /Users/Shared/github/core_framepackstudio/models/hf_download/hub/models--hunyuanvideo-community--HunyuanVideo/
[MISSING]: /Users/Shared/github/core_framepackstudio/models/hf_download/hub/models--lllyasviel--flux_redux_bfl/
[MISSING]: /Users/Shared/github/core_framepackstudio/models/hf_download/hub/models--lllyasviel--FramePack_F1_I2V_HY_20250503/
[MISSING]: /Users/Shared/github/core_framepackstudio/models/hf_download/hub/models--lllyasviel--FramePackI2V_HY/
Searching Group2: Manual:
[!FOUND!]: /Users/Shared/github/core_framepackstudio/models/HunyuanVideo/
[!FOUND!]: /Users/Shared/github/core_framepackstudio/models/flux_redux_bfl/
[!FOUND!]: /Users/Shared/github/core_framepackstudio/models/FramePack_F1_I2V_HY_20250503/
[!FOUND!]: /Users/Shared/github/core_framepackstudio/models/FramePackI2V_HY/
----------------------------
FINAL FOUNDING: It seems all model directories were found. Nothing will be downloaded!
```

# Usage 
You can use Framepack/F1/Studio as you always have. Just start the app and be creative!

## Starting the Apps
The apps have the following names:
- `appframepack.py` : the original Framepack app
- `appf1framepack.py` : the F1 variant
- `appstudio.py` : Framepack Studio

To start just open a terminal, change to the repository directory, enable the virtual environment and start the app. The `--inbrowser` option will automatically open a browser with the UI.

task         | Mac                         | Windows                   | Linux
---          | ---                         | ---                       | ---
activate venv|`. ./.env_mac/bin/activate`  |`.env_win\Scripts\activate`|`. ./.env_lin/bin/activate`


for example the studio app (CrossOS)
```
python appstudio.py --inbrowser
```

A browser should pop up with the UI


To stop the app press `ctrl-c` on the console (CrossOS)





## App Usage



### General Parameters

Definitions:

A "run" is one go of generation. At the end you get a chunk of video. A "run" is composed of iterations (called steps in the UI). A iteration is one attempt at image improvement. the more iterations per run the better image.


- **Video length**: how many seconds your video will have. Since one run is roughly 30 frames this and FPS affects the amount of runs the AI will do.
- **Width**: the width of the video. Bigger width is better quality but slower generation. BEWARE: during generation it seems a full picture must fit in the free VRAM at once. If you have low VRAM the software will crash on the console. The browser UI might not see this and you might be waiting for ages without knowing what happened. If you get crashes lower this value.
- **FPS**: frames per second. It seems one run  of the model produces around 30 frames. FPS=30 is one second of video per iteration. If you set this value to low FPS (like 8), then one iteration will produce almost 4 seconds of video! This way you can get faster generations for sketching. I am not sure if more or less FPS consume more VRAM.
- **Seed**: this is a random number that starts the image generator. A seed can influence the generation. you can leave it at random or try any number you like. Theoretically the same number should produce the same results. Try your lucky number. 

### Advanced Parameters

**Steps**

(default: 25) How many iterations of improvement the AI does on each run. One run is roughly 30 frames of video. More iterations means better image but longer processing. On the console you see the speed in `s/it`; this means "seconds per iteration". If you see `5s/it` it means one run with 25 iterations will take `25*5 = 125 seconds` -> 2 minutes per run. You can try lower iterations for better speed but also for artistic effects. In my experience you can go as low as 15 Steps and have nice results. just try it out! 

**GPU Memory preservation . (This value can fully crash your System)**

Default: 6GB on Windows, 10.3GB on MacOS.

This is a critical setting. The allocated memory is reserved and not used for inference—the code actively preserves it. Once reached, the system avoids loading model layers into VRAM, instead using extended memory (RAM/disk). It’s unclear whether this reservation is strictly for the OS or also includes inference-related tasks. Any unreserved memory will be aggressively utilized.

Risks if set too low:

- Windows/Linux: GPU throttling, slower generation, or system freezes.
- MacOS: System crash, shutdown, and restart.

**Configuration Guide**

- **Unified RAM Systems (MacOS)**: Critical for shared VRAM/core memory. Set this to at least the OS’s baseline requirement (including active apps like your browser). macOS typically needs ≥8GB + browser overhead—verify via system monitor or risk system crashes.

- **Windows/Linux**: Check nvidia-smi for idle VRAM usage (e.g., window manager) and ensure the value is never set below that threshold.

Note: This option only applies if you have <60GB free VRAM.





















# Benchmark

## Overview

I benchmarked the Framepack generation across the systems to see what helps performance and what not. Could not find a guide on what is really needed and how much something helps, so I measured the performance improvement of the accelerators that are used. Lets take a look.

### Accelerator overview

Framepack uses 4 Attention-Accelerators. This is the current official state libraries:

Lib          |MacOS         | Windows                                               | Linux                                                     |
 ---         | ---          | ---                                                   | ---                                                       |   
Torch  attn  | works        | works                                                 | works                                                     |        
xformers     | not available| Pypi* install works. Does NOT support RTX50(Blackwell)| Pypi Version works. Does NOT support RTX50xx(Blackwell)  |
flash_attn   | complicated  | does not work from Pypi. must be compiled separately  | does not work properly from a requirements.txt. must be compiled separately |
sageattention| complicated  | does not work from Pypi. must be compiled separately  | Old version in Pypi for Blackwell. must be compiled separately  |

* Pypi is "is the official third-party software repository for Python"

As you can see they all kind-of do not work or are complicated to setup in away that works with GPUs. Specially the current 50 series of Nvidia. The accelerators are made for CUDA and therefore not available for Mac Silicon. There are projects trying to accomplish this but none in a stable state to my knowledge.
I compiled them all on my own ensuring full support of the newest CUDA version. For xFormers, i also made a fix to support Blackwell cards, that i submitted and hope will be accepted:
https://github.com/facebookresearch/xformers/pull/1254

In the meantime i uploaded my compiled libraries to github so you can just use them directly without having to mess with source code. :D

Triton is a requirement for Flash-Attention. It is not directly available from Pypi. 
It also is not available under windows. Luckily the wonderful triton-windows project provides such a library. 
I also took some hours of effort to compile Sage-attention for Blackwell... only to notice afterwards, that the maker of triton-windows offers that ready-to-go. It was fun at least.


By default, it will just use the built-in PyTorch attention. If available it will use these accelerators in this order:


1. sageattention. IF not found then: 
2. flash_attn. If not found then: 
3. xformers. If not found then:
4. Pytorch Attention (built-in)

If it finds the first  accelerator it will ignore the rest. So if you have Sageattn you dont need to install the others, since they will be ignored anyway.



## Measurements

I ran around 20 batches of generations on each system to get reliable results. Here is my findings.

### Benchmark Setup

I benchmarked my installation on:

&nbsp;  | MacOS     |Windows/Linux
---     | ---       | ---
CPU     | M1        |12Core
RAM     |16GB       |64GB       
GPU     |integrated |RTX 5060ti 
VRAM    | unified   |16GB      
Storage | SSD       |NVME

For the benchmark first i installed all accelerators and ran video generation. After each measurement i removed one accelerator and measured again. That way you can see how the accelerators influence speed.


One run is one second of video using the standard settings of 30FPS. The First run (1st second of video) is always slower since the model must be read from disk. So i only benchmarked the second run.

I measured the setup on Windows and Linux. MacOS wasnt my focus for the benchmark as my hardware is... "*slow*". Also the Accelerators are CUDA centered and not available for MacOS. There are projects trying to support them but not stable to my knowledge.

I used always the same settings. The Values themselves are not relevant to you as they depend on your hardware. Whats important is to see the relative acceleration that you can expect from using each accelerator.

### Benchmark Results

**Speed**

Measurements are in "time for each generation run". For simplicity in this benchmark we assume 1 "run" is roughly one second of final video output (in reality this is affected by FPS).

For example: with sageattention it takes 2:12 minutes to generate 1 second of video (Lower is better):

Lib           |MacOS  |Windows| Linux| 
 ---          | ---   | ---   | ---  | 
Torch         | x:xx  | 4:10  | 4:30 |
xformers      | x:xx  | 2:50  | 4:52 |
flash_attn    | x:xx  | 2:50  | 2:55 |
sageattention | x:xx  | 2:11  |2:25-3:28|

The same measurements, but in `s/I`[(Seconds per Iteration)](###General-Parameters) for each generation run (1 run is one second of final video output). a 5 Seecond video is 5 runs (lower is better):


Lib            |Windows| Linux|
 ---           | ---   | ---  |
Torch          | 10.3  | 10.82|
xformers       | 6.84  | 11.71|
flash_attn     | 6.83  | 7.08 |
sageattention  | 5.25  |5.8-8.35|

SageAttention is the fastest accelerator. On Linux i got mixed results with some runs being fast and others not so much. Still Sage attention is your best bet.

Xformers is currently officially not compatible with blackwell. So i submitted a patch to the project to fix it. Expect a new official release soon i guess. Still it was slower than all of them.

https://github.com/facebookresearch/xformers/pull/1254


**Memory consumption**

The code determines at start, wether you have at least 60GB of free VRAM available (GPU storage). If not it enables all kinds of VRAM saving mechanism (tiling, dicing, unloading models whenever possible, etc). Its in this situation that the memory preservation slider is used.

In general the accelerators didn't influence memory consumption. But Windows used WAY more memory than Linux.
The process reserved RAM on Windows was 80GB. On Linux looked like 15GB. In the first place i am not sure why Linux is NOT consuming that much RAM.

The process needs a lot of memory. In total around 80GB, *on top* of your Systems needs. Ideally GPU VRAM. If you dont have 80GB VRAM, it will use RAM, and if there is not enough and the system allows it, then it is using a harddrive for swap. So you should have some 80GB available in your system + what your OS needs to live: either in a mix of VRAM and RAM or you should at least allow dynamic or enough virtualmemory and have enough disk space your swap drive (windows usually C: and linux the swap partition). This is specially critical on MacOS as the libraries will just take the memory. If there is not enough, the system is going to crash mercilessly.

Also take into account that you can not just have a disk go to 0GB free. MacOS will stop your disk from getting less than around 30GB space and will crash.



### Takeaway

To speed up Framepack this is your best course:

- Using Sageattention as accelerator is the best option. 
- Also you need VRAM the more the merrier.
- Then you need RAM.. the more the merrier.
- Or you need the fastest hard drive you can get: NVMe, SSD, HD... 3.5" Disk?

CPU will not affect generation.



# Known Issues
Documentation of Issues i encountered and know of.

## General

-Sometimes the first iterations will not show the Preview video. But if you let it run you will definitely see the final result. 

-Sometimes the first run will be slower. It gets better.

## OS Specific

**MacOS**

MacOS in general is still not 100% stable in comparisson to WinLin. I could not test as throughly as it deserves. Still its admirable how this can run on a cheap Mac M1 (16GB) and produce actual results.

- The main Issue is my crappy Mac M1 with 16GB. It was a pain to develop and test on it. If i get a MacStudio at some point i might give it more time.
- The new version of Framepack-Studio redesigned the code to communicate to the UI over a complex Fifo Queue. This introduced some heavy GPU-race conditions that only appear in Mac. This causes the GPU to get in a deadlock after the first iteration and it gets stuck at `FlowMatchUniPC.sample()` first sub-iteration. I had to ugly-hack my way through this. I can't explain why this happens, as this does not happen on Windows or Linux.
- The new version of Framepack-Studio introduced libraries that are not MacOS compatible (decord). I had to replace considerable parts of code to retain functionnality. This is not blaming. I understand MacOS was not in their focus at all, as far as i know. 
- You will need lots of storage. I ran Framepack on an M1 16GB and it took a while. Since the OS needs 7-8GB alone to run, i doubt a Mac with 8GB will be more fun than what i had.  Too high generation settings did crash my machine a couple of times due to System-OOM (out of memory). Didnt Expect MacOS Sequoia to die so easily.
- During generation, you will see "warnings" saying that some function is not implemented in Metal and the code is falling back to CPU which might "affect performance". This is not a bug... its a feature! (the only time ever, that this sentence is true and not some half assed joke!): This means we are already using Metal native code, that torch supports but has not implemented yet. Normally this would just crash but 1) Torch can handle it and 2) since the code is already there, when a new version of Torch supports this feature, this part of the code will automatically run faster as soon as you update torch. If i had done code without warnings, this would always stay slow. So this can only get better!.

**Windows**

None. Works like a charm.

**Linux**

Works greatly. 

-Sometimes the server can not be shutdown if you try to shut down during generation. You can see that the GPU is still using memory. You can do one of these:
- run nvidia-smi and note down the process ID (PID) of `python` and kill that process with: `pkill -9 [insert PID here]`
- restart your PC. This kills orphaned processes

This might not be Linux specific. It just happened to me only i Linux during my limited tests.



## Troubleshooting

### Free Disk Storage

You will need it. Here are some tips how to free disk space if you are desperate for those last GB:

CrossOS:
- Clear your pip cache: you can run `pip cache purge` on the terminal. If you have installed lots of projects this will free several GB. Downside: you wont have a cache and next time you install torch it will be redownloaded. Still this should be done from time to time.
-

Windows:
- Disable Hibernation: Hibernation is not so useful anyways.. is it?


Linux:
- Disable hibernation. See windows.

MacOS:
- Restart once in safe mode: A well guarded secret. When you restart in safe mode MacOS will clean up your disk! freed like 15GB on my disk out of the blue.


### Random Issues

- **"RuntimeError: CUDA error: no kernel image is available for execution on the device":** You didnt use my libraries.. and your libraries were not compiled with the CUDA Toolkit that supports your card. If you have a 50 Series card you need libraries compiled with at least CUDA Toolkit 12.8 
- **System crash**: only happened to me when setting the memory preservation low. see the guide.
- **Sometimes the video preview does not appear**: this happens when your system is too busy with calculations. Then the UI will choke, as the generation process will have priority. give it more time. This can happen on Framepack-Studio on Mac more often as this was the hack i needed for it to work at all. 

















# Credits
This project was built based on the power of open source. Ideas built and shared with the world under a permissive licence so that others can take that work and build upon it.

Main credit goes therefore to the original Framepack repository by Lllyasviel, upon all projects (including this one) are built upon:

https://github.com/lllyasviel/FramePack

For the MacOS code 90% of code and credit goes to this guy where i got the code inspiration for the MacOS code:

https://github.com/brandon929/FramePack

Also there are people working on a MacOS branch on the Lllyasviel-Framepack Repo:

https://github.com/mdaiter/FramePack/tree/tinygrad

Also the Framepack-Studio fork of Framepack, with fresh ideas:

https://github.com/colinurbs/FramePack-Studio


The wonderful woct0rdho for his windows implementation of triton:

 https://github.com/woct0rdho/triton-windows


