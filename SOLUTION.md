## How to run

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install tqdm

python validate.py --data_dir ./data --batch_size 64 --n_batches 128 --output results.json
```

Tested on Python 3.13, torch 2.11.0+cu128, RTX 4070 SUPER. CIFAR-100 downloads automatically.

---

## My approach

I kept the optimizer simple: SPSA to estimate gradients + Adam to update the weights. Only `fc.weight` and `fc.bias` are tuned.

The main thing that actually helped was **initializing fc.weight with class mean features**. Before any ZO steps, I run all training images through the ResNet18 backbone, compute the average feature vector for each of the 100 classes, and use those as the initial weights. This gives the model a decent starting point right away — around 17-20% accuracy before optimization even starts. The 128 ZO steps then push it a bit further.

For SPSA I used Rademacher perturbations (`eps=1e-3`) and Adam with `lr=1e-2`.

---

## What I tried but didn't keep

**Small-scale init (std=0.01).** Thought a flat loss landscape would be easier to optimize. It wasn't — the gradient signal basically disappeared and accuracy dropped to ~1%.

**Cone sampling.** Tried biasing the perturbation toward the current momentum direction. Looked good in theory, made things worse in practice. The early gradient estimates are too noisy for this to work.

**More SPSA samples per step.** Averaging 4–8 estimates per step reduces noise, but with 51k parameters the signal-to-noise ratio is already so low that it didn't help much.

**RandomCrop augmentation.** Added random cropping to training transforms. Created a mismatch with the validation pipeline and hurt accuracy.

---

## Main takeaway

With 8192 samples and SPSA on 51k parameters, each gradient update is pretty noisy. Trying to improve the optimizer itself gave small gains. Getting a good starting point (class-mean init) gave the biggest jump by far.
