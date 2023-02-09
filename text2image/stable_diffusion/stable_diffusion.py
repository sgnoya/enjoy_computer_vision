# %%
import os
import random

import numpy as np
import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
from matplotlib import pyplot as plt


def make_deterministic(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = "16:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    # torch.use_deterministic_algorithms(True)


model_name = "stabilityai/stable-diffusion-2"
scheduler = EulerDiscreteScheduler.from_pretrained(
    model_name, subfolder="scheduler"
)
model = StableDiffusionPipeline.from_pretrained(
    model_name, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16
)
model = model.to("cuda")

# %%
prompt = "MONKEY･D･LUFFY on the top of Mount Fuji."
seed = 1234
make_deterministic(seed)
image = model(prompt, height=512, width=512).images[0]
plt.imshow(image)
plt.show()
image.save("luffy.png")

# %%

prompt = "Highly realistic image of a white-haired, long-haired, long-bearded, Japanese-looking hermit who scuba dives in Hawaii, with the naked eye."

seed = 123
make_deterministic(seed)
image = model(prompt, height=512, width=512).images[0]
plt.imshow(image)
plt.show()
image.save("hermit1.png")

seed = 1234
make_deterministic(seed)
image = model(prompt, height=512, width=512).images[0]
plt.imshow(image)
plt.show()
image.save("hermit2.png")

# %%
