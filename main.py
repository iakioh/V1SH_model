# example simulation using V1_model with neighboring textures input

import numpy as np
from v1sh_model.models.V1_model import V1_model
from v1sh_model.inputs.examples import neighboring_textures
from v1sh_model.inputs.visualize import visualize_input, visualize_output

if __name__ == "__main__":
    seed = 42
    model = V1_model(seed=seed, alpha_x=1.0, alpha_y=1.0)
    A, C = neighboring_textures(22, 60, 2.0)

    T = 12.0
    dt = 0.05
    X_gen, Y_gen, I = model.simulate(
        A, C, dt=dt, T=T, verbose=False, noisy=False, mode="wrap"
    )
    smap, A_out = model.saliency_map(X_gen)
    
    visualize_input(A, C)
    visualize_output(A_out, smap)
    
    