# example simulation using V1_model_1 with neighboring textures input

from v1sh_model.models.V1_model_1 import V1_model_1
from v1sh_model.inputs.examples import neighboring_textures

if __name__ == "__main__":
    seed = 42
    model = V1_model_1(seed=seed, alpha_x=1.0, alpha_y=1.0)
    A, C = neighboring_textures(22, 60, 2.0)

    T = 12.0
    dt = 0.001
    X_gen, Y_gen, I = model.simulate(
        A, C, dt=dt, T=T, verbose=False, noisy=True, mode="wrap"
    )