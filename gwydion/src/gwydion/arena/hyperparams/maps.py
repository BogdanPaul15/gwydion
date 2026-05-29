from torch import nn

NET_ARCH_MAP: dict[str, dict[str, list[int]]] = {
	"tiny": {"pi": [64], "vf": [64]},
	"small": {"pi": [64, 64], "vf": [64, 64]},
	"medium": {"pi": [256, 256], "vf": [256, 256]},
}

ACTIVATION_FN_MAP: dict[str, type[nn.Module]] = {
	"tanh": nn.Tanh,
	"relu": nn.ReLU,
	"elu": nn.ELU,
	"leaky_relu": nn.LeakyReLU,
}
