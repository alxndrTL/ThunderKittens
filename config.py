### ADD TO THIS TO REGISTER NEW KERNELS
sources = {
    'attn': {
        'source_files': {
            'h100': 'kernels/attn/h100/h100.cu' # define these source files for each GPU target desired.
        }
    },
    'hedgehog': {
        'source_files': {
            'h100': 'kernels/hedgehog/hh.cu'
        }
    },
    'based': {
        'source_files': {
            'h100': [
                'kernels/based/lin_attn_h100.cu',
            ],
            '4090': [
                'kernels/based/lin_attn_4090.cu',
            ]
        }
    },
    'cylon': {
        'source_files': {
            'h100': 'kernels/cylon/cylon.cu'
        }
    },
    'flux': {
        'source_files': {
            'h100': [
                'kernels/flux/flux_gate.cu',
                'kernels/flux/flux_gelu.cu'
            ]
        }
    },
    'fftconv': {
        'source_files': {
            'h100': 'kernels/fftconv/pc/pc.cu'
        }
    },
    'fused_rotary': {
        'source_files': {
            'h100': 'kernels/rotary/pc.cu'
        }
    },
    'fused_layernorm': {
        'source_files': {
            'h100': 'kernels/layernorm/non_pc/layer_norm.cu'
        }
    },
    'mamba2': {
        'source_files': {
            'h100': 'kernels/mamba2/pc.cu'
        }
    },
    'lin_attn': {
        'source_files': {
            '4090': [
                'my_kernels/lin_attn/4090.cu',
            ]
        }
    }
}

### WHICH KERNELS DO WE WANT TO BUILD?
# (oftentimes during development work you don't need to redefine them all.)
kernels = ['lin_attn']

### WHICH GPU TARGET DO WE WANT TO BUILD FOR?
target = '4090'
