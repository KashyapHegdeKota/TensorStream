from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='tensorstream_ops',
    ext_modules=[
        CppExtension(
            name='tensorstream_ops', 
            sources=['src/voxel_ops.cpp'],
            extra_compile_args=['-O3'] # Turn on optimization!
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)