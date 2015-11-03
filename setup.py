

import distutils.core
import Cython.Build


distutils.core.setup(
    ext_modules = Cython.Build.cythonize('photonic_tomo/_cyquadrature.pyx', gdb_debug=False),
)
