
# User guide

The following pages describe how to use Xplt for plotting of data from Xsuite or simmilar accelerator physics codes.

```{toctree}
:caption: Contents
:maxdepth: 1

examples/colors
examples/twiss
examples/phasespace
examples/hamiltonians
examples/animations
```

:::{tip}
Xsuite is not an explicit dependency, rather an API assumption on available attributes, indices and units. You can use data from any source, for example:

```python
import xplt
import numpy as np

particles = dict(
    x = np.random.normal(size=int(1e5)),  # in m
    px = np.random.normal(size=int(1e5)),  # in rad
)
xplt.PhaseSpacePlot(particles, kind='x-px')
```

:::
