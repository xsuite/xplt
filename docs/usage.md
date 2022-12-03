
# User guide

The following pages describe how to use Xplt for plotting of data from Xsuite or simmilar accelerator physics codes.

```{toctree}
:caption: Contents
:maxdepth: 1

examples/colors
examples/twiss
examples/phasespace
examples/hamiltonians
examples/timestructure
examples/animations
```

:::{tip}
Xsuite is not an explicit dependency, rather an API assumption on available attributes, indices and units. You can use data from any source, for example:

```python
import xplt
import numpy as np
import pandas as pd

# Dictionary
particles = dict(
    x = np.random.normal(size=int(1e5)),  # in m
    px = np.random.normal(size=int(1e5)),  # in rad
)
xplt.PhaseSpacePlot(particles, kind='x-px')

# Pandas DataFrame
df = pd.DataFrame(particles)
xplt.PhaseSpacePlot(df, kind='x-px')

...
```

:::
