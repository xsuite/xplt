
# User guide

The following pages describe how to use Xplt for plotting of data from Xsuite or simmilar accelerator physics codes.

```{toctree}
:caption: Contents
:maxdepth: 1

examples/concepts
examples/line
examples/twiss
examples/phasespace
examples/hamiltonians
examples/timestructure
examples/animations
examples/colors
```

:::{tip}
Xsuite is not an explicit dependency, rather an API assumption on available attributes, indices and units. You can use data from any source, and also custom attributes. See {doc}`examples/concepts` on how to specify units of custom attributes.

```python
import xplt
xplt.apply_style()  # use our matplotlib style sheet
import numpy as np
import pandas as pd

# Dictionary
particles = dict(
    x = np.random.normal(size=int(1e5)),  # in m
    px = np.random.normal(size=int(1e5)),  # in rad
    a = np.random.normal(size=int(1e5)),  # custom attribute
)
xplt.PhaseSpacePlot(particles, kind='x-px', data_unit=dict(a="km"))

# Pandas DataFrame
df = pd.DataFrame(particles)
xplt.PhaseSpacePlot(df, kind='x-px')

...
```

:::
