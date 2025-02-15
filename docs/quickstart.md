
# Quickstart

## Installation

```bash
pip install xplt[recommended]
```

The following extras are available:
- `xplt[minimal]` Minimal installation, certain features like unit conversion and unit resolving are disabled
- `xplt[recommended]` Recommended default
- `xplt[full]` Includes optional dependencies, adds support for pandas data frames

Currently, `pip install xplt` defaults to `xplt[minimal]`. Once [PEP 771](https://peps.python.org/pep-0771/) is established, this will change to `xplt[recommended]` instead.


## Gallery

Click on the plots below to show the respective section in the [User guide](usage):

|Floor plan from survey | Twiss parameters | Phasespace distributions|
|:-:|:-:|:-:|
|[![](gallery/floorplot.png)](examples/line.ipynb#survey) | [![](gallery/twissplot.png)](examples/twiss) | [![](gallery/phasespaceplot.png)](examples/phasespace) |

| Beam positions | Spill time structure | Spill quality |
|:-:|:-:|:-:|
| [![](gallery/bpm.png)](examples/timestructure.ipynb#binned-time-series) | [![](gallery/spill.png)](examples/timestructure) | [![](gallery/spillquality.png)](examples/timestructure.ipynb#spill-quality) |
