# ajdsim

## Description

The package `pearsondist` is a `Python` package designed for constructing Pearson distribution systems. 

## Simple Usage

installment (in the directory that contains `pyproject.py`):

```bash
pip install .
```

An example:

```python
from pearsondist import Pearson8
import numpy as np
# provide the first eight moments, no more, no less
moment = [0.0679246, 0.0200644, 0.0011987, 0.0013033,
          -0.0002338, 0.0002833,-0.0001786, 0.0001697]
pearson = Pearson8(moment)
x = np.linspace(-1, 1, 100)  # adjust the support accordingly
pdf = pearson.pdf(x)        # density values
```

## Documentation

The documentation would probably be hosted on <http://www.yyschools.com/pearsondist/>

## Ongoing Development

This code is being developed on an on-going basis at the author's [Github site](https://github.com/xmlongan/pearsondist).

## Support

For support in using this software, submit an [issue](https://github.com/xmlongan/pearsondist/issues/new).
