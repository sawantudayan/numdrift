# numdrift
A high-performance, memory-efficient library for numeric data manipulation with native missing data support.

## Features
- Native support for missing data
- Fast performance (at least as good as Pandas)
- Memory-efficient storage optimized for large datasets
- Sparse array support for reduced memory footprint
- Serialization & I/O support

## Installation

To install Numdrift, use pip: 
```shell script
pip install numdrift
```


## Usage

```python
import numdrift as nd

# Create a Numdrift array
arr = nd.mdarray([1, 2, 3, None, 5])

# Perform operations on the array
result = arr.mean()
print(result)


