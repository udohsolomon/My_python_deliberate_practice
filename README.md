
# Python deliberate Practice

## An Introduction to Numerical Computing with Python

While the Python language is an excellent tool for general-purpose programming, with a highly readable syntax, rich and powerful data types (strings, lists, sets, dictionaries, arbitrary length integers, etc) and a very comprehensive standard library, it was not designed specifically for mathematical and scientific computing.  Neither the language nor its standard library have facilities for the efficient representation of multidimensional datasets, tools for linear algebra and general matrix manipulations (an essential building block of virtually all technical computing), nor any data visualization facilities.

In particular, Python lists are very flexible containers that can be nested arbitrarily deep and which can hold any Python object in them, but they are poorly suited to represent efficiently common mathematical constructs like vectors and matrices.  In contrast, much of our modern heritage of scientific computing has been built on top of libraries written in the Fortran language, which has native support for vectors and matrices as well as a library of mathematical functions that can efficiently operate on entire arrays at once.

## Basics of Numpy arrays

We now turn our attention to the Numpy library, which forms the base layer for the entire 'scipy ecosystem'.  Once you have installed numpy, you can import it as


```python
import numpy as np
```


```python
x = range(50000)
y = np.arange(50000)

%timeit [e**2  for e in x]
%timeit y**2
```

    10 loops, best of 3: 20.5 ms per loop
    10000 loops, best of 3: 71.8 µs per loop


### The Numpy array structure
<center>
<img src="files/array_memory_strides.png" width=70%>
</center>

### Arrays vs lists

As mentioned above, the main object provided by numpy is a powerful array.  We'll start by exploring how the numpy array differs from Python lists.  We start by creating a simple list and an array with the same contents of the list:


```python
lst = [10, 20, 30, 40]
arr = np.array([10, 20, 30, 40])
```

Elements of a one-dimensional array are accessed with the same syntax as a list:


```python
lst[0]
```




    10




```python
arr[0]
```




    10




```python
arr[-1]
```




    40




```python
arr[2:]
```




    array([30, 40])



The first difference to note between lists and arrays is that arrays are *homogeneous*; i.e. all elements of an array must be of the same type.  In contrast, lists can contain elements of arbitrary type. For example, we can change the last element in our list above to be a string:


```python
lst[-1] = 'a string inside a list'
lst
```




    [10, 20, 30, 'a string inside a list']



but the same can not be done with an array, as we get an error message:


```python
arr[-1] = 'a string inside an array'
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-10-29c0bfa5fa8a> in <module>()
    ----> 1 arr[-1] = 'a string inside an array'
    

    ValueError: invalid literal for int() with base 10: 'a string inside an array'


### Array memory representation

The information about the type of an array is contained in its *dtype* attribute:


```python
x = np.array([[1, 2], [3, 4]], dtype=np.uint8)
print(x)
```


```python
[b for b in bytes(x.data)]
```


```python
arr = np.array([10, 20, 123123])
arr.dtype
```


```python
[b for b in bytes(arr.data)]
```

Once an array has been created, its dtype is fixed and it can only store elements of the same type.  For this example where the dtype is integer, if we store a floating point number it will be automatically converted into an integer:


```python
arr[-1] = 1.234
arr
```

Strange things can also happen when manipulating values in an array:


```python
x = np.array([0, 127, 255], dtype=np.uint8)
print(x)
```


```python
x + 1
```


```python
x - 1
```

### Array creation

Above we created an array from an existing list; now let us now see other ways in which we can create arrays, which we'll illustrate next.  A common need is to have an array initialized with a constant value, and very often this value is 0 or 1 (suitable as starting value for additive and multiplicative loops respectively); `zeros` creates arrays of all zeros, with any desired dtype:


```python
np.zeros((5, 5), dtype=np.float64)
```


```python
np.zeros((2, 3), dtype=np.int64)
```


```python
np.zeros((3,2), dtype=complex)
```

and similarly for `ones`:


```python
np.ones(5)
```

Then there are the `linspace` and `logspace` functions to create linearly and logarithmically-spaced grids, respectively, with a fixed number of points and including both ends of the specified interval:


```python
np.linspace(0, 1, 5)   # start, stop, num
```


```python
np.logspace(1, 4, 4)  # Logarithmic grid between 10^1 and 10^4
```

Finally, it is often useful to create arrays with random numbers that follow a specific distribution.  The `np.random` module provides several random number generators.  For example, here we produce an array of 5 random samples taken from a standard normal distribution (0 mean and variance 1):


```python
rng = np.random.RandomState(0)  # <-- seed value, do not have to specify, but useful for reproducibility
```


```python
rng.normal(loc=5, scale=1, size=5)
```

Or the same, but from a uniform distribution:


```python
uni = rng.uniform(-10, 10, size=5)  # 5 random numbers, picked from a uniform distribution between -10 and 10
print(uni)
```

## Indexing with other arrays

Above we saw how to index arrays with single numbers and slices, just like Python lists.  But arrays allow for a more sophisticated kind of indexing which is very powerful: you can index an array with another array, and in particular with an array of boolean values.  This is particluarly useful to extract information from an array that matches a certain condition.

Consider for example that in the array `uni` we want to replace all values above 0 with the value 10.  We can do so by first finding the *mask* that indicates where this condition is true or false:


```python
mask = uni > 0
mask
```

Now that we have this mask, we can use it to either read those values or to reset them to 0:


```python
print('Array:', uni)
print('Masked array:', uni[mask])
```


```python
uni[mask] = 10
print(uni)
```

### Arrays with more than one dimension

Most of the array creation methods can be used to construct >1D arrays:


```python
np.zeros((3, 4))
```


```python
np.zeros((2, 3, 2, 2))
```


```python

```

We can also reshape arrays to fit the desired shape:


```python
np.arange(12)
```


```python
arr = np.arange(12).reshape((3, 4))
arr
```

With two-dimensional arrays we start seeing the power of numpy: while a nested list can be indexed using repeatedly the `[ ]` operator, multidimensional arrays support a more direct indexing syntax with a single `[ ]` and a set of indices separated by commas:


```python
arr[0][1]
```


```python
arr[:, 0]
```

If you only provide one index, then you will get an array with one fewer dimension containing that row:


```python
print('First row:  ', arr[0])
print('Second row: ', arr[1])
```

## Slicing, repeating, tiling

Extracting elements from NumPy array works pretty much like in lists:


```python
x = np.array([1,2,3,4,5,6,7])
print(x[3:])
print(x[::-1])
```


```python
print(x[::-2])
```


```python
print(x[::-5])
```


```python
print(x[::-10])
```


```python
print(x[3::-1])
```


```python
print(x[-2::-3])
```

### Other numpy functions and array properties

Now that we have seen how to create arrays with more than one dimension, let's take a look at some other properties.


```python
print('Data type                :', arr.dtype)
print('Total number of elements :', arr.size)
print('Number of dimensions     :', arr.ndim)
print('Shape (dimensionality)   :', arr.shape)
print('Memory used (in bytes)   :', arr.nbytes)
```

There are also many useful functions in numpy that operate on arrays, e.g.:


```python
print('Minimum and maximum             :', np.min(arr), np.max(arr))
print('Sum and product of all elements :', np.sum(arr), np.prod(arr))
print('Mean and standard deviation     :', np.mean(arr), np.std(arr))
```

For these methods, the above operations area all computed on all the elements of the array.  But for a multidimensional array, it's possible to do the computation along a single dimension, by passing the `axis` parameter; for example:


```python
print('For the following array:\n', arr)
print('The sum of elements along the rows is    :', np.sum(arr, axis=1))
print('The sum of elements along the columns is :', np.sum(arr, axis=0))
```

As you can see in this example, the value of the `axis` parameter is the dimension which will be *consumed* once the operation has been carried out.  This is why to sum along the rows we use `axis=0`.

Another widely used property of arrays is the `.T` attribute, which allows you to access the transpose of the array (NumPy does this without making a copy of the array, by manipulating its strides):


```python
print('Array:\n', arr)
print('Transpose:\n', arr.T)
```

We don't have time here to look at all the numpy functions that operate on arrays, but here's a complete list.  Simply try exploring some of these IPython to learn more, or read their description in their docstrings or the [Numpy documentation](http://docs.scipy.org/doc/numpy/reference/):

```
np.ALLOW_THREADS              np.compress                   np.irr                        np.pv
np.BUFSIZE                    np.concatenate                np.is_busday                  np.r_
np.CLIP                       np.conj                       np.isclose                    np.rad2deg
np.ComplexWarning             np.conjugate                  np.iscomplex                  np.radians
np.DataSource                 np.convolve                   np.iscomplexobj               np.random
np.ERR_CALL                   np.copy                       np.isfinite                   np.rank
np.ERR_DEFAULT                np.copysign                   np.isfortran                  np.rate
np.ERR_IGNORE                 np.copyto                     np.isinf                      np.ravel
np.ERR_LOG                    np.core                       np.isnan                      np.ravel_multi_index
np.ERR_PRINT                  np.corrcoef                   np.isneginf                   np.real
np.ERR_RAISE                  np.correlate                  np.isposinf                   np.real_if_close
np.ERR_WARN                   np.cos                        np.isreal                     np.rec
np.FLOATING_POINT_SUPPORT     np.cosh                       np.isrealobj                  np.recarray
np.FPE_DIVIDEBYZERO           np.count_nonzero              np.isscalar                   np.recfromcsv
np.FPE_INVALID                np.cov                        np.issctype                   np.recfromtxt
np.FPE_OVERFLOW               np.cross                      np.issubclass_                np.reciprocal
np.FPE_UNDERFLOW              np.csingle                    np.issubdtype                 np.record
np.False_                     np.ctypeslib                  np.issubsctype                np.remainder
np.Inf                        np.cumprod                    np.iterable                   np.repeat
np.Infinity                   np.cumproduct                 np.ix_                        np.require
np.MAXDIMS                    np.cumsum                     np.kaiser                     np.reshape
np.MachAr                     np.datetime64                 np.kron                       np.resize
np.ModuleDeprecationWarning   np.datetime_as_string         np.ldexp                      np.restoredot
np.NAN                        np.datetime_data              np.left_shift                 np.result_type
np.NINF                       np.deg2rad                    np.less                       np.right_shift
np.NZERO                      np.degrees                    np.less_equal                 np.rint
np.NaN                        np.delete                     np.lexsort                    np.roll
np.PINF                       np.deprecate                  np.lib                        np.rollaxis
np.PZERO                      np.deprecate_with_doc         np.linalg                     np.roots
np.PackageLoader              np.diag                       np.linspace                   np.rot90
np.RAISE                      np.diag_indices               np.little_endian              np.round
np.RankWarning                np.diag_indices_from          np.load                       np.round_
np.SHIFT_DIVIDEBYZERO         np.diagflat                   np.loads                      np.row_stack
np.SHIFT_INVALID              np.diagonal                   np.loadtxt                    np.s_
np.SHIFT_OVERFLOW             np.diff                       np.log                        np.safe_eval
np.SHIFT_UNDERFLOW            np.digitize                   np.log10                      np.save
np.ScalarType                 np.disp                       np.log1p                      np.savetxt
np.Tester                     np.divide                     np.log2                       np.savez
np.True_                      np.division                   np.logaddexp                  np.savez_compressed
np.UFUNC_BUFSIZE_DEFAULT      np.dot                        np.logaddexp2                 np.sctype2char
np.UFUNC_PYVALS_NAME          np.double                     np.logical_and                np.sctypeDict
np.VisibleDeprecationWarning  np.dsplit                     np.logical_not                np.sctypeNA
np.WRAP                       np.dstack                     np.logical_or                 np.sctypes
np.abs                        np.dtype                      np.logical_xor                np.searchsorted
np.absolute                   np.e                          np.logspace                   np.select
np.absolute_import            np.ediff1d                    np.long                       np.set_numeric_ops
np.add                        np.einsum                     np.longcomplex                np.set_printoptions
np.add_docstring              np.emath                      np.longdouble                 np.set_string_function
np.add_newdoc                 np.empty                      np.longfloat                  np.setbufsize
np.add_newdoc_ufunc           np.empty_like                 np.longlong                   np.setdiff1d
np.add_newdocs                np.equal                      np.lookfor                    np.seterr
np.alen                       np.errstate                   np.ma                         np.seterrcall
np.all                        np.euler_gamma                np.mafromtxt                  np.seterrobj
np.allclose                   np.exp                        np.mask_indices               np.setxor1d
np.alltrue                    np.exp2                       np.mat                        np.shape
np.alterdot                   np.expand_dims                np.math                       np.short
np.amax                       np.expm1                      np.matrix                     np.show_config
np.amin                       np.extract                    np.matrixlib                  np.sign
np.angle                      np.eye                        np.max                        np.signbit
np.any                        np.fabs                       np.maximum                    np.signedinteger
np.append                     np.fastCopyAndTranspose       np.maximum_sctype             np.sin
np.apply_along_axis           np.fft                        np.may_share_memory           np.sinc
np.apply_over_axes            np.fill_diagonal              np.mean                       np.single
np.arange                     np.find_common_type           np.median                     np.singlecomplex
np.arccos                     np.finfo                      np.memmap                     np.sinh
np.arccosh                    np.fix                        np.meshgrid                   np.size
np.arcsin                     np.flatiter                   np.mgrid                      np.sometrue
np.arcsinh                    np.flatnonzero                np.min                        np.sort
np.arctan                     np.flexible                   np.min_scalar_type            np.sort_complex
np.arctan2                    np.fliplr                     np.minimum                    np.source
np.arctanh                    np.flipud                     np.mintypecode                np.spacing
np.argmax                     np.float                      np.mirr                       np.split
np.argmin                     np.float128                   np.mod                        np.sqrt
np.argpartition               np.float16                    np.modf                       np.square
np.argsort                    np.float32                    np.msort                      np.squeeze
np.argwhere                   np.float64                    np.multiply                   np.stack
np.around                     np.float_                     np.nan                        np.std
np.array                      np.floating                   np.nan_to_num                 np.str
np.array2string               np.floor                      np.nanargmax                  np.str0
np.array_equal                np.floor_divide               np.nanargmin                  np.str_
np.array_equiv                np.fmax                       np.nanmax                     np.string_
np.array_repr                 np.fmin                       np.nanmean                    np.subtract
np.array_split                np.fmod                       np.nanmedian                  np.sum
np.array_str                  np.format_parser              np.nanmin                     np.swapaxes
np.asanyarray                 np.frexp                      np.nanpercentile              np.sys
np.asarray                    np.frombuffer                 np.nanprod                    np.take
np.asarray_chkfinite          np.fromfile                   np.nanstd                     np.tan
np.ascontiguousarray          np.fromfunction               np.nansum                     np.tanh
np.asfarray                   np.fromiter                   np.nanvar                     np.tensordot
np.asfortranarray             np.frompyfunc                 np.nbytes                     np.test
np.asmatrix                   np.fromregex                  np.ndarray                    np.testing
np.asscalar                   np.fromstring                 np.ndenumerate                np.tile
np.atleast_1d                 np.full                       np.ndfromtxt                  np.timedelta64
np.atleast_2d                 np.full_like                  np.ndim                       np.trace
np.atleast_3d                 np.fv                         np.ndindex                    np.transpose
np.average                    np.generic                    np.nditer                     np.trapz
np.bartlett                   np.genfromtxt                 np.negative                   np.tri
np.base_repr                  np.get_array_wrap             np.nested_iters               np.tril
np.bench                      np.get_include                np.newaxis                    np.tril_indices
np.binary_repr                np.get_printoptions           np.nextafter                  np.tril_indices_from
np.bincount                   np.getbufsize                 np.nonzero                    np.trim_zeros
np.bitwise_and                np.geterr                     np.not_equal                  np.triu
np.bitwise_not                np.geterrcall                 np.nper                       np.triu_indices
np.bitwise_or                 np.geterrobj                  np.npv                        np.triu_indices_from
np.bitwise_xor                np.gradient                   np.numarray                   np.true_divide
np.blackman                   np.greater                    np.number                     np.trunc
np.bmat                       np.greater_equal              np.obj2sctype                 np.typeDict
np.bool                       np.half                       np.object                     np.typeNA
np.bool8                      np.hamming                    np.object0                    np.typecodes
np.bool_                      np.hanning                    np.object_                    np.typename
np.broadcast                  np.histogram                  np.ogrid                      np.ubyte
np.broadcast_arrays           np.histogram2d                np.oldnumeric                 np.ufunc
np.broadcast_to               np.histogramdd                np.ones                       np.uint
np.busday_count               np.hsplit                     np.ones_like                  np.uint0
np.busday_offset              np.hstack                     np.outer                      np.uint16
np.busdaycalendar             np.hypot                      np.packbits                   np.uint32
np.byte                       np.i0                         np.pad                        np.uint64
np.byte_bounds                np.identity                   np.partition                  np.uint8
np.bytes0                     np.iinfo                      np.percentile                 np.uintc
np.bytes_                     np.imag                       np.pi                         np.uintp
np.c_                         np.in1d                       np.piecewise                  np.ulonglong
np.can_cast                   np.index_exp                  np.pkgload                    np.unicode
np.cast                       np.indices                    np.place                      np.unicode_
np.cbrt                       np.inexact                    np.pmt                        np.union1d
np.cdouble                    np.inf                        np.poly                       np.unique
np.ceil                       np.info                       np.poly1d                     np.unpackbits
np.cfloat                     np.infty                      np.polyadd                    np.unravel_index
np.char                       np.inner                      np.polyder                    np.unsignedinteger
np.character                  np.insert                     np.polydiv                    np.unwrap
np.chararray                  np.int                        np.polyfit                    np.ushort
np.choose                     np.int0                       np.polyint                    np.vander
np.clip                       np.int16                      np.polymul                    np.var
np.clongdouble                np.int32                      np.polynomial                 np.vdot
np.clongfloat                 np.int64                      np.polysub                    np.vectorize
np.column_stack               np.int8                       np.polyval                    np.version
np.common_type                np.int_                       np.power                      np.void
np.compare_chararrays         np.int_asbuffer               np.ppmt                       np.void0
np.compat                     np.intc                       np.print_function             np.vsplit
np.complex                    np.integer                    np.prod                       np.vstack
np.complex128                 np.interp                     np.product                    np.warnings
np.complex256                 np.intersect1d                np.promote_types              np.where
np.complex64                  np.intp                       np.ptp                        np.who
np.complex_                   np.invert                     np.put                        np.zeros
np.complexfloating            np.ipmt                       np.putmask                    np.zeros_like
```

## Operating with arrays

Standard mathematical operations Just Work (TM):


```python
arr1 = np.arange(4)
arr2 = np.arange(10, 14)
print(arr1, '+', arr2, '=', arr1 + arr2)
```

Note, that, unlike in MATLAB, operations are performed element-wise:


```python
print(arr1, '*', arr2, '=', arr1 * arr2)
```

While this means that, in principle, arrays must always match in their dimensionality in order for an operation to be valid, numpy will *broadcast* dimensions when possible.  For example, suppose that you want to add the number 1.5 to `arr1`, this works:


```python
arr1 + 3
```

<img src="files/broadcast_1D.png"/>

### The broadcasting rules

This broadcasting behavior is powerful, especially because when numpy broadcasts to create new dimensions or to 'stretch' existing ones, it doesn't replicate the data.  In the example above the operation is carried *as if* the 3 was a 1-d array with 3 in all of its entries, but no actual array was ever created.  This can save memory in cases when the arrays in question are large, with significant performance implications.

The general rule is: when operating on two arrays, NumPy compares their shapes element-wise. It starts with the trailing dimensions, and works its way forward, creating dimensions of length 1 as needed. Two dimensions are considered compatible when

* they are equal or either is None or one
* either dimension is 1 or ``None``, or if dimensions are equal

If these conditions are not met, a `ValueError: frames are not aligned` exception is thrown, indicating that the arrays have incompatible shapes. The size of the resulting array is the maximum size along each dimension of the input arrays.

Examples below:

```
(9, 5)   (9, 5)   (9, 5)   (9, 1)
   ( )   (9, 1)   (   5)   (   5)
------   ------   ------   ------
(9, 5)   (9, 5)   (9, 5)   (9, 5)

```

<img src="files/broadcast_rougier.png"/>

Sketch from [Nicolas Rougier's NumPy tutorial](http://www.labri.fr/perso/nrougier/teaching/numpy/numpy.html)

### Visual illustration of broadcasting
<center>
<img src="files/numpy_broadcasting.svg" width=80%>
</center>

Sometimes, it is necessary to modify arrays before they can be used together.  Numpy allows you to add new axes to an array by indexing with `np.newaxis`:


```python
c = np.arange(5)
d = np.arange(6)

print(c.shape)
print(d.shape)

c + d
```


```python
c = np.arange(5)
d = np.arange(6)

c = c[:, np.newaxis]

print(c.shape)
print('  ', d.shape)
print('-------')
print((c + d).shape)
 
#   d d d d d d
#
# c              c c c c c c   d d d d d d
# c              c c c c c c   d d d d d d
# c     +      = c c c c c c + d d d d d d
# c              c c c c c c   d d d d d d
# c              c c c c c c   d d d d d d

c + d
```

For the full broadcasting rules, please see the official Numpy docs, which describe them in detail and with more complex examples.

Also see: [G-Node Summer School Advanced NumPy tutorial](https://github.com/stefanv/teaching/blob/master/2014_assp_split_numpy/numpy_advanced.ipynb)

As we mentioned before, Numpy ships with a full complement of mathematical functions that work on entire arrays, including logarithms, exponentials, trigonometric and hyperbolic trigonometric functions, etc.  Furthermore, scipy ships a rich special function library in the `scipy.special` module that includes Bessel, Airy, Fresnel, Laguerre and other classical special functions.  For example, sampling the sine function at 100 points between $0$ and $2\pi$ is as simple as:


```python
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
```

## Linear algebra in numpy

Numpy ships with a basic linear algebra library, and all arrays have a `dot` method whose behavior is that of the scalar dot product when its arguments are vectors (one-dimensional arrays) and the traditional matrix multiplication when one or both of its arguments are two-dimensional arrays:


```python
v1 = np.array([2, 3, 4])
v2 = np.array([1, 0, 1])
print(v1, '·', v2, '=', v1.dot(v2))
```


```python
A = np.arange(6).reshape(2, 3)
print('A:\n', A)
print('v1:\n', v1)
```


```python
A.dot(v1)
```

For matrix-matrix multiplication, the same dimension-matching rules must be satisfied, e.g. consider the difference between $A \times A^T$:


```python
print(A.dot(A.T))
```

and $A^T \times A$:


```python
print(A.T.dot(A))
```

In Python 3.5, we'll be able to write this as:

```
A.T @ A
```

Furthermore, the `numpy.linalg` module includes additional functionality such as determinants, matrix norms, Cholesky, eigenvalue and singular value decompositions, etc.  For even more linear algebra tools, `scipy.linalg` contains the majority of the tools in the classic LAPACK libraries as well as functions to operate on sparse matrices.  We refer the reader to the Numpy and Scipy documentations for additional details on these.

## Reading and writing arrays to disk

Numpy lets you read and write arrays into files in a number of ways.  In order to use these tools well, it is critical to understand the difference between a *text* and a *binary* file containing numerical data.  In a text file, the number $\pi$ could be written as "3.141592653589793", for example: a string of digits that a human can read, with in this case 15 decimal digits.  In contrast, that same number written to a binary file would be encoded as 8 characters (bytes) that are not readable by a human but which contain the exact same data that the variable `pi` had in the computer's memory.  

The tradeoffs between the two modes are thus:

* Text mode: occupies more space, precision can be lost (if not all digits are written to disk), but is readable and editable by hand with a text editor.  Can *only* be used for one- and two-dimensional arrays.

* Binary mode: compact and exact representation of the data in memory, can't be read or edited by hand.  Arrays of any size and dimensionality can be saved and read without loss of information.

First, let's see how to read and write arrays in text mode.  The `np.savetxt` function saves an array to a text file, with options to control the precision, separators and even adding a header:


```python
arr = np.arange(10).reshape(2, 5)
print(arr)                            
np.savetxt('test.out', arr)
```


```python
!cat test.out
```

And this same type of file can then be read with the matching `np.loadtxt` function:


```python
arr2 = np.loadtxt('test.out')
print(arr2)
```

For binary data, Numpy provides the `np.save` and `np.savez` routines.  The first saves a single array to a file with `.npy` extension, while the latter can be used to save a *group* of arrays into a single file with `.npz` extension.  The files created with these routines can then be read with the `np.load` function.

Let us first see how to use the simpler `np.save` function to save a single array:


```python
np.save('test.npy', arr)
# Now we read this back
arr_loaded = np.load('test.npy')

print(arr)
print(arr_loaded)

print(arr_loaded.dtype)

# Let's see if any element is non-zero in the difference.
# A value of True would be a problem.
print('Any differences?', np.any(arr - arr_loaded))
```

Now let us see how the `np.savez_compressed` function works.


```python
np.savez_compressed('test.npz', first=arr, second=arr2)
arrays = np.load('test.npz')
arrays.files
```

The object returned by `np.load` from an `.npz` file works like a dictionary:


```python
arrays['first']
```

This `.npz` format is a very convenient way to package compactly and without loss of information, into a single file, a group of related arrays that pertain to a specific problem.  At some point, however, the complexity of your dataset may be such that the optimal approach is to use one of the standard formats in scientific data processing that have been designed to handle complex datasets, such as NetCDF or HDF5.  

Fortunately, there are tools for manipulating these formats in Python, and for storing data in other ways such as databases.  A complete discussion of the possibilities is beyond the scope of this tutorial, but of particular interest for scientific users we at least mention the following:

* The `scipy.io` module contains routines to read and write Matlab files in `.mat` format and files in the NetCDF format that is widely used in certain scientific disciplines.

* For manipulating files in the HDF5 format, there are two excellent options in Python: the **PyTables** project offers a high-level, object oriented approach to manipulating HDF5 datasets, while the **h5py** project offers a more direct mapping to the standard HDF5 library interface.  Both are excellent tools; if you need to work with HDF5 datasets you should read some of their documentation and examples and decide which approach is a better match for your needs.


```python
arrays['second']
```

## Advanced Language Strings 


```python
'funky town'.capitalize()
```


```python
'funky town'.capitalize().split()
```


```python
[x.capitalize() for x in 'funky town'.split()]
```


```python
'I want to take you to funky town'.split('to')
```

#### .strip(), .join() and .replace()


```python
csv_string = 'cat, dog, spam, mouse, 3.1432'
```


```python
csv_string.strip('\t')
```


```python
csv_clean = [x.strip() for x in csv_string.split(',')]
```


```python
csv_clean
```


```python
csv_string = "cat, dog, spam, mouse,, 3.1432"
```


```python
csv_string.replace(",,", ",")
```




    'cat, dog, spam, mouse, 3.1432'



### String formatting 


```python
'On {0}, I feel {1}'.format('Saturday', 'excited')
```




    'On Saturday, I feel excited'




```python
'{desire} to {place}'.format(desire = 'Fly me', place = 'the moon')
```




    'Fly me to the moon'




```python
f = {'desire': 'Fly me', 'place': 'the moon'}
```


```python
'{desire} to {place}'.format(**f)
```




    'Fly me to the moon'



### Formating comes after a colon :


```python
'{0:00.2f}'.format(3.14159,42)
```




    '3.14'



# File I/O (read/write) 

.open() and .close() are built in functions

open modes: r (read), w (write), r+ (read + update), rb (read as binary stream, ...), rt (read as text file)

Writing data: .write() or .writelines()


```python
%%file mydata.dat
This is my zeroth file I/O. Zing!
```

    Writing mydata.dat



```python
file_stream = open('mydata.data', 'r')
print(type(file_stream))
file_stream.close()
```


```python
f = open('test.dat', 'w')
f.write('This is mt first I/O file. Zing! again.')
f.close()

```


```python
! cat mydata.dat

! cat test.dat
```

    This is my zeroth file I/O. Zing!This is mt first I/O file. Zing! again.


```python
f = open('test.dat', 'w')
f.writelines([ "a = ['This is mt first I/O file.']\n", 'Zing! again.'])
f.close()
print(type('test.dat'))
! cat test.dat
```

    <class 'str'>
    a = ['This is mt first I/O file.']
    Zing! again.


```python
f = open('test.dat', 'r')
data = f.readlines()
f.close()
data
type(data)
```




    list



# Lambda functions

Anonymous functions from Lisp and functional programming


```python
import math
tmp = lambda x: x**2
print(type(tmp))
```

    <class 'function'>



```python
tmp(4)
```




    16




```python
# forget about creating a function name... just do it

(lambda x, y: x**2 + y)(2, 6)
```




    10




```python
#Creating a list of lambda functions 
lamfun = [lambda x: x**2, lambda x: x**3, lambda x: x**4, \
             lambda y: math.sqrt(y) if y >= 0 else 'Really?' '{0:00.1f}'.format(y)]
```


```python
for l in lamfun: print(l(-2))
```

    4
    -8
    16
    Really?-2.0

# python_deliberate_practice
