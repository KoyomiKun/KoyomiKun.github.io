# Fluent Python
## Chapter1 List

### 1.1 生成器表达式
生成器表达式更适合用来创建序列类型，因为其背后遵守了迭代器协议，可以逐个产生元素，更加节省内存

```python
# 生成tuple
symbols = '(*^%$#'
t = tuple(ord(symbol) for symbol in symbols)
# 计算笛卡尔积
colors = ['black', 'white']

sizes = ['S', 'M', 'L']

for tshirt in ("{} {}".format(c, s) for c in colors for s in sizes):
    print(tshirt)
```

### 1.2 元组拆包
unpack a tuple as multiple args

```python
t = (20,8)
divmod(*t) # same to `divmod(20,8)`
```

平行赋值

```python
a, b, *rest = range(5)
# [0, 1, [2, 3, 4, ]]
```

### 1.3 序列增量赋值(+= \*=)

#### 1.3.1 +=
```python
a+=b
```

+= 可能导致两种结果：
1. a实现了__iadd__方法; 调用此方法。对于可变序列来说a会就地改动，在a本身的内存上增加一个b，相当于`append(b)` 
2. a未实现__iadd__方法; 调用__add__方法。a会和b相加，得到新的对象，赋值给a。

#### 1.3.2 \*=
和`+=` 一样，\*=也是两种结果，区别在于调用的是__imul__和__mul__方法。

```python
l = [1, 2, 3, ]
id(l) # 140533065687936
l *= 2
id(l) # 140533065687936 一样，因为可变对象有__imul__方法

--------

l = (1, 2, 3, )
id(l) # 140533065380992
l *= 2
id(l) # 140533065342016 不一样，因为不可变对象没有__imul__方法

```

***一个有趣的问题*** :
```python
t = (1, 2, [30, 40, ])
t[2] += [50, 60, ]
```

上述代码的执行结果是什么：

A. t变成(1, 2, [30, 40, 50, 60, ])

B. tuple 不支持非元素赋值，抛出`TypeError` 异常

C. A B 都不对

D. A B 都对

`Answer: D` 

[python tutor](http://pythontutor.com/visualize.html#mode=edit) 的结果显示，不仅t的值改变了， 同时还抛出错误

> 原因是***增量赋值不是一个原子操作*** 
> 
> 先完成内部列表的相加操作，再将加好的内容赋值给元组
> 
> 第一步能够成功，第二步却会失败

所以***最好不要把可变对象放在元组里面***


### 1.4 bisect 管理有序序列

#### 1.4.1 bisect.bisect

`bisect(haystack, needle)` : find a needle in a haystack

position must satisfy that if needle insert into haystack[index], haystack keeps in ascending order 

Ex: `bisect([1, 3, 4, 6, ], 2)` returns `1` 

***Options:*** 

1. `bisect(haystack, needle, lo=1, hi=2 )`  
means search in `range(lo, hi)` 
2. `bisect_left(haystack, needle)`  
compared to `bisect` and `bisect_right` , `bisect_left` returns the index that `haystack[:index+1]` < needle rather than `haystack[:index+1]` <= needle 


#### 1.4.2 bisect.insort

`insort(haystack, needle)` : find the place and insert into it

*Equals to* 
```python
index = bisect(haystack, needle)
haystack.insert(needle, index)
```
but faster than two steps
#### Usages:
Bisect module suits for GRADING:

```python
def grading(score, break_points=[60, 80, 90, ], grades='FCBA'):
    index = bisect.bisect(break_points, score)
    return grades[index]
print(grading(100)) # A
print(grading(60)) # C
print(grading(40)) # F
```

### 1.5 Array

If we need *a list only consists of numbers*, `array.array`  is more efficient than `list`

Advantages:
1. cost less time and less memory 

2. `array.tofile` and `array.fromfile` 

***Tips*** :From python 3.4, `array.sort()` are not supported. For sorting an array, we should say `a = array(typecode, sorted(a))` 

#### 1.5.1 MemoryView

As the name implies, MemoryView module provides views of memory, which means one piece of memory can be considered as float, int, list, char, etc. 

```python
from array import array

nums = array('h', [-2, -1, 0, 1, 2, ])

memv = memoryview(nums)
print((len(memv), memv[0]))  # (5, -2)

memv_oct = memv.cast('B')
print(memv_oct.tolist())  # [254, 255, 255, 255, 0, 0, 1, 0, 2, 0]
```

'h' means `short` , 'B' means `unsigned short` 

## Chapter2 Dict and Set

### 2.1 Mapping

#### 2.1.1 Hashable

1. implement `__hash__()` method and `__qe__()` method

2. atom types `str, bytes, num` are hashable, as well as frozenset. As for `tuple`, iff elements in tuple are all hashable, this tuple is hashable.  
```python
tt = (1, 2, (3, 4, ))
print(hash(tt))  # 3794340727080330424

tt = (1, 2, [3, 4, ])
print(hash(tt))  # TypeError: unhashable type: 'list'
```

***ALL UNCHANGEABLE TYPES ARE HASHABLE***   
except something containing changeable elements

### 2.2 Dict Comprehension

To create a dict:

```python
a = dict(one=1, two=2, three=3)
b = {'one':1, 'two':2, 'three':3}
c = dict(zip(['one','two','three'], [1, 2, 3, ]))
d = dict([('two', 2), ('one', 1), ('three', 3)])
e = dict({'three':3, 'one':1, 'two':2})

# dict comprehension
lists = [('one', 1), ('two', 2), ('three', 3)]
f = {en: num for en, num in lists} # {'one': 1, 'two': 2, 'three': 3}
```

### 2.3 Useful Mapping Functions

1. `d.popitem` : pop the first insert element, with arg `last=True` , pop the last insert element.

2. `d.update(m, [**kargs])` : if m has `keys` method, treat m as mapper; else as iterator (key, value)

3. `d.get(k, default)` : replace `d[k]` , set a default value if can't find k.
 
4. `d.setdefault(k, []).append(new_value)` : same as `if key not in d: d[key] = [] ; d[key].append(new_value)` 

### 2.4 Mappings with Flexible Key Lookup

#### 2.4.1 Defaultdict

`d = defaultdict(list)` return an empty list when `__getitem__` is invoked, including `d[k]` , but excluding `d.get(k)` , instead it returns `None` 

| list | int | float | MyClass                                     | 
|:-----|:----|:------|:--------------------------------------------|
| []   | 0   | 0.0   | <__main__.MyClass object at 0x7f12b7f25430> | 

#### 2.4.2 __missing\_\_

If a class extends `dict` and provide `__missing__` method, when key is not found, this method will be called.
y

`__missing__` example: 

```python
def __missing__(self, key):
	if isinstance(key, str):
		raise KeyError(key)
	return self[str(key)]
```

***Tips***: `d.keys()` returns a view, which is responsed quickly on query.

### 2.5 Other Dicts

#### 2.5.1 OrderedDict

Keep the dict in insertion time order and pop the last element.

#### 2.5.2 ChainMap

A chain of mappers, look up all mappers inside it.

for example, in python official varible lookup progress:

```python
import builtins
pylookup = ChainMap(locals(), globals(), vars(builtins))
```

#### 2.5.3 Counter

count the elements in hashable objects 

```python
from collections import Counter
ct = Counter('abcdabcdabca')
print(ct)               # Counter({'a': 4, 'b': 3, 'c': 3, 'd': 2})
ct.most_common(2)       # [('a', 4), ('b', 3)]
ct + Counter('ddaactg') # Counter({'a': 6, 'b': 3, 'c': 4, 'd': 4, 't': 1, 'g': 1})
```

### 2.6 UserDict

In order to create a new dict class, it is a better choice to extends `UserDict` rather than simple `dict` .

Indeed, `data` attribute of `UserDict` is subclass of `dict` .

For examle:
```python
from collections import UserDict


class StrKeydict(UserDict):

    """User Dict with string key"""

    def __init__(self):
        UserDict.__init__(self)

    def __missing__(self, key):
        if isinstance(key, str):
            raise KeyError(key)
        return self[str(key)]

    def __contains__(self, key):
        return str(key) in self.data

    def __setitem__(self, key, item):
        self.data[str(key)] = item

```

### 2.7 Unchangable Mapper

`MappingProxyType` is a dynamic view of mapper:

```python
# Create a MappingProxyType
d = {1:'A'}

d_proxy = MappingProxyType(d)
d_proxy[2] = 'X' # TypeError: 'mappingproxy' object does not support item assignment

d[2] = 'R'
d_proxy # {1: 'A', 2: 'R'} 
```
### 2.8 Set

elements in set must be *hashable* , because to ensure uniqueness , python use `__hash__` and `__eq__` to delete duplicated elements.

***Tips*** : `Set` is not hashable, but `frozenset` is.

#### 2.8.1 Create a set

```python
# Empty set
s = set() # s = {} create a dictory
# Set literals
s = {1} # faster than s = set([1])
# set comprehension
s = {chr(i) for i in range(32, 256) if 'SIGN' in name(chr(i), '')} # {'=', '$', '£', '¥', '<', '°', '®', '§', '÷', 'µ', '¤', '×', '>', '#', '+', '¢', '©', '¶', '¬', '±', '%'}
```

#### 2.8.2 Set Operations

```python
s1 = {1, 2, 4, 6, 7, }
s2 = {4, 5, 7, 3, 6, }

print(s1 & s2) # {4, 6, 7}
print(s1 | s2) # {1, 2, 3, 4, 5, 6, 7}
print(s1 - s2) # {1, 2}
```

####  2.8.3 Hash Table

Hash table is a sparse array, whose cells called `Bucket`. A `bucket` contains 2 fields: reference to keys and reference to values.
 
In order to keep 1/3 spaces empty, python will *copy original array* into a new location having bigger space.

***Hash*** :

When we tend to insert an element into a hash table, the first step is to calculate the *hash value* with `hash()` method.

> `==` means 'hash equal', `hash(1.0) == hash(1)` is true, but it doesn't means `1` and `1.0` have the the data structure.

> After python 3.3, the calculation of hash values of `str`, `bytes`, and `datetime` add salting progress. `Salt` is a constant value formed randomly at the start of python thread.

***Progress of retrieving item***:
![progress of retrieving item](/home/komikun/Pictures/screenshoots/2020-08-01_19-59-11_screenshot.png) 


***Tips*** :

1. According to `CPython` , if an int object's value can be arraged in a machine `word` , its hash value equals to its value

2. When facing *Hash collision* , there is an special alogrithm to handle it

Keys: `Quadratic Probing` , `perturb` 

[Collision Handling Method](https://www.kawabangga.com/posts/2493) 

Keys: `Why use perturb` , `Hash collision attack` 

[Hash Collision Attack](https://blog.zhenlanghuo.top/2017/05/17/%E5%93%88%E5%B8%8C%E7%A2%B0%E6%92%9E%E6%94%BB%E5%87%BB/) 
```python
def lookup(d, key):
    perturb = j = hash(key)
    while True:
        cell = d.data[j % d.size]
        if cell.key is EMPTY:
            raise IndexError
        if cell.key is not DELETED and (cell.key is key or cell.key == key):
            return cell.value
        j = (5 * j) + 1 + perturb
        perturb >>= PERTURB
```

## Chapter3 Text versus Bytes

### 3.1 Characters

#### 3.1.1 Unicode

1. Indentified by `code point` , ranging between 0 and 1114111.

2. Start with `U+` and following 4-6 hexadecimal digits, Ex:'G'=>U+1D11E 


3. Actual Bytes depends on the *encoding* method, such as `utf-8` .

```python
s = 'lang£'
print(len(s)) # 5

b = s.encode('utf8')
print(b)      # b'lang\xc2\xa3'
print(len(b)) # 6 

print(b.decode('utf8'))  # lang£
```

### 3.2 Bytes

changable `bytes` unchangable `bytearray` 

1. Elements in range(0, 255)

2. slice of bytes is also bytes, although contains only one element; But the single element is a int in (0, 255)

```python
s = 'lang£'
b = bytes(s, encoding='utf_8')
print(b[0])  # 108
print(b[:1]) # b'l'
```

***Create Bytes*** :

```python
s = 'lang£'
b = bytes(s, encoding='utf_8')
print(b)  # b'lang\xc2\xa3'
b = bytes.fromhex('31 4B CE A9')
print(b)  # b'1K\xce\xa9'
li = [1, 3, 9, 90, 128, 255]
b = bytes(li)
print(b)  # b'\x01\x03\tZ\x80\xff'
b = bytes(array('h', [-2, -1, 0, 1, 2]))
print(b)  # b'\xfe\xff\xff\xff\x00\x00\x01\x00\x02\x00'
```

## Chapter4 Functions

### 4.1 Function is Object

```python
def factorial(n):
    """return n!

    :n: factor
    :returns: n!

    """
    return 1 if n < 2 else n * factorial(n-1)


print(factorial.__doc__)
return n!

    :n: factor
    :returns: n!
```

### 4.2 Higher-order Function 

the function that return a function OR accept a function as parameter is called `higher-order function` 

#### 4.2.1 Map, Filter and Reduce

`list(map(fact, range(6)))` == `[fact(n) for n in range(6)]` 

`list(map(factorial, filter(lambda n:n % 2, range(6))))` == `[factorial(n) for n in range(6) if n % 2]` 

reduce are always used to sums:

```python
from functools import reduce
from operator import add

print(reduce(add, range(100))) # sum(range(100))
```

### 4.3 Lambda Expression

1. We ***Can't*** assign value in lambda or use `while` , `try` , etc.

2. In common, lambda expression used to be parameters in functions.

### 4.4 Callable Objects

There are 7 types of callable objects in python:

1. `def` or `lambda` created by pythoner 

2. built-in functions implemented with C(`len` , `time.strftime` , etc.)

3. built-in method such as `dict.get` 

4. method defined in class.

5. class.

6. objects implementing `__call__` 

7. generator fuction created by `yield` 

```python
print([callable(obj) for obj in (abs, str, 13)])
# [True, True, False]
```

### 4.5 User-defined Callable Type

*ANY* python objects can be callable iff implement `__call__` 

```python
import random


class BingoCase(object):
    """random list"""

    def __init__(self, items):
        self._items = list(items)
        random.shuffle(self._items)

    def pick(self):
        try:
            return self._items.pop()
        except IndexError:
            raise LookupError('pick from empty BingoCase')

    def __call__(self):
        return self.pick()


bingo = BingoCase(range(3))

print(bingo.pick()) # 0
print(bingo()) # 2
```

The difference of method between class and function:

```python
class C: pass
obj = C()

def func(): pass
print(sorted(set(dir(func)) - set(dir(C))))
#  ['__annotations__', '__call__', '__closure__', '__code__', '__defaults__',
#  '__get__', '__globals__', '__kwdefaults__', '__name__','__qualname__']
```
![attritubes](/home/komikun/Pictures/screenshoots/2020-08-03_11-04-12_screenshot.png)

 
### 4.6 Keyword-only Argument
 
***This function are used to create HTML tags*** 

```python
def tags(name, *content, cls=None, **attrs):
    if cls:
        attrs['class'] = cls
    if attrs:
        attr_str = ''.join(' {}={}'.format(attr, value)
                           for attr, value in attrs.items())
    else:
        attr_str = ''
    if content:
        return '\n'.join('<{} {}>{}</{}>'.format(name, attr_str, c, name) for c
                         in content)
    else:
        return '<{} {} />'.format(name, attr_str)


print(tags('img', 'i dont know', align='right',  cls='/plain/img'))
# <img  align=right class=/plain/img>i dont know</img>
```

### 4.7 Get Parameters' Information

Function objects have an tuple attribute called `__defaults__` to save default value of `positional parameters` and `keyword parameters` , default value of `keyword` is in `__kwdefaults__` , and the name of parameters  is in `__code__` :

```python
def test(name, ass='name', *awg, **kwag):
    pass

print(test.__defaults__)# ('name',)
print(test.__kwdefaults__) # None
print(test.__code__.co_argcount)# 2
print(test.__code__.co_varnames) # ('name', 'ass', 'awg', 'kwag')
```

***Tips*** :

The order of parameters: 1. positional only 2. keyword or positional 3. variable positional 4. keyword only 5. variable keyword 

python *DOES NOT* support 1, but some function implement with C support this(ex:`divmod`).

EX:`def test(a, b=3 , *arg, d, e=5, f, **kargs)` 
> a, b: 2  
\*arg: 3  
d, e, f: 4  
\*\*kargs: 5 

#### 4.7.1 Inspect Module 

`inspect.signature` returns a `inspect.Signature` object with an attribute `parameters` . That is an ordered mapper, mapping name of paras with `inspect.Parameter` .

```python
import inspect
def test(name, ass='name', *awg, **kwag):
    pass
sig = inspect.signature(test)
print(sig) # (name, ass='name', *awg, **kwag)
print(sig.parameters)
# OrderedDict([('name', <Parameter "name">), ('ass', <Parameter "ass='name'">), ('awg', <Parameter "*awg">), ('kwag', <Parameter "**kwag">)])
```
attr `Parameter` alse has its own attrs:`name` , `default` , and `kind` .



### 4.8 Functional Programming

#### 4.8.1 Operator Module

`operator` module provide several functions to santisfy arithmetic operations.

```python
reduce(operator.mul, range(1, n+1))
# reduce(lambda a, b: a*b, range(1, n+1))
sorted(data, key=itemgetter(1))
# sorted(data, key=lambda fields: fields[1])
# itemgetter uses [], so it support any class implement __getitem__
# correspond with itemgetter, there exists attrgetter.
```

#### 4.8.2 Functools Module

As we all know, `functools` module provide `reduce` function to handle elements one by one. In addition, `partial` and `partialmethod` also offer us useful funcs.

***Ex1*** :
```python
from operator import mul
from functools import partial

triple = partial(mul, 3)
print(triple(7)) # 21

print(list(map(triple, range(1, 10)))) # [3, 6, 9, 12, 15, 18, 21, 24, 27]
```

`partial`'s first arg is a callable object, following several binding positional args and keyword args. 

***Ex2*** :
```python
import unicodedata
import functools

nfc = functools.partial(unicodedata.normalize, 'NFC')
s1 = 'cafe\u0301'
s2 = 'café'

print(s1 == s2) # False
print(nfc(s1) == nfc(s2)) # True
```

## Chapter5 Closure and Decorators

### 5.1 Basic Decorators

Decorator *replace* function with another function:

```python
def deco(func):
    def inner():
        print('running inner()')
    return inner


@deco
def target():
    print('running target()')


target() # running inner()
# equals to :
# def target():
#	print('running target()')
# target = decorate(target) 	

```

***Tips:*** 
Python execute `decorator` right after the difinition, exactly, importing modules.


### 5.2 Closure 

```python
class Averager():
    """Averager Caculator"""

    def __init__(self):
        self.sum = 0
        self.count = 0

    def __call__(self, new_value):
        self.sum += new_value
        self.count += 1
        return self.sum/self.count


avg = Averager()
print(avg(1)) # 1.0
print(avg(2)) # 1.5
print(avg(3)) # 2.0
```

Another version:

```python
def make_averager():
    count = 0
    total = 0

    def averager(new_value):
        nonlocal count, total
        count += 1
        total += new_value
        return total / count
    return averager


avg = make_averager()
print(avg(1))                           # 1.0
print(avg(2))                           # 1.5
print(avg(3))                           # 2.0
print(avg.__code__.co_varnames)         # ('new_value',)
print(avg.__code__.co_freevars)         # ('count', 'total')
print(avg.__closure__[0].cell_contents) # 3
```

![free variable](/home/komikun/Pictures/screenshoots/2020-08-05_21-46-47_screenshot.png) 

`nonlocal` explicit express those variates are free-variates.

***HOW TO USE DECORATORS*** :

```python
import random
import time
from functools import wraps


def clock(func):
    @wraps(func)
    def clocked(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args)
        elapsed = time.perf_counter() - t0

        name = func.__name__
        arg_lst = []
        if args:
            arg_lst.append(', '.join(repr(arg) for arg in args))
        if kwargs:
            pairs = ['{}={}'.format(k, w) for k, w in sorted(kwargs.items())]
            arg_lst.append(', '.join(pairs))
        arg_str = ', '.join(arg_lst)

        print('[{:8f}] {}({}) -> {}'.format(elapsed, name, arg_str, result))
        return result
    return clocked


@clock
def test(li: list):
    for i in li:
        pass
    return len(li)


test([random.randint(4, 10) for _ in range(5)])
```

`wraps` copy `__name__` , `__doc__` , etc. from `func` to `clocked` .

### 5.3 Decorators in Standard Lib 

#### 5.3.1 Lru_cache

`functools.lru_cache` realize the *memoization* function by caching the return result.

```python
@clock
@lru_cache()
def test(n: int):
    if n < 2:
        return n
    return test(n-1) + test(n-2)


test(30)
# [0.000720s] test(30) -> 832040
# [71.819455s] test(30) -> 832040
```

`lru_cache` tips:

1. `lru_cache` has two parameters `maxsize` and `typed` , the prior one define the max number of elements it caching, and the later one define whether distinguish 1 and 1.0 (int and float type) or not.

2. `maxsize` had better to be 2^n.

3. `lru_cache` use dict to save the result, parameters as key and return result as value. So all parameters *MUST BE* hashable.

#### 5.3.2 Singledispatch

As the name says, to understand this method , we should divide `singledispatch` into `single` and `dispatch` .

***Dispatch*** :

`Dispatch` means this decorator helps us dispatch different missions to different kinds of parameters.

```python
from functools import singledispatch
from collections import abc
import numbers
import html


@singledispatch
def htmlize(obj):
    content = html.escape(repr(obj))
    return '<pre>{}</pre>'.format(content)


@htmlize.register(str)
def _(text):
    content = html.escape(text).replace('\n', '<br>\n')
    return '<p>{0}</p>'.format(content)


@htmlize.register(numbers.Integral)
def _(n):
    return '<pre>{0}(0x{0:x})</pre>'.format(n)


@htmlize.register(tuple)
@htmlize.register(abc.MutableSequence)
def _(seq):
    inner = '<\li>\n<li>'.join(htmlize(item) for item in seq)
    return '<ul>\n<li>' + inner + '</li>\n</ul>'
```

In common, we use `<<base_function>>.register(<<type>>)` to decorate generic functions.`base function` should be abstract class such as `numbers.Integral` and `abc.MutableSequence` , rather than `int` or `list` .

### 5.4 Multiple Decorators

```python
@d1
@d2
def f():print('f')

# equals to:
def f(): print('f')
f = d1(d2(f))
```




