## Heuristics

### How to build a model out of Heuristics

```python

@dataclass
class AluminumCan(ClassHeuristic):
  brightness: float 
  colour_ranges: 
```

```
@dataclass
class AluminumCan(CompositeHeuristic):
  
  Brightness(): 
```

```python
aluminum_can = Heuristic([
  ShapeFilter(
      min_width=0,
      max_width=100,
      min_height=0,
      max_height=100,
  )
  Brightness(min='', max='') | Edginess(min='', max=''),
  Cornerness(min='', max='')
])
```

2 design options: 
* Any list in a list, could be treated as an OR (I think this works?).
* Or use the pipe for or, ideally don't want to have to use anything for &, because 
that should be the default. Would be quite nice to use something like an arrow based system though. 
* Or your own:

```python
Any(),
Or(), # or is not the same as all any, because it's one OR the other, not both -> only defined for two options.
All()
Not()
```

system

* Also need to find a way to do NotX.

- Measures (https://kornia.readthedocs.io/en/latest/feature.html): 
  - kornia.feature.gftt_response
  - 
