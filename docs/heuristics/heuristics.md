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
  Cornerness(min='', max=''), 
  Circleness(min='', max=''),
  Squareness(min='', max=''),
  Reflectivity(min='', max=''),
  Transparency(min='', max=''),
])
```

```python
Redirection(
  trigger_class='car',
  heuristic=ShapeFilter(...),
  output_class='truck'
)

```

```python
FalsePostive(
  trigger_class='car', 
  heurstic=[], 
)
```

Don't know when this becomes prohibitively slow to compute. 

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
  - Somehow include Hu Moments (how you'd recognise a circle or square)
  - Hog features and SIFT features
  - Should be able to define regions of 'colour space' (as you do when creating a colour in Microsoft Paint or similar) https://theconversation.com/how-rainbow-colour-maps-can-distort-data-and-be-misleading-16715 (rainbow-color-map)
  - This is what HSV is useful for https://stackoverflow.com/questions/42882498/what-are-the-ranges-to-recognize-different-colors-in-rgb-space
  - Great post on measurement of water levels https://stackoverflow.com/questions/54950777/opencv-drawing-contours-with-various-methods-on-a-poor-image
