## Testing 

### Testing SQL

Should be possible to check that the SQL generated is valid for the target database.
<https://towardsdatascience.com/jinja-sql-%EF%B8%8F-7e4dff8d8778>

This also looks promising <https://pypi.org/project/python-bigquery-validator/>

### Testing Requirements

Possible you could just have a test that each file e.g. detectron2_train.py 
can be run, the complication is the GPU <-> CPU dynamic.
