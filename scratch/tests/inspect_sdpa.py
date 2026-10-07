from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
import inspect

fn = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", None)
print("Nome:", fn.__name__)
lines = inspect.getsourcelines(fn)[0]
for l in lines[:30]:
    print(l.rstrip())
