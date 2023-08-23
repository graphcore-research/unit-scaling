Limitations
===========

:code:`unit-scaling` is a new library and (despite our best efforts!) we can't guarantee
it will be bug-free or feature-complete. We're keen to assist anyone who wants to use
the library, and help them work through any issues.

Known limitations of the library include:

1. **Op coverage:** we've currently focussed on adding common transformer operations — other ops may be missing (though we can add most requested ops without difficulty).
2. **Using transforms with torch.compile:** currently our transforms (for example :code:`unit_scale`, :code:`simulate_fp8`) can't be used directly with :code:`torch.compile`. We provide a special compilation function to get around this: :code:`unit_scaling.transforms.compile` (see docs for more details), though this only works with :code:`unit_scale` and not :code:`simulate_fp8`.
3. **Distributed training:** although we suspect distributed training will still work reasonably well with the current library, we haven't tested this.

This list is not exhaustive and we encourage you to get in touch if you have
feature-requests not listed here.
