Limitations
===========

:code:`unit-scaling` is currently still in early stages of development, and many
important features have not yet been implement. The following is a list of some key
missing functionality on the development roadmap:

1. **Fused scales:** currently the scaling factors introduced by unit-scaled operations
   are not fused-in. This means that the library will currently slow down training as it
   requires many more trips to memory than a non-unit-scaled model (we intend to address
   this problem next).
2. **Causal masking**
3. **Positional embeddings**
4. **Flash attention**
5. **Distributed training:** the interplay between unit scaling and
   distributed training libraries has not yet been investigated.
6. **Limited set of ops:** the set of ops currently supported is limited to those
   required for training transformers. Unit scaling is a more general method that will
   work for arbitrary models once support has been implemented.

This list is not exhaustive and we encourage users to get in touch if they have
feature-requests not listed here.
