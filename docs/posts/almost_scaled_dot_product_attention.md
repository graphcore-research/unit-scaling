# Almost-scaled dot-product attention

[Notebook](TODO-notebook-link) | _TL;DR: Scaled dot product attention isn't properly scaled, and that's a good thing!_

Transformers seem to be all you need, but we don't fully understand why they work so well. While working on [unit scaling](https://arxiv.org/abs/2303.11257), we noticed something surprising about attention, the heart of the transformer architecture, and how the outputs are scaled.

Many deep learning modules are designed and initialised to roughly preserve variance in the forward and/or backward (gradient) passes. Dot product attention explicitly includes a scaling factor for this ({math}`d_{head}^{-1/2}`). But this is _insufficient for attention to preserve variance_. We have derived a post-scaling factor for attention:

```{figure} img/attention_scaling.png
---
width: 30em
align: center
alt: "attention scaling: regular attention is underscaled to sigma=0.1 when d_seq=256, but scaled to sigma=1.0 when using a sqrt(d_seq/e) multiplier"
---
```
<p/>

With this in mind, here is a "fix" for the output scale of standard attention (our change in red):

```{math}
A^{\prime} &= Q K^T \cdot d_{head}^{-1/2}

Z &= \mathrm{Softmax}(A^{\prime})\, V \color{red}{\cdot (d_{seq}/e)^{1/2}}
```

In this post, we'll look at the variance-scaling behaviour of attention, and explain this scaling factor, before seeing that it makes training dynamics _worse_, not better.

## Where does {math}`(d_{seq}/e)^{1/2}` come from?

Attention contains the expression {math}`Z=\mathrm{Softmax}(A^{\prime})V`. If we modify this slightly to {math}`Z=\mathrm{Softmax}(A^{\prime}/t)V`, we can think about three cases:

 - {math}`t\to \infty`, the output is flat, and the scale of {math}`Z` is {math}`d_{seq}^{-1/2}`, from a multiplication by {math}`d_{seq}^{-1}` then a sum over {math}`d_{seq}` uncorrelated values
 - {math}`t\to 0`, the output is a single unit spike, and the scale of {math}`Z` is {math}`1`, since attention selects a single element of {math}`V \sim N(0, 1)`
 - {math}`t \gt 1/2`, with some assumptions, the output follows a log-normal distribution, and the scale of {math}`Z` is {math}`(e^{t^{-2}}/d_{seq})^{1/2}`; _we explain this further in the [companion notebook](TODO-notebook-link)_

```{figure} img/softmax_temperature.png
---
align: center
width: 30em
alt: "effect of softmax temperature, flat when temperature is infinite, a spike when temperature is zero and a bumpy corve when temperature is one"
---
```
<p/>

We find that the log-normal scaling rule works well for temperature near 1, so propose multiplying by the inverse, i.e. scale attention output by {math}`(d_{seq}/e)^{1/2}`.

## Does it work? ...No!

We tested this change, introducing "fully scaled attention" in a full transformer model—a small autoregressive character language model trained on Shakespeare. This is what we saw from a learning rate sweep:

```{figure} img/scaled_attention_lr_sweep.png
---
align: center
width: 25em
alt: "learning rate sweep for baseline (standard attention) and fully scaled attention. Fully scaled attention behaves worse than the baseline (final training loss 1.2 for baseline, 1.4 for fully scaled)"
---
```
<p/>

This is most unfortunate. It seems that under-scaled tensors coming out of the attention block are important and helpful for transformer training dynamics. It isn't just tiny Shakespare models—we've also seen this effect when training BERT.

Note that it shouldn't be a question of model expressivity, as the scale factor can be "moved" into the output projection for each head; both standard attention and fully scaled attention can behave identically during inference. We don't yet have an explanation for this behaviour, but find it intriguing that such a (presumed) accident of under-scaling turns out to be helpful for training dynamics!

Unit scaling has a solution for this, allowing unit-scaled tensors while retaining the original training dynamics. The bad training behaviour must come from scale-dependent operations, in particular when attention's residual output is added to the skip connection. So, we found that we can reproduce the same dynamics as the original model by applying a relative weight to the residual vs skip connections.

## Conclusion

It is helpful to think through the scales of tensors in deep learning models. Indeed, careful reasoning about scale is the core principle underpinning unit scaling (which also considers the scale of gradients, not just activations).

In today's example, we saw how to "fix" attention's scaling behaviour, multiplying the outputs by {math}`(d_{seq}/e)^{1/2}`, so that the outputs are unit-variance. However we also saw that this change can make training dynamics worse, not better. Why this happens is, as far as we know, an open question.

If you're interested to find out more, check out our [accompanying notebook](TODO-notebook-link) and [unit scaling](https://arxiv.org/abs/2303.11257) paper.
