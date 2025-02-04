[Documentation](https://nbos.ca/doc/cont_arith_code/)

## How to Use

For a given value`:T`, you provide a Model (magenta) against which the
value you want to encode decomposes into a stream of symbols indexed by
$s_0,s_1,s_2,...$`:i64` inside Distributions (green) which the model
produces. The model is updated by the index of the previous symbol
before being called to generate the next distribution.

![](https://nbos.ca/images/interface.png)

In practice, you implement for your model the
[trait](https://nbos.ca/doc/cont_arith_code/trait.Model.html)

```Rust
type Index = i64;
pub trait Model<T> {
    fn next_distr(&mut self) -> Box<dyn UnivariateDistribution>;
    fn push(&mut self, s: Index) -> Option<T>;
}
```

where
[`UnivariateDistribution`](https://nbos.ca/doc/cont_arith_code/trait.UnivariateDistribution.html)
is

```Rust
pub trait UnivariateDistribution {
    fn truncated(&self) -> Box<dyn TruncatedDistribution>;
}
```

and
[`TruncatedDistribution`](https://nbos.ca/doc/cont_arith_code/trait.TruncatedDistribution.html)
is any distribution with a [quantile
function](https://en.wikipedia.org/wiki/Quantile_function) that returns
a bin's index (with remainder) for any cumulative probability `cp: f64`
$\in (0,1)$ and does so with successive truncations of the probability
mass until both bounds of the probability mass fall within the same
index.

```Rust
pub trait TruncatedDistribution {
    fn quantile(&self, cp: f64) -> (Index, f64); // returns (s, s_rem)
    fn truncate(&mut self, cp: f64, s: Index, s_rem: f64, bit: bool);
    fn lo(&self) -> Index; // symbol the lower-bound is in
    fn hi(&self) -> Index; // symbol the upper-bound is in
    fn is_resolved(&self) -> bool { self.lo() == self.hi() }
}
```

Included are implementations for
[`Categorical<T>`](https://nbos.ca/doc/cont_arith_code/distribution/categorical/index.html)
distributions on domain `T` with per-symbol frequencies/probabilities
and
[`Gaussian`](https://nbos.ca/doc/cont_arith_code/distribution/gaussian/index.html)
distributions on domain `i64`, where each integer gets a bin with a
$\pm$`0.5: f64` width around it.

Care must be taken when implementing for `TruncatedDistribution`, as the
code length will only be as *optimal* as the modeling of its probability
mass is *precise*, in particular with `quantile` and
`truncate`. Moreover, the algorithm will loop if the most informative
splits (i.e. at `cp = 0.5`) produce no significant progress in the
probability mass.
