use std::fmt::Debug;
use std::collections::BTreeMap;

use logaddexp::LogSumExp;

use crate::*;
use crate::map::Map;

////////////////////////////// CANONICAL //////////////////////////////

#[derive(Debug,Clone,PartialEq,PartialOrd)]
pub struct Categorical<T>{
    /// Number of observations
    pub count: usize,
    /// Symbol -> (count, log-probability)
    pub map: Map<T,(usize,f64)>
}

impl<T> Categorical<T> {
    pub fn singleton(k: T) -> Categorical<T> {
	let count = 1;
	let map = Map::singleton(k,(1,(0.0))); // log(1)
	Categorical{ count, map }
    }

    fn from_sorted(counts: Vec<(T,usize)>) -> Self {
	debug_assert!(counts.iter().all(|(_,n)| *n > 0));
	let count = counts.iter().map(|(_,n)| n).sum();
	let ln_count = (count as f64).ln();
	let lps = counts.into_iter()
	    .map(|(k,n)| (k, (n, (n as f64).ln() - ln_count)))
	    .collect::<Vec<_>>();
	let map = Map(lps);
	Categorical{ count, map }
    }

    /// Fit a cateogrical distribution to the given dataset
    pub fn from_data<I>(data: I) -> Self
    where
	T: Ord,
	I: IntoIterator<Item=T>
    {
	let mut counts = BTreeMap::new();
	for item in data {
	    *counts.entry(item).or_insert(0) += 1;
	}
	Self::from_sorted(counts.into_iter().collect())
    }

    pub fn len(&self) -> usize {
	self.map.len()
    }

    /// Looks up a key using binary search
    pub fn lookup(&self, key: &T) -> Option<f64>
    where T: Clone + Ord
    {
        self.map.lookup(key).copied().map(|p| p.1)
    }

    pub fn index_of(&self, key: &T) -> Option<usize>
    where T: Clone + Ord
    {
	self.map.index_of(key)
    }

    pub fn get_key(&self, i: usize) -> &T {
	self.map.get_key(i)
    }

    /// Log-probability mass function
    pub fn log_pmf(&self, key: &T) -> f64
    where T: Clone + Ord
    {
	self.lookup(key)
	    .map(|lp| lp)
	    .unwrap_or(f64::NEG_INFINITY)
    }

    /// Entropy of the categorical in nats
    pub fn entropy(&self) -> f64 {
	-self.map.values().map(|&(n,lp)| (n as f64) * lp).sum::<f64>()
	    / self.count as f64
    }

    /// Kullback-Leibler divergence D(P||Q) of self (P) from a second
    /// Categorical (Q) (nats). Panics if some entry of self is absent
    /// in `Q`.
    pub fn kld(&self, q: &Self) -> f64
    where T: Clone + Ord
    {
	self.map.iter().map(|(k,(_,lp))| {
	    lp.exp() * (*lp - q.map.lookup(&k).unwrap().1)
	}).sum::<f64>()
    }

    pub fn log_probability(&self, val: &T) -> f64
    where T: Clone + Ord
    {
	self.map.lookup(val)
	    .map(|(_,r)| *r)
	    .unwrap_or(f64::NEG_INFINITY)
    }
}

impl<T: Debug> UnivariateDistribution for Categorical<T> {
    fn truncated(&self) -> Box<dyn TruncatedDistribution> {
       Box::new(
           TruncatedCategorical{
               lo: 0,
               ln_ps: self.map.values()
                   .map(|(_,r)| *r)
                   .collect() // Vec
           }
       )
    }
}


impl<T: Clone + Ord + Debug + 'static> Model<T> for Categorical<T> {
    fn push(&mut self, s: i64) -> Option<T> {
	Some(self.get_key(s as usize).clone())
    }
    fn next_distr(&mut self) -> Box<dyn UnivariateDistribution> {
	Box::new(self.clone())
    }
}

////////////////////////////// TRUNCATED //////////////////////////////

#[derive(Debug,Clone,PartialEq,PartialOrd)]
pub struct TruncatedCategorical {
    lo: usize,
    pub ln_ps: Vec<f64>, // vector of log-probabilities
}

impl TruncatedCategorical {
    /// Construction given a vector of log-probabilities
    pub fn new(ln_ps: Vec<f64>) -> TruncatedCategorical {
	let mut cat = TruncatedCategorical{ lo: 0, ln_ps };
	cat.trim_left();
	cat.trim_right();
	cat.normalize();
	cat // ^•⩊•^
    }

    pub fn normalize(&mut self) {
	let total_ln_p = self.ln_ps.iter().copied().ln_sum_exp();
	for lp in self.ln_ps.iter_mut() {
	    *lp -= total_ln_p
	}
    }

    fn trim_left(&mut self) {
	let non_zero = |lp| lp != f64::NEG_INFINITY;
	if non_zero(self.ln_ps[0]) { return } // interrupt
	let lo = self.ln_ps.iter().position(|&lp| non_zero(lp)).unwrap();
	self.ln_ps = self.ln_ps[lo..].to_vec();
	self.lo += lo;
    }

    fn trim_right(&mut self) {
	let non_zero = |lp| lp != f64::NEG_INFINITY;
	if non_zero(*self.ln_ps.last().unwrap()) { return } // interrupt
	let hi = self.ln_ps.iter().rposition(|&lp| non_zero(lp)).unwrap();
	self.ln_ps.truncate(hi + 1); // keep hi
    }
}

impl TruncatedDistribution for TruncatedCategorical {
    /// Given a cummulative probability [0-1], return the category/bin
    /// in which it falls (intervals are inclusive of lower-bound,
    /// exclusive of upper-bount, i.e. `[lo,hi)`). Cummulative
    /// probability is counted from self.lo_split (our zero) up to
    /// self.hi_split (our one). Returns the index of the bin and where
    /// (as a ratio) the cummulative probability falls within that
    /// bin. Ratio on lo and hi bounds are counted relative to their
    /// respective prior split.
    fn quantile(&self, cp: f64) -> (i64, f64) {
	// compute cummulative probabilities
	let ps = self.ln_ps.iter()
	    .map(|lp| (lp.exp())) // lin space
	    .collect::<Vec<_>>();

	let mut cp1 = 0.0;
	let mut cps = ps.iter().map(|&p| { cp1 += p; cp1 }).collect::<Vec<_>>();
	let total_p = *cps.last().unwrap();
	let scale = total_p.recip();
	for cp in cps.iter_mut() { *cp *= scale } // re-normalize

	// find split-point
	let s = cps.partition_point(|&cp1| cp1 <= cp); // [ )[ )|[ )[ )[ )
	let s_lo = if s == 0 { 0.0 } else { cps[s - 1] };
	let s_rem = (cp - s_lo) / (scale * ps[s]);

	((s + self.lo) as i64, s_rem)
    }

    fn truncate(&mut self, cp: f64, s: i64, s_rem: f64, bit: bool) {
	let i = s as usize - self.lo; // actual index
	if bit { // 1
	    self.lo += i;
	    self.ln_ps = self.ln_ps[i..].to_vec();
	    self.ln_ps[0] += (1.0 - s_rem).ln();
	    let lccp = (1.0 - cp).ln();
	    for lp in self.ln_ps.iter_mut() {
		*lp -= lccp;
	    }

	} else { // 0
	    self.ln_ps.truncate(i+1); // keep ln_ps[s]
	    *self.ln_ps.last_mut().unwrap() += s_rem.ln();
	    self.trim_right(); // required when you hit a bound with 0s
	    let lcp = cp.ln();
	    for lp in self.ln_ps.iter_mut() {
		*lp -= lcp;
	    }
	}
    }

    fn lo(&self) -> i64 { self.lo as i64 }
    fn hi(&self) -> i64 { (self.lo + self.ln_ps.len() - 1) as i64 }

    /// Assumes input is trimmed
    fn is_resolved(&self) -> bool { self.ln_ps.len() == 1 }
}
