use crate::*;

#[derive(Debug,Clone,Copy,PartialEq,Eq,PartialOrd,Ord,Hash)]
pub struct Uniform {
    pub lo: i64, // lower-bound (inclusive)
    pub hi: i64  // upper-bound (exclusive)
}

impl Uniform {
    pub fn unit(n: i64) -> Self {
	Uniform{ lo: n,
		 hi: n+1 }
    }

    /// Interval of 0 to n (exclusive)
    pub fn iota(n: i64) -> Self {
	Uniform{ lo: 0,
		 hi: n }
    }

    /// hi - lo
    pub fn span(&self) -> i64 {
	self.hi - self.lo
    }

    pub fn join(&self, rhs: &Self) -> Self {
	Uniform{ lo: self.lo.min(rhs.lo),
		 hi: self.hi.max(rhs.hi) }
    }

    /// Probability mass within the bounds
    pub fn pm(&self) -> f64 {
	(self.span() as f64).recip()
    }

    /// Entropy of the distribution in nats
    pub fn entropy(&self) -> f64 {
	(self.span() as f64).ln()
    }

    /// Probability mass function
    pub fn pmf(&self, x: i64) -> f64 {
	if x >= self.lo && x < self.hi {
	    self.pm()
	} else {
	    0.0
	}
    }

    /// Log of probability mass function
    pub fn log_pmf(&self, x: i64) -> f64 {
	(self.pmf(x)).ln()
    }

    /// Kullback-Leibler divergence in nats
    pub fn kld(&self, q: &Self) -> f64 {
	debug_assert!(q.lo <= self.lo && q.hi >= self.hi);
	(self.pm() / q.pm()).ln()
    }

    /// Jensen-Shannon distance (sqrt of JSD)
    pub fn jensen_shannon(&self, rhs: &Self) -> f64 {
	let join = self.join(rhs);
	let jsd = (0.5) * (self.kld(&join) + rhs.kld(&join));
	jsd.sqrt()
    }
}

impl UnivariateDistribution for Uniform {
    fn truncated(&self) -> Box<dyn TruncatedDistribution> {
	let lo = (self.lo, (0.0));
	let hi = (self.hi, (0.0)); // (self.hi - 1, (1.0))?
	Box::new( TruncatedUniform{ lo, hi } )
    }
}

impl Model<i64> for Uniform {
    fn push(&mut self, s: Index) -> Option<i64> {
	Some(s)
    }
    fn next_distr(&mut self) -> Box<dyn UnivariateDistribution> {
	Box::new(*self)
    }
}

#[derive(Debug,Clone,PartialEq)]
pub struct TruncatedUniform {
    pub lo: (i64,f64), // lower-bound (floored, rem)
    pub hi: (i64,f64), // upper-bound (floored, rem)
}


/// Separate the fractional part of a number of the integer part
pub fn floor_rem(x: f64) -> (i64,f64) {
    // TODO: rewrite with trunc() and fract() (neg is special case?)
    let x_floor = x.floor();
    let x_int = x_floor as i64;
    if x.is_finite() { (x_int, ((x - x_floor))) }
    else { (x_int,(0.0)) }
}

impl TruncatedUniform {
    pub fn span(&self) -> (i64,f64) {
	let mut span0 = self.hi.0 - self.lo.0;
	let mut span1 = self.hi.1 - self.lo.1;
	if span1 < 0.0 {
	    span0 -= 1;
	    span1 += 1.0;
	}
	(span0,span1)
    }
}

impl TruncatedDistribution for TruncatedUniform {
    fn lookup(&self, cp: f64) -> (Index, f64) {
	let span = self.span();
	let span = span.1 + span.0 as f64; // cram into f64
	let inc = floor_rem(cp * span);
	let mut s = self.lo.0 + inc.0;
	let mut s_split = self.lo.1 + inc.1;
	if s_split >= 1.0 {
	    s_split -= 1.0;
	    s += 1;
	}
	(s, s_split)
    }

    fn truncate(&mut self, _cp: f64, s: Index, s_split: f64, bit: bool) {
	if bit {
	    self.lo = (s, s_split);
	} else {
	    self.hi = (s, s_split);
	}
    }

    fn lo(&self) -> Index { self.lo.0 }
    fn hi(&self) -> Index { self.hi.0 }
}
