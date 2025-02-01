use std::f64::consts::*;

use logaddexp::LogAddExp;

use crate::*;
use crate::special;
use crate::distribution::categorical::TruncatedCategorical;
use crate::distribution::uniform::floor_rem;

////////////////////////////// STANDARD NORMAL //////////////////////////////

/// PDF of the standard normal
pub fn pdf(z: f64) -> f64 {
    let factor = // 1/sqrt(2pi)
	std::f64::consts::TAU.sqrt().recip();
    factor * (z * z * -0.5).exp()
}

/// log-PDF of the standard normal
pub fn log_pdf(z: f64) -> f64 {
    let factor = -0.5 * std::f64::consts::TAU.ln();
    factor * z * z // parabola
}

/// Returns a numerically stable `log(abs(cdf(hi) - cdf(lo)))` for the
/// standard normal. Should only return `log(0.0)` if the bounds are the
/// same or if the log-PDF at their mid point is `f64::NEG_INFINITY` (or
/// almost).
pub fn log_probability(mut lo: f64, mut hi: f64) -> f64 {
    debug_assert!(lo <= hi);
    if lo == hi { return f64::NEG_INFINITY }
    if lo > 0.0 { // both right tail since hi >= lo
	let tmp = -lo;
	lo = -hi; // switch to left
	hi = tmp; // switch to left
    }
    let lp = special::ln_sub_exp(special::log_ndtr(lo),
				 special::log_ndtr(hi));
    if lp > f64::NEG_INFINITY { lp }
    else { // ran out of precision
	// linear approximation (assume locally uniform)
	let lpd = log_pdf((lo + hi) * 0.5); // take PD at midpoint
	let ln_delta = (hi - lo).ln();
	lpd + ln_delta // height * width
    }
}

/// Quantile of the standard normal
pub fn quantile(cp: f64) -> f64 {
    SQRT_2 * statrs::function::erf::erf_inv(2.0 * cp - 1.0)
}

/// Quantile-exp (inverse of the log-CDF) of the standard normal
pub fn quantile_exp(lcp: f64) -> f64 {
    special::ndtri_exp(lcp)
}

/// log-CDF of the standard normal
pub fn log_cdf(x: f64) -> f64 {
    special::log_ndtr(x)
}

/// Does a linear interpolation on the area under the curve of the PDF
/// of the standard normal between two points. Does the computation in
/// log-scale so we're numerically stable in the tails
pub fn lerp(lo: f64, hi: f64, cp: f64) -> f64 {
    let mut lp = log_probability(lo,hi);
    if lo <= 0.0 { // left-tail
	lp += cp.ln();
	lp = lp.ln_add_exp(special::log_ndtr(lo));
	quantile_exp(lp)
    } else { // right-tail
	lp += (1.0 - cp).ln();
	lp = lp.ln_add_exp(special::log_ndtr(-hi));
	-quantile_exp(lp)
    }
}

////////////////////////////// PARAMETRIZED //////////////////////////////

#[derive(Debug,Copy,Clone,PartialEq,PartialOrd)]
pub struct Gaussian {
    pub s0: usize, // number of values
    pub s1: i64,   // sum of values
    pub s2: u128,  // sum of squares
    pub ddof: u8,  // 0 if values are population, 1 if samples
    pub mean: f64, // s1 / s0
    pub stdev: f64 // sqrt( (s2 - (s1^2)/s0) / s0 - ddof )
}

impl Gaussian {
    pub fn variance(&self) -> f64 {
	self.stdev * self.stdev
    }

    pub fn from_sums(s0: usize, s1: i64, s2: u128, ddof: u8) -> Self {
	if s0 == 0 {
	    debug_assert!(s1 == 0 && s2 == 0);
	    panic!("Empty Gaussian distribution undefined")
	}
	if s0 == 1 {
	    let mean = s1 as f64;
	    let stdev = 0.0; // degen
	    Gaussian{ s0, s1, s2, ddof, mean, stdev }
	} else {
	    let s0f = s0 as f64;
	    let s1f = s1 as f64;
	    let s2f = s2 as f64;
	    let mean = s1f / s0f;
	    let var = (s2f - s1f*s1f/s0f) / (s0f - ddof as f64);
	    let stdev = var.sqrt();
	    Gaussian{ s0, s1, s2, ddof, mean, stdev }
        }
    }

    /// One pass fit of a Gaussian to given data, given the desired
    /// delta-degree-of-freedom (0 if data is the whole population, 1 if
    /// data is a sample)
    pub fn from_values<I>(data: I, ddof: u8) -> Self
    where I: Iterator<Item=i64>
    {
	let mut s0 = 0usize;
	let mut s1 = 0i64;
	let mut s2 = 0u128;
	for x in data {
	    s0 += 1;
	    s1 = s1.checked_add(x).unwrap();
	    let x_abs = x.abs() as u128;
	    let x2 = x_abs * x_abs;
	    s2 = s2.checked_add(x2).unwrap();
	}
	Self::from_sums(s0, s1, s2, ddof)
    }

    /// PDF of the normal
    pub fn pdf(&self, x: f64) -> f64 {
	let scale = self.stdev.recip();
	let scaled_delta = scale * (x - self.mean);
	let pi = std::f64::consts::PI;
	scale * (2.0 * pi).sqrt().recip()
	    * f64::exp(scaled_delta * scaled_delta * -0.5)
    }

    /// CDF of the normal
    pub fn cdf(&self, x: f64) -> f64 {
	if self.stdev > 0.0 {
	    let delta = x - self.mean;
	    let denom = self.stdev * SQRT_2;
	    (0.5) * (1.0 + libm::erf(delta/denom))
	} else if x >= self.mean {
	    1.0
	} else {
	    0.0
	}
    }

    /// log-CDF of the normal
    pub fn log_cdf(&self, x: f64) -> f64 {
	let z = (x - self.mean) / self.stdev;
	log_cdf(z)
    }

    pub fn log_pdf(&self, x: f64) -> f64 {
	let z = (x - self.mean) / self.stdev;
	log_pdf(z)
    }

    /// Log of the complement of the CDF of the normal
    pub fn log_survival(&self, x: f64) -> f64 {
	let z = (x - self.mean) / self.stdev;
	log_cdf(-z)
    }

    /// Returns a numerically stable `log(abs(cdf(hi) - cdf(lo)))`
    pub fn log_probability(&self, lo: f64, hi: f64) -> f64 {
	if self.stdev > 0.0 { // not degenerate
	    let precision = self.stdev.recip();
	    let lo_z = (lo - self.mean) * precision;
	    let hi_z = (hi - self.mean) * precision;
	    log_probability(lo_z,hi_z)
	} else if lo < self.mean && hi >= self.mean {
	    0.0 // log(1)
	} else {
	    f64::NEG_INFINITY // log(0)
	}
    }

    pub fn quantile(&self, cp: f64) -> f64 {
	quantile(cp) * self.stdev
	    + self.mean
    }

    /// Returns the index of the bin from a division of the curve into
    /// `q` subsets of equal probability
    pub fn q_cdf(&self, q: usize, x: f64) -> usize {
	let cp = self.cdf(x); // 0-1
	let n = cp * q as f64;
	(n.floor() as usize).min(q-1)
    }

    /// Returns the `i`'th `q`'tile of self, i.e. the value at the
    /// boundary between the `i`th and the `i+1`th bin of equal `1/q`
    /// probability for this Gaussian. Input `i` should be between `1`
    /// and `q-1` for a real value: `i=0` will return
    /// `f64::NEG_INFINITY` and `i=q` will return `f64::INFINITY`. Call
    /// with `q*2` and `i*2 + 1` to get the middle of the bin above the
    /// boundary
    pub fn q_quantile(&self, q: usize, i: usize) -> f64 {
	let cp = (i as f64) / q as f64;
	self.quantile(cp)
    }

    /// Does a linear interpolation on the area under the curve of the
    /// PDF of the gaussian between two points. Does the computation in
    /// log-scale so we're numerically stable in the tails
    pub fn lerp(&self, lo: f64, hi: f64, cp: f64) -> f64 {
	if self.stdev > 0.0 { // not degenerate
	    let precision = self.stdev.recip();
	    let lo_z = (lo - self.mean) * precision;
	    let hi_z = (hi - self.mean) * precision;
	    let res_z = lerp(lo_z,hi_z,cp);
	    res_z * self.stdev + self.mean
	} else {
	    debug_assert!(lo <= self.mean && hi >= self.mean);
	    self.mean
	}
    }

    /// Definite integral (equiv. to `self.cdf(hi) - self.cdf(lo)`)
    pub fn definite_integral(&self, lo: f64, hi: f64) -> f64 {
	let delta_lo = lo - self.mean;
	let delta_hi = hi - self.mean;
	let factor = (self.stdev * SQRT_2).recip(); // recip of denom
	(0.5) * (libm::erf(delta_hi * factor)
		    - libm::erf(delta_lo * factor))
    }

    /// Translate the distribution to the right (positive) or to the
    /// left (negative) by adding a given amount to the mean
    pub fn translate(&self, d: f64) -> Gaussian {
	let mut g = *self;
	g.mean += d;
	g
    }
}

impl UnivariateDistribution for Gaussian {
    fn truncated(&self) -> Box<dyn TruncatedDistribution> {
	let gaussian = self.clone().translate(0.5); // see NOTE below
	let lo = i64::MIN;
	let one = 0.0; // log(1)
	let zero = f64::NEG_INFINITY;
	let mut bins = // new() trims 0 and 2 off
	    TruncatedCategorical::new(vec![one]);
	bins.ln_ps = vec![zero,one,zero];
	let hi = i64::MAX;
	let ln_prob = 0.0; // log(1);
	Box::new(TruncatedGaussian{ gaussian, lo, bins, hi, ln_prob })
    }
}

impl Model<i64> for Gaussian {
    fn push(&mut self, s: i64) -> Option<i64> {
	Some(s)
    }
    fn next_distr(&mut self) -> Box<dyn UnivariateDistribution> {
	Box::new(*self)
    }
}

#[derive(Debug,Clone,PartialEq,PartialOrd)]
pub struct TruncatedGaussian {
    pub gaussian: Gaussian, // prior distribution over the numbers (see NOTE)
    pub lo: i64, // floored lower-bound
    pub bins: TruncatedCategorical, // [lo_crem, mid, hi_rem] cat. distr.
    pub hi: i64, // floored upper-bound
    pub ln_prob: f64, // log-probability of all bins
}

// NOTE: For the whole implementation of truncated_gaussian, the bins
// are the widths of whole numbers so [0.0, 1.0), [1.0, 2.0), etc. Since
// this disagrees with how we fit a gaussian to values (we would have to
// add 0.5 to each value of the sample to get a mle), we add 0.5 to the
// mean whenever converting into a truncated_gaussian, so gaussians fit
// as you would expect and we can call gaussian::log_probability as you
// would expect, etc.
impl TruncatedDistribution for TruncatedGaussian {
    fn lookup(&self, cp: f64) -> (i64, f64) {
	let (case, case_split) = self.bins.lookup(cp);
	if      case == 0 { (self.lo, case_split) }
	else if case == 2 { (self.hi, case_split) }
	else { // mid
	    assert!(case == 1);
	    let lo = (self.lo + 1) as f64;
	    let hi = self.hi as f64;
	    let lerp = self.gaussian.lerp(lo,hi,case_split);
	    let (mut s, mut s_split) = floor_rem(lerp);

	    let progress =
		( s > self.lo + 1 || (s == self.lo + 1
					    && s_split > 0.0) )
		&& s < self.hi;

	    if !progress && cp == 0.5 {
		// (ran out of precision) assume locally uniform
		let delta = self.hi - (self.lo + 1);
		debug_assert!(delta > 0); // by case
		let inc = case_split * delta as f64;
		let inc_floor = inc.floor();
		s = self.lo + 1 + inc_floor as i64;
		s_split = inc - inc_floor;
	    }

	    (s, s_split)
	}
    }

    fn truncate(&mut self, cp: f64, s: i64, s_split: f64, bit: bool) {
	if s == self.lo { // lo
	    if bit {
		self.bins.ln_ps[0] += (1.0 - s_split).ln();
		self.bins.normalize();
	    } else { // solved
		self.hi = self.lo;
		// [1,0,0] but could be [0,0,1]:
		self.bins.ln_ps[0] = 0.0;
		self.bins.ln_ps[1] = f64::NEG_INFINITY;
		self.bins.ln_ps[2] = f64::NEG_INFINITY;
	    }

	} else if s == self.hi { // hi
	    if bit { // solved
		self.lo = self.hi;
		// [0,0,1] but could be [1,0,0]:
		self.bins.ln_ps[0] = f64::NEG_INFINITY;
		self.bins.ln_ps[1] = f64::NEG_INFINITY;
		self.bins.ln_ps[2] = 0.0;

	    } else {
		self.bins.ln_ps[2] += s_split.ln();
		self.bins.normalize();
	    }

	    // FIXME: case is broken when precision is 52
	} else { // mid
	    debug_assert!(self.lo < s && s < self.hi);
	    let s_lo = s as f64;
	    let s_hi = (s + 1) as f64;
	    // TODO: ideally we would have an integer version of
	    // log_probability so we don't lose the delta by converting
	    // to floats (e.g. mean is very high) but the largest ints
	    // we model are 52 bits long and 2^51 just fits in an f64
	    //
	    // (we might actually be able to get away with a single call
	    // to log_pdf, and give up the whole definite integral thing)
	    let s_lp = if s_lo != s_hi {
		self.gaussian.log_probability(s_lo, s_hi)
	    } else {
		self.gaussian.log_pdf(s_lo)
	    };
	    assert!(s_lp > f64::NEG_INFINITY); // P > 0.0
	    assert!(s_lp <= 0.0); // P <= 1.0

	    let lo_bin;
	    let hi_bin;
	    if bit {
		self.lo = s; // new lo
		let lccp = (1.0 - cp).ln();
		self.ln_prob += lccp;
		hi_bin = self.bins.ln_ps[2] - lccp; // just scale
		lo_bin = s_lp // absolute log-prob
		    + (1.0 - s_split).ln() // fraction
		    - self.ln_prob; // re-base
		debug_assert!(lo_bin > f64::NEG_INFINITY);
	    } else {
		self.hi = s; // new hi
		let lcp = cp.ln();
		self.ln_prob += lcp;
		lo_bin = self.bins.ln_ps[0] - lcp; // just scale
		hi_bin = s_lp // absolute lp
		    + s_split.ln() // fraction
		    - self.ln_prob; // re-base
		debug_assert!(hi_bin > f64::NEG_INFINITY || s_split == 0.0);
	    }

	    let mid_bin = if self.lo + 1 == self.hi {
		f64::NEG_INFINITY
	    } else {
		let lop1 = (self.lo + 1) as f64;
		let hi = self.hi as f64;
		let mut mid_lp =
		    self.gaussian.log_probability(lop1,hi);
		mid_lp -= self.ln_prob; // re-base
		mid_lp
	    };

	    self.bins.ln_ps = // (new() would try to trim)
		vec![lo_bin,mid_bin,hi_bin];
	    self.bins.normalize();
	}
    }

    fn lo(&self) -> i64 { self.lo }
    fn hi(&self) -> i64 { self.hi }
}
