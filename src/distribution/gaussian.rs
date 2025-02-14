use std::f64::consts::*;

use libm::erf;
use logaddexp::LogAddExp;
use statrs::function::erf::erf_inv;

use crate::*;
use crate::special;
use crate::distribution::categorical::TruncatedCategorical;

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
	eprintln!("\x1b[31mRan out of precision on Gaussian tail\x1b[0m");
	// linear approximation (assume locally uniform)
	let lpd = log_pdf((lo + hi) * 0.5); // take PD at midpoint
	let ln_delta = (hi - lo).ln();
	lpd + ln_delta // height * width
    }
}

/// Quantile of the standard normal
pub fn quantile(cp: f64) -> f64 {
    SQRT_2 * erf_inv(2.0 * cp - 1.0)
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

#[derive(Copy,Clone,PartialEq,PartialOrd)]
/// Numerically stable Gaussian distribution
pub struct Gaussian {
    /// Number of values
    pub s0: usize,
    /// Sum of values
    pub s1: i64,
    /// Sum of squares
    pub s2: u128,
    /// `0` if values are population, `1` if sample
    pub ddof: u8,
    /// `s1 / s0`
    pub mean: f64,
    /// `sqrt( (s2 - (s1^2)/s0) / s0 - ddof )`
    pub stdev: f64
}

impl std::fmt::Debug for Gaussian {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Gaussian {{ μ: {}, σ: {} }} ({}, {}, {})",
            self.mean, self.stdev, self.s0, self.s1, self.s2
        )
    }
}

impl Gaussian {
    pub fn variance(&self) -> f64 {
	self.stdev * self.stdev
    }

    fn from_sums(s0: usize, s1: i64, s2: u128, ddof: u8) -> Self {
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

    /// One pass fit (Max Likelihood Estimation) of a Gaussian to given
    /// data, given the desired delta-degree-of-freedom (0 if data is
    /// the whole population, 1 if data is a sample)
    pub fn mle<I>(data: I, ddof: u8) -> Self
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
	    (0.5) * (1.0 + erf(delta/denom))
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
	} else if lo <= self.mean && hi >= self.mean {
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
	if self.stdev <= 0.0 {
	    self.mean
	} else { // not degenerate
	    let precision = self.stdev.recip();
	    let lo_z = (lo - self.mean) * precision;
	    let hi_z = (hi - self.mean) * precision;
	    let res_z = lerp(lo_z,hi_z,cp);
	    res_z * self.stdev + self.mean
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

    /// Count the information (bits) of a given index in the Gaussian,
    /// i.e. the log-probability of the +/-0.5 interval around an
    /// integer, in base 2
    pub fn bits(&self, s: Index) -> f64 {
	let res = -self.log_probability(s as f64 - 0.5, s as f64 + 0.5)
	    / std::f64::consts::LN_2;
	if res == 0.0 { 0.0 } else { res } // -0.0
    }
}

impl UnivariateDistribution for Gaussian {
    fn truncated(&self) -> Box<dyn TruncatedDistribution> {
	let mut bins = TruncatedCategorical::new(vec![0.0]);
	// new() trims 0 and 2 off, so we have to manually:
	bins.ln_ps = vec![f64::NEG_INFINITY,
			  0.0,
			  f64::NEG_INFINITY];
	let lo;
	let hi;
	if self.stdev <= 0.0 { // (already resolved)
	    let s = self.mean.round() as i64;
	    lo = s;
	    hi = s;
	} else { // bounds at infinity
	    lo = i64::MIN;
	    hi = i64::MAX;
	}

	let ln_prob = 0.0; // log(1);
	Box::new(TruncatedGaussian{ gaussian: self.clone(),
				    lo, bins, hi, ln_prob })
    }
}

#[derive(Debug,Clone,PartialEq,PartialOrd)]
pub struct TruncatedGaussian {
    pub gaussian: Gaussian, // prior distribution over the numbers
    pub lo: i64, // integer nearest to lower-bound
    pub bins: TruncatedCategorical, // [lo_crem, mid, hi_rem] cat. distr.
    pub hi: i64, // integer nearest to upper-bound
    pub ln_prob: f64, // log-probability of all bins
}

impl TruncatedGaussian {
    /// Count the information (bits) of a given index in the truncated
    /// Gaussian, i.e. the log-probability of the +/-0.5 interval around
    /// an integer, in base 2
    pub fn bits(&self, s: Index) -> f64 {
	let ln2 = std::f64::consts::LN_2;
	if s < self.lo { f64::INFINITY }
	else if s == self.lo { -self.bins.ln_ps[0] / ln2 }
	else if s == self.hi { -self.bins.ln_ps[2] / ln2 }
	else if s > self.hi { f64::INFINITY }
	else {
	    let s_lo = s as f64 - 0.5;
	    let s_hi = s as f64 + 0.5;
	    -(self.gaussian.log_probability(s_lo,s_hi)
	      - self.ln_prob) / ln2
	}
    }
}

impl TruncatedDistribution for TruncatedGaussian {
    fn quantile(&self, cp: f64) -> (i64, f64) {

	// degen case:
	if self.gaussian.stdev <= 0.0 {
	    return (self.gaussian.mean.round() as i64, 0.5)
	}

	// normal case:
	let (case, case_cp) = self.bins.quantile(cp);
	if      case == 0 { (self.lo, case_cp) }
	else if case == 2 { (self.hi, case_cp) }
	else { // midle bin
	    debug_assert!(case == 1);
	    let lo = self.lo as f64 + 0.5;
	    let hi = self.hi as f64 - 0.5;
	    let lerp = self.gaussian.lerp(lo,hi,case_cp);
	    let s_f64 = lerp.round();
	    let s_lo = s_f64 - 0.5;
	    let s = s_f64 as i64;

	    let s_case_cp = (self.gaussian.log_probability(lo,s_lo)
			     // normalize like case_cp is:
			     + self.bins.ln_ps[1]).exp();

	    debug_assert!(s_case_cp <= case_cp);
	    let s_rem = case_cp - s_case_cp;
	    debug_assert!(0.0 <= s_rem && s_rem < 1.0);

	    let progress =
		( s > self.lo + 1 || (s == self.lo + 1
					    && s_rem > 0.0) )
		&& s < self.hi;

	    if !progress && cp == 0.5 {
		eprintln!("\x1b[31mFailed to progress on a half split\x1b[0m");
		// // (ran out of precision) assume locally uniform
		// // TODO: lerp in the log-domain instead since it's flatter?
		// let delta = self.hi - (self.lo + 1);
		// debug_assert!(delta > 0); // by case
		// let inc = case_split * delta as f64;
		// let inc_floor = inc.floor();
		// s = self.lo + 1 + inc_floor as i64;
		// s_rem = inc - inc_floor;
		unimplemented!()
	    }

	    (s, s_rem)
	}
    }

    fn truncate(&mut self, cp: f64, s: i64, s_rem: f64, bit: bool) {
	// case lo:
	if s == self.lo {
	    if bit {
		self.bins.ln_ps[0] += (1.0 - s_rem).ln();
		self.bins.normalize();
		let lccp = (1.0 - cp).ln();
		self.ln_prob += lccp;
	    } else { // solved
		self.hi = self.lo;
		// [1,0,0] but could be [0,0,1]:
		self.bins.ln_ps[0] = 0.0;
		self.bins.ln_ps[1] = f64::NEG_INFINITY;
		self.bins.ln_ps[2] = f64::NEG_INFINITY;
		let lcp = cp.ln();
		self.ln_prob += lcp;
	    }

	// case hi:
	} else if s == self.hi { // hi
	    if bit { // solved
		self.lo = self.hi;
		// [0,0,1] but could be [1,0,0]:
		self.bins.ln_ps[0] = f64::NEG_INFINITY;
		self.bins.ln_ps[1] = f64::NEG_INFINITY;
		self.bins.ln_ps[2] = 0.0;
		let lccp = (1.0 - cp).ln();
		self.ln_prob += lccp;
	    } else {
		self.bins.ln_ps[2] += s_rem.ln();
		self.bins.normalize();
		let lcp = cp.ln();
		self.ln_prob += lcp;
	    }

	// case mid:
	} else {
	    debug_assert!(self.lo < s && s < self.hi);

	    // FIXME: doing this in floats is lazy but
	    let s_lo = s as f64 - 0.5;
	    let s_hi = s as f64 + 0.5;
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
		    + (1.0 - s_rem).ln() // fraction
		    - self.ln_prob; // re-base
		debug_assert!(lo_bin > f64::NEG_INFINITY);
	    } else {
		self.hi = s; // new hi
		let lcp = cp.ln();
		self.ln_prob += lcp;
		lo_bin = self.bins.ln_ps[0] - lcp; // just scale
		hi_bin = s_lp // absolute lp
		    + s_rem.ln() // fraction
		    - self.ln_prob; // re-base
		debug_assert!(hi_bin > f64::NEG_INFINITY || s_rem == 0.0);
	    }

	    let mid_bin = if self.lo + 1 == self.hi {
		f64::NEG_INFINITY
	    } else {
		let lo = self.lo as f64 + 0.5;
		let hi = self.hi as f64 - 0.5;
		let mut mid_lp =
		    self.gaussian.log_probability(lo,hi);
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

/// Separate the fractional part of a number of the integer part
pub fn floor_rem(x: f64) -> (i64,f64) {
    // TODO: rewrite with trunc() and fract() (neg is special case?)
    let x_floor = x.floor();
    let x_int = x_floor as i64;
    if x.is_finite() { (x_int, ((x - x_floor))) }
    else { (x_int,(0.0)) }
}

////////////////////////////// MODELS //////////////////////////////

impl Model<i64> for Gaussian {
    fn push(&mut self, s: i64) -> Option<i64> {
	Some(s)
    }
    fn next_distr(&mut self) -> Box<dyn UnivariateDistribution> {
	Box::new(*self)
    }
}

#[derive(Debug,Clone,PartialEq,PartialOrd)]
/// Static (non-optimal) Gaussian model of a set of integers
pub struct WithReplacement {
    pub count: usize,
    pub vec: Vec<i64>,
    pub distr: Gaussian
}

impl WithReplacement {
    /// Construction from a set of values
    pub fn mle<I>(values: I) -> Self where I: Iterator<Item=i64> {
	let vec = Vec::new();
	let distr = Gaussian::mle(values, 0);
	let count = distr.s0;
	WithReplacement{ count, vec, distr }
    }
}

impl Model<Vec<i64>> for WithReplacement {
    fn push(&mut self, x: i64) -> Option<Vec<i64>> {
	self.vec.push(x);
	self.count -= 1;
	if self.count == 0 { Some(std::mem::take(&mut self.vec)) }
	else { None }
    }
    fn next_distr(&mut self) -> Box<dyn UnivariateDistribution> {
	Box::new(self.distr.clone())
    }
}

#[derive(Debug,Clone,PartialEq,PartialOrd)]
/// Updating (optimal) Gaussian model of a set of integers
pub struct WithoutReplacement {
    pub vec: Vec<i64>,
    pub distr: Gaussian // contains count as s0
}

impl WithoutReplacement {
    /// Construction from a set of values
    pub fn mle<I>(values: I) -> Self where I: Iterator<Item=i64> {
	WithoutReplacement{ vec: Vec::new(),
			    distr: Gaussian::mle(values, 0) }
    }

    pub fn count(&self) -> usize {
	self.distr.s0
    }
}

impl Model<Vec<i64>> for WithoutReplacement {
    fn push(&mut self, x: i64) -> Option<Vec<i64>> {
	self.vec.push(x);

	// update
	let s0 = self.distr.s0 - 1;
	let s1 = self.distr.s1 - x;
	let x128 = x.abs() as u128;
	let s2 = self.distr.s2 - x128*x128;

	if s0 == 0 {
	    Some(std::mem::take(&mut self.vec)) // finished
	}
	else {
	    self.distr = Gaussian::from_sums(s0,s1,s2,0);
	    None
	}
    }
    fn next_distr(&mut self) -> Box<dyn UnivariateDistribution> {
	Box::new(self.distr.clone())
    }
}
