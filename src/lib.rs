mod map;
mod special;
pub mod distribution;

use std::fmt::Debug;
use std::mem;
use std::io::{self,Write};
use std::iter::Iterator;
use std::cmp::Ordering;

/// Models produce distributions and consume their resolutions (as
/// `i64`s) until a value `T` pops out. Used by both Encoder and Decoder
/// symmetrically.
pub trait Model<T> {
    /// Push a value into the model and updates its state. Returns a
    /// values once it's been fully described by a series of indexes.
    fn push(&mut self, s: Index) -> Option<T>;
    /// Get the distribution for the next symbol from the model.
    fn next_distr(&mut self) -> Box<dyn UnivariateDistribution>;
}

/// A pair of models can model a pair of values.
// TODO: other tuples?
impl<A: Model<T>, T,
     B: Model<U>, U> Model<(T,U)> for (Option<T>,(A,B)) {
    fn push(&mut self, s: Index) -> Option<(T,U)> {
	match self.0 {
	    None => {
		self.0 = self.1.0.push(s);
		None
	    }
	    Some(_) => {
		let u = self.1.1.push(s)?;
		Some((std::mem::take(&mut self.0).unwrap(), u))
	    }
	}
    }
    fn next_distr(&mut self) -> Box<dyn UnivariateDistribution> {
	match self.0 {
	    None    => self.1.0.next_distr(),
	    Some(_) => self.1.1.next_distr()
	}
    }
}

/// A univariate distribution that can be truncated.
pub trait UnivariateDistribution: Debug {
    fn truncated(&self) -> Box<dyn TruncatedDistribution>;
}

/// A single sample from a distribution
pub struct Sample<D,T> {
    pub distr: D,
    pub resolve: Box<dyn FnMut(Index) -> T>
}

impl<D: UnivariateDistribution + Clone + 'static,
     T> Model<T> for Sample<D,T> {
    fn push(&mut self, s: Index) -> Option<T> {
	Some((self.resolve)(s)) // always produce
    }
    fn next_distr(&mut self) -> Box<dyn UnivariateDistribution> {
	Box::new(self.distr.clone())
    }
}

/// A vector of values from a distribution. MLE for best results. The
/// model is static and therefore not optimal (values don't get removed
/// from the distribution, so like a bag "with replacement")
pub struct Samples<D,T> {
    pub count: usize,
    pub vec: Vec<T>,
    pub sampl: Sample<D,T>
}

impl<D,T> Samples<D,T> {
    /// Construction
    pub fn repeatedly(sampl: Sample<D,T>, count: usize) -> Self {
	Samples{ count, vec: Vec::new(), sampl }
    }
}

impl<D: UnivariateDistribution + Clone + 'static,
     T> Model<Vec<T>> for Samples<D,T> {
    fn push(&mut self, s: Index) -> Option<Vec<T>> {
	let val = (self.sampl.resolve)(s); // always produces
	self.vec.push(val);
	self.count -= 1;
	if self.count == 0 { Some(std::mem::take(&mut self.vec)) }
	else { None }
    }
    fn next_distr(&mut self) -> Box<dyn UnivariateDistribution> {
	Box::new(self.sampl.distr.clone())
    }
}

/// Bins of a univariate distribution are indexed by `i64`s
pub type Index = i64;

/// A univariate distribution with a truncated probability mass between
/// a lower- and upper-bound.
pub trait TruncatedDistribution: Debug {
    /// Quantile (inverse CDF) and decompose of the remaining
    /// probability mass. Given `cp` $\in$ [0-1), returns the symbol
    /// index (`s: i64`) in which it falls as well as the fraction
    /// (`s_rem: f64` $\in$ [0-1)) achieved towards symbol `s+1` in the
    /// cumulative probability assigned to `s`. Indexes must increase
    /// monotonically w.r.t. `cp`, and `s_rem` must increase linearly
    /// w.r.t. `cp` inside each `s`.
    fn quantile(&self, cp: f64) -> (Index, f64); // returns (s, s_rem)
    /// Split the remaining probability mass with the given bit
    /// (false:0:left :: true:1:right) at the given cumulative
    /// probability, which we already know splits at index `s` with
    /// remainder `s_rem`.
    fn truncate(&mut self, cp: f64, s: Index, s_rem: f64, bit: bool);
    /// Return the index of the lower-bound of the interval of the
    /// truncation.
    fn lo(&self) -> Index;
    /// Return the index of the upper-bound of the interval of the
    /// truncation.
    fn hi(&self) -> Index;
    /// True iff `self.lo() == self.hi()`
    fn is_resolved(&self) -> bool { self.lo() == self.hi() }
}

/// Turns a model and a value into binary code
pub struct Encoder<'a> {
    pub head: Vec<(Index,Box<dyn TruncatedDistribution>)>, // (target, distr)
    pub tail: Box<dyn Iterator<Item=(Index,Box<dyn TruncatedDistribution>)> + 'a>,
}

impl<'a> Encoder<'a> {
    /// Construct an Encoder given a model that produces distributions
    /// and the indexes of the values in those distributions that,
    /// pushed to the model, will produce the encoded value
    pub fn new<T,M,I>(mut model: M, atoms: I) -> Encoder<'a>
    where
	M: Model<T> + 'a,
	I: Iterator<Item=Index> + 'a
    {
	Encoder{
	    head: Vec::new(),
	    tail: Box::new(
		atoms.map(move |s| {
		    let udistr = model.next_distr();
		    // println!("{:.2} bits for {} in {:?}",
		    // 	     udistr.info(s), s, udistr);
		    let tdistr = udistr.truncated();
		    model.push(s); // update
		    (s,tdistr)
		}) // filtering out resolved for 0 info special case
		   .filter(|(_,tdistr)| !tdistr.is_resolved()) // (hacky)
	    )
	}
    }
}

impl<'a> Iterator for Encoder<'a> {
    type Item = bool;
    fn next(&mut self) -> Option<bool> {

	let mut stack = Vec::new(); // recursive call stack
	let mut cp = 0.5; // a bit will split the distribution in half
	let mut mbit = None;

	let mut head_iter = mem::take(&mut self.head).into_iter();
	while let Some((target, distr)) =
	    head_iter.next().or_else(|| self.tail.next()) // chain head + tail
	{
	    // compute cursor position
	    let (s, s_rem) = distr.quantile(cp);
	    stack.push((target, distr, cp, s, s_rem));

	    match s.cmp(&target) {
		Ordering::Less => {
		    mbit = Some(true); // 1
		    break
		}

		Ordering::Equal => {
		    if s_rem > 0.0 {
			// next cp is where cp falls within s/target
			cp = s_rem;
			// bit not determined, check w/ next distr
			continue

		    } else { // edge case
			debug_assert!(s_rem == 0.0);
			mbit = Some(true); // 1
			break
		    }
		}

		Ordering::Greater => {
		    mbit = Some(false); // 0
		    break
		}
	    }
	}

	if stack.is_empty() { return None } // end
	let bit = mbit.unwrap_or(
	    // case: the cursor fell all the way to the end though all
	    // the remaining symbols in their distributions and we have
	    // to select the largest interval between left/right

	    // if cp is the correct projection of s_rem's through all
	    // distributions, then it's just
	    cp < 0.5
	);


	// split distributions on the stack
	for (target, mut distr, cp, s, s_rem) in stack {
	    distr.truncate(cp, s, s_rem, bit);
	    if !distr.is_resolved() { self.head.push((target,distr)) }
	}

	// append distributions that weren't reached
	for tc in head_iter {
	    self.head.push(tc)
	}

	Some(bit)
    }
}

/// Byte interface for `Encoder`
pub struct Encoder8<'a>(Encoder<'a>);
impl<'a> Encoder8<'a> {
    pub fn new<T,S,I>(s: S, atoms: I) -> Encoder8<'a>
    where
	S: Model<T> + 'a,
	I: Iterator<Item=Index> + 'a
    {
	Encoder8(Encoder::new::<T,S,I>(s,atoms))
    }
}

impl<'a> Iterator for Encoder8<'a> {
    type Item = u8;
    fn next(&mut self) -> Option<Self::Item> {
	let mut byte = (self.0.next()? as u8) << 7; // most significant
	for bit_position in 1..8 {
	    if self.0.next().unwrap_or(false) {
		byte |= 1 << (7 - bit_position);
	    }
	}
	// print!("{:08b}\n",byte);
	io::stdout().flush().unwrap();
	Some(byte)
    }
}

/// Turns a model and binary code into a value
pub struct Decoder<M: Model<T>, I: Iterator<Item=bool>, T> {
    // data & model
    pub code: I, // Iterator<Item=bool>
    pub model: M, // Models<T>
    t_phantom: std::marker::PhantomData<T>,

    // distribution
    pub distr: Box<dyn TruncatedDistribution>,

    // split histories
    pub lo_splits: Vec<f64>,
    pub hi_splits: Vec<f64>,
    pub last_bit: bool
}

impl<M,I,T> Decoder<M,I,T>
where
    M: Model<T>,
    I: Iterator<Item=bool>,
{
    /// Construct a Decoder for a value given a model and code
    pub fn new(mut model: M, code: I) -> Decoder<M,I,T> {
	let distr = model.next_distr().truncated(); // first distr
	let lo_splits = Vec::new();
	let hi_splits = Vec::new();
	let last_bit = false; // doesn't matter at initialization
	Decoder{ code, model, t_phantom: std::marker::PhantomData,
		 distr, lo_splits, hi_splits, last_bit }
    }

    /// Run the decoder
    pub fn decode(&mut self) -> T { loop {
	// 1) RESOLVE DISTRIBUTRION:

	// consume bits, split distribution until resolved
	let half = 0.5;
	while !self.distr.is_resolved() {

	    let (s, s_rem) = self.distr.quantile(half);
	    self.last_bit = self.code.next()
		.expect("Bitstream ended before a value was decoded");

	    if self.last_bit {
		// distr's lo gets moved up one half
		if s != self.distr.lo() { self.lo_splits = vec![] }
		self.lo_splits.push(s_rem);
	    } else {
		// distr's hi gets moved down
		if s != self.distr.hi() { self.hi_splits = vec![] }
		self.hi_splits.push(s_rem);
	    }

	    self.distr.truncate(half, s, s_rem, self.last_bit);
	}

	// resolved
	let target = self.distr.lo();
	if let Some(res) = self.model.push(target) {
	    return res // finished
	}

	// 2) SUMMON NEW DISTR; APPLY SPLIT HISTORY:
	let udistr = self.model.next_distr();
	self.distr = udistr.truncated();

	// case: /0*1/
	if self.last_bit {
	    // 0s first
	    for cp in mem::take(&mut self.hi_splits) {
		let (s, s_rem) = self.distr.quantile(cp);
		if s != self.distr.hi() { self.hi_splits = vec![] }
		self.distr.truncate(cp, s, s_rem, false); // 0
		self.hi_splits.push(s_rem);
	    }
	    // 1 last
	    debug_assert!(self.lo_splits.len() == 1);
	    let lo_cp = self.lo_splits[0];
	    let (s, s_rem) = self.distr.quantile(lo_cp);
	    self.distr.truncate(lo_cp, s, s_rem, true); // 1
	    self.lo_splits = vec![s_rem];

	// case: /1*0/
	} else {
	    // 1s first
	    for cp in mem::take(&mut self.lo_splits) {
		let (s, s_rem) = self.distr.quantile(cp);
		if s != self.distr.lo() { self.lo_splits = vec![] }
		self.distr.truncate(cp, s, s_rem, true); // 1
		self.lo_splits.push(s_rem);
	    }
	    // 0 last
	    debug_assert!(self.hi_splits.len() == 1);
	    let hi_cp = self.hi_splits[0];
	    let (s, s_rem) = self.distr.quantile(hi_cp);
	    self.distr.truncate(hi_cp, s, s_rem, false); // 0
	    self.hi_splits = vec![s_rem];
	}
    }}
}

#[derive(Clone,PartialEq,Eq)]
struct BytesToBits<I: Iterator<Item=u8>> {
    current_byte: u8,
    shift: i8,
    bytes: I,
}

impl<I> BytesToBits<I> where I: Iterator<Item=u8> {
    fn new(bytes: I) -> BytesToBits<I> {
	let current_byte = 0;
	let shift = -1;
	BytesToBits{ current_byte, shift, bytes }
    }
}

impl<I> Iterator for BytesToBits<I> where I: Iterator<Item=u8> {
    type Item = bool;
    fn next(&mut self) -> Option<Self::Item> {
	if self.shift < 0 {
	    self.current_byte = self.bytes.next()?;
	    self.shift = 7;
	}
	let bit = self.current_byte & (1 << self.shift) != 0;
	self.shift -= 1;
	Some(bit)
    }
}

/// Byte interface for `Decoder`
pub struct Decoder8< S: Model<T>, I: Iterator<Item=u8>, T>
    (Decoder<S,BytesToBits<I>,T>);

impl<S,I,T> Decoder8<S,I,T>
where
    I: Iterator<Item=u8>,
    S: Model<T>,
{
    pub fn new(model: S, bytes: I) -> Decoder8<S,I,T> {
	Decoder8(Decoder::new(model, BytesToBits::new(bytes)))
    }

    pub fn decode(&mut self) -> T { self.0.decode() }
}
