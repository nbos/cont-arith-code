use rand::{rng,Rng};
use cont_arith_code::*;
use cont_arith_code::distribution::gaussian as g;

fn main() {
    _demo();
    // _gen_data();
}

fn _gen_data() {
    for i in 1..100 {
	let xs = [vec![0;i-1],vec![1i64]].concat();
	let mle = g::WithReplacement::mle(xs.iter().copied());

	let mut bits = 0.0f64;
	let mut g = mle.clone();
	for &x in xs.iter() {
	    let info = g.distr.bits(x);
	    bits += info;
	    g.push(x);
	}

	// encode
	let code = Encoder::new(mle.clone(), xs.iter().copied())
            .collect::<Vec<_>>();

	let decoded = Decoder::new(mle.clone(),
				   code.iter().copied()).decode();
	assert_eq!(xs, decoded);

	println!("{},{},{}",i,bits,code.len());
    }
}

fn _demo() {
    let sets: Vec<Vec<i64>> = vec![
	vec![0],
	vec![1],
	vec![-1,1],
	vec![-1234,1234],
	vec![1,0,-1],
	vec![0,0,1],
	vec![0,0,0,1],
	vec![0,0,0,0,1],

	[vec![0;10], vec![1]].concat(),
	[vec![0;20], vec![1]].concat(),
	[vec![0;40], vec![1]].concat(),
	[vec![0;80], vec![1]].concat(),

	[vec![0;200], vec![1]].concat(),
	[vec![1], vec![0;200]].concat(),
	[vec![0;100], vec![123], vec![0;100]].concat(),

	_sample_uniform(3,-5e3,5e3),
	_sample_uniform(10,-5e3,5e3),
	_sample_uniform(100,-5e3,5e3),
	_sample_uniform(1000,-5e3,5e3),
    ];

    for xs in sets { _test(&xs) }
    println!("\x1b[92mTest complete!\x1b[0m");
}

/// Test a Gaussian MLE on the given dataset, encode, decode and report
/// performance
fn _test(xs: &[i64]) {
    println!("Set: {:?}", xs);

    // fit model
    let mle = g::WithReplacement::mle(xs.iter().copied());
    // let mle = g::WithoutReplacement::mle(xs.iter().copied());
    println!("Model: {:?}", mle);

    // count information w.r.t. model
    let mut bits = 0.0f64;
    let mut bitss = Vec::new();
    let mut g = mle.clone();
    for &x in xs.iter() {
	let info = g.distr.bits(x);
	bits += info;
	bitss.push(info);
	g.push(x);
    }
    println!("Information Content: {}", bits);
    println!("Expected code length: {}", bits.ceil());
    println!("Information Contributions: [{}]",
	     bitss.iter().map(|h| format!("{:.3}",h))
	     .collect::<Vec<String>>().join(", "));

    // encode
    let code = Encoder::new(mle.clone(), xs.iter().copied())
        .collect::<Vec<_>>();
    let codestring = code.iter()
	.map(|b| if *b {'1'} else {'0'})
	.collect::<String>();
    println!("Code: {}", codestring);
    println!("Code length: {} bits", code.len());

    // report stats
    let delta = code.len() - bits.ceil() as usize;
    print!("Analysis: ");
    if delta < 1 {
	print!("\x1b[92m"); // green
    } else if delta < 2 {
	print!("\x1b[33m"); // yellow
    } else {
	print!("\x1b[31m"); // red
    }

    print!("{:+.2} bits\x1b[0m ", delta);

    let relative = if delta == 0 {
	0.0 // NaN mitigation
    } else {
	delta as f64 * 100.0 / bits.ceil() };

    if relative < 1.0 {
	print!("\x1b[92m"); // green
    } else if relative < 2.0 {
	print!("\x1b[33m"); // yellow
    } else {
	print!("\x1b[31m"); // red
    }

    println!("({:+.1}%)\x1b[m compared to estimate", relative);

    // decode
    let decoded = Decoder::new(mle.clone(),
			       code.iter().copied()).decode();
    if xs == decoded {
	println!("\x1b[92mDecoding successful\x1b[0m");
    } else {
	println!("\x1b[31mDecoding unsuccessful\x1b[0m");
	dbg![&xs,&mle,&codestring,&decoded];
	panic!();
    }

    println!();
}

/// Generates a random vector of `i64` values.
///
/// # Arguments:
/// * `size` - The number of random elements to generate.
/// * `min` - The minimum value for the random range (inclusive).
/// * `max` - The maximum value for the random range (inclusive).
///
/// # Returns:
/// A `Vec<i64>` containing the random values.
fn _sample_uniform(size: usize, min: f64, max: f64) -> Vec<i64> {
    let mut rng = rng();
    (0..size).map(|_| rng.random_range(min..=max) as i64)
        .collect()
}
