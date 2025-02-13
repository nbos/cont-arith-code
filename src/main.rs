use rand::{rng,Rng};
use cont_arith_code::*;
use cont_arith_code::distribution::gaussian as g;

fn main() {
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

	rand_data(3,-50,50),
	rand_data(10,-50,50),
	rand_data(100,-50,50),
	rand_data(1000,-50,50),
    ];

    for xs in sets {
	println!("Set: {:?}", xs);

	// fit model
	// let gaussian = g::WithReplacement::mle(xs.iter().copied());
	let gaussian = g::WithoutReplacement::mle(xs.iter().copied());
	println!("Model: {{ μ: {}, σ: {} }} ({}, {}, {})",
		 gaussian.distr.mean,
		 gaussian.distr.stdev,
		 gaussian.distr.s0,
		 gaussian.distr.s1,
		 gaussian.distr.s2 );

	// count information w.r.t. model
	let mut bits = 0.0f64;
	let mut bitss = Vec::new();
	let mut g = gaussian.clone();
	for &x in xs.iter() {
	    let info = g.distr.bits(x);
	    bits += info;
	    bitss.push(info);
	    g.push(x);
	}
	println!("Expected code length: {}", bits);
	println!("Code length contributions: [{}]",
		 bitss.iter().map(|h| format!("{:.3}",h))
		 .collect::<Vec<String>>().join(", "));

	// encode
	let code = Encoder::new(gaussian.clone(), xs.iter().copied())
            .collect::<Vec<_>>();
	let codestring = code.iter()
	    .map(|b| if *b {'1'} else {'0'})
	    .collect::<String>();
	println!("Code: {}", codestring);
	println!("Code length: {} bits", code.len());

	let codelen = code.len() as f64;
	let delta = codelen - bits;
	print!("Analysis: ");
	if delta < 1.0 {
	    print!("\x1b[92m"); // green
	} else if delta < 2.0 {
	    print!("\x1b[33m"); // yellow
	} else {
	    print!("\x1b[31m"); // red
	}

	print!("{:+.2}\x1b[0m bits ", delta);

	let relative = if delta == 0.0 {
	    0.0 // NaN mitigation
	} else {
	    delta * 100.0 / codelen };

	if relative < 1.0 {
	    print!("\x1b[92m"); // green
	} else if delta < 2.0 {
	    print!("\x1b[33m"); // yellow
	} else {
	    print!("\x1b[31m"); // red
	}

	println!("({:.1}%)\x1b[m compared to estimate", relative);

	// decode
	let decoded = Decoder::new(gaussian.clone(),
				   code.iter().copied()).decode();
	if xs == decoded {
	    println!("\x1b[92mDecoding successful\x1b[0m");
	} else {
	    println!("\x1b[31mDecoding unsuccessful\x1b[0m");
	    dbg![&xs,&gaussian,&codestring,&decoded];
	    panic!();
	}

	println!();
    }

    println!("\x1b[92mTest complete!\x1b[0m");
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
fn rand_data(size: usize, min: i64, max: i64) -> Vec<i64> {
    let mut rng = rng();
    (0..size).map(|_| rng.random_range(min..=max))
        .collect()
}
