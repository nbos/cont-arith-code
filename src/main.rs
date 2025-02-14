use rand::{rng,Rng};
use cont_arith_code::*;
use cont_arith_code::distribution::gaussian as g;

fn main() {
    _demo();
    // _gen_data();
}

fn _gen_data() {
    for i in 1..101 {
	// let xs = _sample_uniform(i,-5,5);
	let xs = _sample_normal(i,0.0,2.8867);
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
	vec![8,8,8,8,8,8,8,8,8,8,8,8],
	vec![-1,1],
	vec![-1234,1234],
	vec![1,0,-1],
	vec![0,0,1],
	vec![0,0,0,1],
	vec![0,0,0,0,1],

	[vec![0;9], vec![1]].concat(),
	[vec![0;19], vec![1]].concat(),
	[vec![0;39], vec![1]].concat(),
	[vec![0;79], vec![1]].concat(),

	[vec![0;199], vec![1]].concat(),
	[vec![1], vec![0;199]].concat(),
	[vec![0;99], vec![123], vec![0;100]].concat(),

	_sample_uniform(3,-500,500),
	_sample_uniform(10,-500,500),
	_sample_uniform(100,-500,500),
	// _sample_uniform(1000,-500,500),
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
    println!("Information Content: {} bits", bits);
    println!("Expected code length: {} bits", bits.ceil());
    println!("Information Contributions (bits): [{}]",
	     bitss.iter().map(|h| format!("{:.2}",h))
	     .collect::<Vec<String>>().join(", "));

    // encode
    let code = Encoder::new(mle.clone(), xs.iter().copied())
        .collect::<Vec<_>>();
    let codestring = code.iter()
	.map(|b| if *b {'1'} else {'0'})
	.collect::<String>();
    println!("Code: \'{}\'", codestring);
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

    println!("({:+.1}%)\x1b[m compared to expected", relative);

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

fn _sample_uniform(size: usize, min: i64, max: i64) -> Vec<i64> {
    let mut rng = rng();
    (0..size).map(|_| rng.random_range(min..=max))
        .collect()
}

fn _sample_normal(size: usize, mean: f64, std_dev: f64) -> Vec<i64> {
    let mut rng = rng(); // Create a random number generator
    (0..size)
        .map(|_| {
            let u: f64 = rng.random();
            let z = crate::distribution::gaussian::quantile(u);
            (mean + z * std_dev).round() as i64
        })
        .collect()
}
