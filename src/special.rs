use std::f64::consts::*;

use libm::expm1;
use errorfunctions::*;

/// Copied from `scipy/scipy/special/_faddeeva.cxx` which is their
/// definition of `logcdf` for `norm` (standard normal):
///
/// "Log of the CDF of the normal distribution for double x.
///
/// Let F(x) be the CDF of the standard normal distribution.  This
/// implementation of log(F(x)) is based on the identities
///
///   F(x) = erfc(-x/√2)/2
///        = 1 - erfc(x/√2)/2
///
/// We use the first formula for x < -1, with erfc(z) replaced by
/// erfcx(z)*exp(-z**2) to ensure high precision for large negative
/// values when we take the logarithm:
///
///   log F(x) = log(erfc(-x/√2)/2)
///            = log(erfcx(-x/√2)/2)*exp(-x**2/2))
///            = log(erfcx(-x/√2)/2) - x**2/2
///
/// For x >= -1, we use the second formula for F(x):
///
///   log F(x) = log(1 - erfc(x/√2)/2)
///            = log1p(-erfc(x/√2)/2)"
pub fn log_ndtr(x: f64) -> f64 {
    let t: f64 = x * FRAC_1_SQRT_2;
    if x < -1.0 {
	(0.5 * (-t).erfcx()).ln()
	    - t*t
    } else {
	(0.5 * -t.erfc()).ln_1p()
    }
}

pub fn polevl(z: f64, coeff: &[f64]) -> f64 {
    if coeff.len() == 0 {
        return 0.0;
    }

    let mut sum = coeff[0];
    for c in coeff[1..].iter() {
        sum = *c + z * sum;
    }
    sum
}

pub const NDTRI_EXP_P1: &[f64] = &[
    4.05544892305962419923,
    3.15251094599893866154e1,
    5.71628192246421288162e1,
    4.40805073893200834700e1,
    1.46849561928858024014e1,
    2.18663306850790267539,
    -1.40256079171354495875e-1,
    -3.50424626827848203418e-2,
    -8.57456785154685413611e-4
];

pub const NDTRI_EXP_Q1: &[f64] = &[
    1.0,
    1.57799883256466749731e1,
    4.53907635128879210584e1,
    4.13172038254672030440e1,
    1.50425385692907503408e1,
    2.50464946208309415979,
    -1.42182922854787788574e-1,
    -3.80806407691578277194e-2,
    -9.33259480895457427372e-4
];

pub const NDTRI_EXP_P2: &[f64] = &[
    3.23774891776946035970,
    6.91522889068984211695,
    3.93881025292474443415,
    1.33303460815807542389,
    2.01485389549179081538e-1,
    1.23716634817820021358e-2,
    3.01581553508235416007e-4,
    2.65806974686737550832e-6,
    6.23974539184983293730e-9
];

pub const NDTRI_EXP_Q2: &[f64] = &[
    1.0,
    6.02427039364742014255,
    3.67983563856160859403,
    1.37702099489081330271,
    2.16236993594496635890e-1,
    1.34204006088543189037e-2,
    3.28014464682127739104e-4,
    2.89247864745380683936e-6,
    6.79019408009981274425e-9
];

/// Copied from `scipy/scipy/special/_ndtri_exp.pxd`:
/// Return inverse of log CDF of normal distribution for very small
/// y. For p sufficiently small, the inverse of the CDF of the normal
/// distribution can be approximated to high precision as a rational
/// function in sqrt(-2.0 * log(p)).
pub fn ndtri_exp_small_y(y: f64) -> f64 {
    let x;
    if y >= -f64::MAX * 0.5 {
	x = (-2.0 * y).sqrt()
    } else {
        x = SQRT_2 * (-y).sqrt()
    }
    let x0 = x - x.ln() / x;
    let z = 1.0 / x;
    let x1;
    if x < 8.0 {
        x1 = z * polevl(z, NDTRI_EXP_P1)
	    / polevl(z, NDTRI_EXP_Q1);
    } else {
        x1 = z * polevl(z, NDTRI_EXP_P2)
	    / polevl(z, NDTRI_EXP_Q2);
    }
    x1 - x0
}

/// sqrt(2*pi)
pub const SQRT_2PI: f64 = 2.50662827463100050242E0;

/* approximation for 0 <= |y - 0.5| <= 3/8 */

pub const NDTRI_P0: &[f64] = &[
    -5.99633501014107895267E1,
    9.80010754185999661536E1,
    -5.66762857469070293439E1,
    1.39312609387279679503E1,
    -1.23916583867381258016E0,
];

pub const NDTRI_Q0: &[f64] = &[
    1.00000000000000000000E0,
    1.95448858338141759834E0,
    4.67627912898881538453E0,
    8.63602421390890590575E1,
    -2.25462687854119370527E2,
    2.00260212380060660359E2,
    -8.20372256168333339912E1,
    1.59056225126211695515E1,
    -1.18331621121330003142E0,
];

/* Approximation for interval z = sqrt(-2 log y ) between 2 and 8
 * i.e., y between exp(-2) = .135 and exp(-32) = 1.27e-14.
 */

pub const NDTRI_P1: &[f64] = &[
    4.05544892305962419923E0,
    3.15251094599893866154E1,
    5.71628192246421288162E1,
    4.40805073893200834700E1,
    1.46849561928858024014E1,
    2.18663306850790267539E0,
    -1.40256079171354495875E-1,
    -3.50424626827848203418E-2,
    -8.57456785154685413611E-4,
];

pub const NDTRI_Q1: &[f64] = &[
    1.00000000000000000000E0,
    1.57799883256466749731E1,
    4.53907635128879210584E1,
    4.13172038254672030440E1,
    1.50425385692907503408E1,
    2.50464946208309415979E0,
    -1.42182922854787788574E-1,
    -3.80806407691578277194E-2,
    -9.33259480895457427372E-4,
];

/* Approximation for interval z = sqrt(-2 log y ) between 8 and 64
 * i.e., y between exp(-32) = 1.27e-14 and exp(-2048) = 3.67e-890.
 */

pub const NDTRI_P2: &[f64] = &[
    3.23774891776946035970E0,
    6.91522889068984211695E0,
    3.93881025292474443415E0,
    1.33303460815807542389E0,
    2.01485389549179081538E-1,
    1.23716634817820021358E-2,
    3.01581553508235416007E-4,
    2.65806974686737550832E-6,
    6.23974539184983293730E-9,
];

pub const NDTRI_Q2: &[f64] = &[
    1.00000000000000000000E0,
    6.02427039364742014255E0,
    3.67983563856160859403E0,
    1.37702099489081330271E0,
    2.16236993594496635890E-1,
    1.34204006088543189037E-2,
    3.28014464682127739104E-4,
    2.89247864745380683936E-6,
    6.79019408009981274425E-9,
];

/// Copied from `scipy/scipy/special/cephes/ndtri.c` which is
/// their implementation of the quantile function for the standard
/// normal:
/// "Returns the argument, x, for which the area under the Gaussian
/// probability density function (integrated from minus infinity to x)
/// is equal to y.
///
/// For small arguments 0 < y < exp(-2), the program computes z = sqrt(
/// -2.0 * log(y) ); then the approximation is x = z - log(z)/z - (1/z)
/// P(1/z) / Q(1/z).  There are two rational functions P/Q, one for 0 <
/// y < exp(-32) and the other for y up to exp(-2).  For larger
/// arguments, w = y - 0.5, and x/sqrt(2pi) = w + w**3 R(w**2)/S(w**2)).
///
/// ACCURACY:
///                      Relative error:
/// arithmetic   domain        # trials      peak         rms
///    IEEE     0.125, 1        20000       7.2e-16     1.3e-16
///    IEEE     3e-308, 0.135   50000       4.6e-16     9.8e-17
///
/// ERROR MESSAGES:
///
///   message         condition    value returned
/// ndtri domain       x < 0        NAN
/// ndtri domain       x > 1        NAN
///
/// Cephes Math Library Release 2.1:  January, 1989
/// Copyright 1984, 1987, 1989 by Stephen L. Moshier
/// Direct inquiries to 30 Frost Street, Cambridge, MA 02140"
pub fn ndtri(y0: f64) -> f64 {
    if y0 <= 0.0 {
	return f64::NEG_INFINITY;
    }
    if y0 >= 1.0 {
	return f64::INFINITY;
    }
    let mut code = 1;
    let mut y = y0;
    if y > (1.0 - 0.13533528323661269189) { // 0.135... = exp(-2)
	y = 1.0 - y;
	code = 0;
    }

    if y > 0.13533528323661269189 {
	y = y - 0.5;
	let y2 = y * y;
	let mut x = y + y * (y2 * polevl(y2, NDTRI_P0)
			     / polevl(y2, NDTRI_Q0));
	x *= SQRT_2PI;
	return x;
    }
    let x = (-2.0 * y.ln()).sqrt();
    let x0 = x - x.ln() / x;

    let z = 1.0 / x;
    let x1;
    if x < 8.0 { // y > exp(-32) = 1.2664165549e-14
	x1 = z * polevl(z, NDTRI_P1)
	    / polevl(z, NDTRI_Q1);
    } else {
	x1 = z * polevl(z, NDTRI_P2)
	    / polevl(z, NDTRI_Q2);
    }
    let mut x = x0 - x1;
    if code != 0 {
	x = -x;
    }
    return x;
}


/// log1p(-exp(-2))
pub const LOG1P_MEXP_MTWO: f64 = -0.14541345786885906;

/// Copied from `scipy/scipy/special/_ndtri_exp.pxd` which is the
/// inverse of `log_ndtr`:
/// "Implementation of the inverse of the logarithm of the CDF of the
/// standard normal distribution.
///
/// Copyright: Albert Steppi
///
/// Distributed under the same license as SciPy
///
///
/// Implementation Overview
///
/// The inverse of the CDF of the standard normal distribution is
/// available in scipy through the Cephes Math Library where it is
/// called ndtri.  We call our implementation of the inverse of the log
/// CDF ndtri_exp.  For -2 <= y <= log(1 - exp(-2)), ndtri_exp is
/// computed as ndtri(exp(y)).
///
/// For 0 < p < exp(-2), the cephes implementation of ndtri uses an
/// approximation for ndtri(p) which is a function of z = sqrt(-2.0 *
/// log(p)). Letting y = log(p), for y < -2, ndtri_exp uses this
/// approximation in log(p) directly.  This allows the implementation to
/// achieve high precision for very small y, whereas ndtri(exp(y))
/// evaluates to infinity. This is because exp(y) underflows for y < ~
/// -745.1.
///
/// When p > 1 - exp(-2), the Cephes implementation of ndtri uses the
/// symmetry of the normal distribution and calculates ndtri(p) as
/// -ndtri(1 - p) allowing for the use of the same approximation. When y
/// > log(1 - exp(-2)) this implementation calculates ndtri_exp as
/// -ndtri(-expm1(y)).
///
/// Accuracy
///
/// Cephes provides the following relative error estimates for ndtri
///                      Relative error:
/// arithmetic   domain        # trials      peak         rms
///    IEEE     0.125, 1        20000       7.2e-16     1.3e-16
///    IEEE     3e-308, 0.135   50000       4.6e-16     9.8e-17
///
/// When y < -2, ndtri_exp must have relative error at least as small as
/// the Cephes implementation of ndtri for p < exp(-2). It relies on the
/// same approximation but does not have to lose precision by passing
/// from p to log(p) before applying the approximation.
///
/// Relative error of ndtri for values of the argument p near 1 can be
/// much higher than claimed by the above chart. For p near 1, symmetry
/// is exploited to replace the calculation of ndtri(p) with -ndtri(1 -
/// p). The inverse of the normal CDF increases so rapidly near the
/// endpoints of [0, 1] that the loss of precision incurred by the
/// subtraction 1 - p due to limitations in binary approximation can
/// make a significant difference in the results. Using version 9.3.0
/// targeting x86_64-linux-gnu we've observed the following
///
///                                               Estimated Relative Error
///  ndtri(1e-8)      = -5.612001244174789        ''
/// -ndtri(1 - 1e-8)  = -5.612001243305505        1.55e-10
///  ndtri(1e-16)     = -8.222082216130435        ''
/// -ndtri(1 - 1e-16) = -8.209536151601387        0.0015
///
/// If expm1 is correctly rounded for y in [log(1 - exp(-2), 0), then
/// ndtri_exp(y) should have the same relative error as ndtri(p) for p >
/// 1 - exp(-2). As seen above, this error may be higher than
/// desired. IEEE-754 provides no guarantee on the accuracy of expm1
/// however, therefore accuracy of ndtri_exp in this range is platform
/// dependent.
///
/// The case
///
///     -2 <= y <= log(1 - exp(-2)) ~ -0.1454
///
/// corresponds to
///
///      ~ 0.135 <= p <= ~ 0.865
///
/// The derivative of ndtri is sqrt(2 * pi) * exp(ndtri(x)**2 / 2).  It
/// is ~4.597 at x ~ 0.135, decreases monotonically to sqrt(2 * pi) ~
/// 2.507 at x = 0 and increases monotonically again to ~4.597 at x ~
/// 0.865.
///
/// It can be checked that all higher derivatives follow a similar
/// pattern.  Their absolute value takes on a maximum (for this
/// interval) at x ~ 0.135, decrease to a minimum at x = 0 and increases
/// to the same maximum at x ~ 0.865.  Derivatives of all orders are
/// positive at x=log(1 - exp(-2)). Thus the worst possible loss of
/// precision of ndtri(exp(x)) in the interval [0, log(1 - exp(-2))] due
/// to error in calculating exp(x) must occur near x=log(1 -
/// exp(-2)). By symmetry, the worst possible loss of precision in [-2,
/// log(1 - exp(-2)] must occur near the endpoints. We may observe
/// empirically that error at the endpoints due to exp is not
/// substantial.  Assuming that exp is accurate within +-ULP (unit of
/// least precision), we observed a value of at most ~6.0474e-16 for
///
///     abs(ndtri(x + epsilon) - ndtri(x))
///
/// if x is near exp(-2) or 1 - exp(-2) and epsilon is equal to the unit
/// of least precision of x.
///
/// (IEEE-754 provides no guarantee on the accuracy of exp, but for most
/// compilers on most architectures an assumption of +-ULP should be
/// reasonable.)
///
/// The error here is on the order of the error in the Cephes
/// implementation of ndtri itself, leading to an error profile that is
/// still favorable."
pub fn ndtri_exp(y: f64) -> f64 {
    if y < -f64::MAX { f64::NEG_INFINITY }
    else if y < -2.0 { ndtri_exp_small_y(y) }
    else if y > LOG1P_MEXP_MTWO { -ndtri(-expm1(y)) }
    else { ndtri(y.exp()) }
}

/// Copied from
/// `JuliaStats/LogExpFunctions.jl/src/basifuns.jl`:`log_sub_exp`:
/// Return `log(abs(exp(x) - exp(y)))`, preserving numerical accuracy
pub fn ln_sub_exp(x: f64, y: f64) -> f64 {
    let delta = {
	if x == y && (x.is_finite() || x < 0.0) {
	    // ensures that `delta = 0` if `x = y = -inf` (but not for
	    // `x = y = +inf`)
	    0.0
	} else {
	    (x - y).abs()
	}
    };
    f64::max(x,y) + ln1mexp(-delta)
}

pub const LOGHALF: f64 = -LN_2; // ln(1/2)

/// Copied from
/// `JuliaStats/LogExpFunctions.jl/src/basifuns.jl`:`log1mexp`:
/// Return `log(1 - exp(x))`. Modified from Martin Maechler (2012)
/// [“Accurately Computing log(1 − exp(−
/// |a|))”](http://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf)
pub fn ln1mexp(x:f64) -> f64 {
    if x < LOGHALF {
	(-x.exp()).ln_1p()
    } else {
	(-expm1(x)).ln()
    }
}
