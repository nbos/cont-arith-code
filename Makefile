cont-arith-code:
	cargo build

all: cont-arith-code

doc:
	cargo doc --no-deps

release:
	cargo build --release
profile:
	RUSTFLAGS="-C debuginfo=2" cargo build --release
