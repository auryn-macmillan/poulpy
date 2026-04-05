fn main() {
    use poupy_FHE-LLM::params::Precision;
    use poupy_FHE-LLM::params::SecurityLevel;

    println!("Successfully imported poupy_FHE-LLM!");
    println!("Precision: {:?}", Precision::Int8);
    println!("SecurityLevel: {:?}", SecurityLevel::Bits100);
}
