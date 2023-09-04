use ark_relations::r1cs::SynthesisError;

#[derive(Clone, Debug)]
pub enum SaverError {
    UnexpectedBase(u8),
    InvalidDecomposition,
    SynthesisError(SynthesisError),
    AtLeastOneNonNoneRequired,
    VectorShorterThanExpected(usize, usize),
    MalformedEncryptionKey(usize, usize),
    MalformedDecryptionKey(usize, usize),
    IncompatibleEncryptionKey(usize, usize),
    IncompatibleDecryptionKey(usize, usize),
    InvalidProof,
    InvalidCommitment,
    InvalidDecryption,
    CouldNotFindDiscreteLog,
    InvalidPairingPowers,
    PairingCheckFailed,
}

impl From<SynthesisError> for SaverError {
    fn from(e: SynthesisError) -> Self {
        Self::SynthesisError(e)
    }
}

