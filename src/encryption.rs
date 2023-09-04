//! Encryption, decryption, verifying commitment
use crate::{
    circuit::BitsizeCheckCircuit,
    error::SaverError,
    keygen::{EncryptionKey, FEncryptionKeyDest, SecretKeyST, PreparedDecryptionKey, PreparedEncryptionKey, SecretKey},
    saver_groth16, 
    setup::{PreparedEncryptionGens, EncryptionGens},
    utils,
    serde_utils::*
};
use ark_ec::{
    pairing::{Pairing, PairingOutput},
    AffineRepr, CurveGroup, VariableBaseMSM,
};
use ark_ff::{PrimeField, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{
    cfg_into_iter, cfg_iter,
    marker::PhantomData,
    ops::{Add, Mul, Neg, Sub},
    rand::RngCore,
    vec,
    vec::Vec,
    UniformRand,
};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

use crate::utils::CHUNK_TYPE;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Ciphertext used with Groth16
#[serde_as]
#[derive(
    Clone, PartialEq, Eq, Debug, CanonicalSerialize, CanonicalDeserialize, Serialize, Deserialize,
)]
pub struct Ciphertext<E: Pairing> {
    #[serde_as(as = "ArkObjectBytes")]
    pub X_r: E::G1Affine,
    #[serde_as(as = "Vec<ArkObjectBytes>")]
    pub enc_chunks: Vec<E::G1Affine>,
    #[serde_as(as = "ArkObjectBytes")]
    pub commitment: E::G1Affine,
}

macro_rules! impl_enc_funcs {
    () => {
        /// Decrypt this ciphertext returning the plaintext and commitment to randomness
        pub fn decrypt(
            &self,
            sk: &SecretKey<E::ScalarField>,
            dk: impl Into<PreparedDecryptionKey<E>>,
            g_i: &[E::G1Affine],
            chunk_bit_size: u8,
        ) -> crate::Result<(E::ScalarField, E::G1Affine)> {
            Encryption::decrypt(&self.X_r, &self.enc_chunks, sk, dk, g_i, chunk_bit_size)
        }

        /// Same as `Self::decrypt` but takes pairing powers (see `PreparedDecryptionKey::pairing_powers`)
        /// that can be precomputed for faster decryption
        pub fn decrypt_given_pairing_powers(
            &self,
            sk: &SecretKey<E::ScalarField>,
            dk: impl Into<PreparedDecryptionKey<E>>,
            g_i: &[E::G1Affine],
            chunk_bit_size: u8,
            pairing_powers: &[Vec<PairingOutput<E>>],
        ) -> crate::Result<(E::ScalarField, E::G1Affine)> {
            Encryption::decrypt_given_pairing_powers(
                &self.X_r,
                &self.enc_chunks,
                sk,
                dk,
                g_i,
                chunk_bit_size,
                pairing_powers,
            )
        }

        /// Verify that the ciphertext correctly commits to the message
        pub fn verify_commitment(
            &self,
            ek: impl Into<PreparedEncryptionKey<E>>,
            gens: impl Into<PreparedEncryptionGens<E>>,
        ) -> crate::Result<()> {
            Encryption::verify_ciphertext_commitment(
                &self.X_r,
                &self.enc_chunks,
                &self.commitment,
                ek,
                gens,
            )
        }
    };
}

pub struct Encryption<E: Pairing>(PhantomData<E>);

impl<E: Pairing> Encryption<E> {
    /// Encrypt a message `m` in exponent-Elgamal after breaking it into chunks of `chunk_bit_size` bits.
    /// Returns the ciphertext, commitment and randomness created for encryption. This is "Enc" from algorithm
    /// 2 in the paper
    /// Ciphertext vector contains commitment `psi` as the last element
    pub fn encrypt<R: RngCore>(
        rng: &mut R,
        message: &E::ScalarField,
        ek: &EncryptionKey<E>,
        g_i: &[E::G1Affine],
        chunk_bit_size: u8,
    ) -> crate::Result<(Ciphertext<E>, E::ScalarField)> {
        let decomposed = utils::decompose(message, chunk_bit_size)?;
        let (mut ct, r) = Self::encrypt_decomposed_message(rng, decomposed, ek, g_i)?;
        Ok((
            Ciphertext {
                X_r: ct.remove(0),
                commitment: ct.remove(ct.len() - 1),
                enc_chunks: ct,
            },
            r,
        ))
    }

    /// Return the encryption and Groth16 proof
    pub fn encrypt_with_proof<R: RngCore>(
        rng: &mut R,
        message: &E::ScalarField,
        ek: &EncryptionKey<E>,
        snark_pk: &saver_groth16::ProvingKey<E>,
        chunk_bit_size: u8,
    ) -> crate::Result<(Ciphertext<E>, E::ScalarField, ark_groth16::Proof<E>)> {
        let g_i = saver_groth16::get_gs_for_encryption(&snark_pk.pk.vk);
        let (ct, r) = Encryption::encrypt(rng, message, ek, g_i, chunk_bit_size)?;
        let decomposed_message = utils::decompose(message, chunk_bit_size)?
            .into_iter()
            .map(|m| E::ScalarField::from(m as u64))
            .collect::<Vec<_>>();
        let circuit =
            BitsizeCheckCircuit::new(chunk_bit_size, None, Some(decomposed_message), true);
        let proof = saver_groth16::create_proof(circuit, &r, snark_pk, ek, rng)?;
        Ok((ct, r, proof))
    }

    /// Same as `Self::encrypt` but takes the SNARK verification key instead of the generators used for Elgamal encryption
    pub fn encrypt_given_snark_vk<R: RngCore>(
        rng: &mut R,
        message: &E::ScalarField,
        ek: &EncryptionKey<E>,
        snark_vk: &ark_groth16::VerifyingKey<E>,
        chunk_bit_size: u8,
    ) -> crate::Result<(Ciphertext<E>, E::ScalarField)> {
        let g_i = saver_groth16::get_gs_for_encryption(snark_vk);
        Self::encrypt(rng, message, ek, g_i, chunk_bit_size)
    }

    /// Decrypt the given ciphertext and return the message and a "commitment" to randomness to help in
    /// verifying the decryption without knowledge of secret key. This is "Dec" from algorithm 2 in the paper
    pub fn decrypt(
        c_0: &E::G1Affine,
        c: &[E::G1Affine],
        sk: &SecretKey<E::ScalarField>,
        dk: impl Into<PreparedDecryptionKey<E>>,
        g_i: &[E::G1Affine],
        chunk_bit_size: u8,
    ) -> crate::Result<(E::ScalarField, E::G1Affine)> {
        let (chunks, nu) = Self::decrypt_to_chunks(c_0, c, sk, dk, g_i, chunk_bit_size)?;
        Ok((utils::compose(&chunks, chunk_bit_size)?, nu))
    }

    /// Same as `Self::decrypt` but expects pairing powers (see `PreparedDecryptionKey::pairing_powers`)
    /// that can be precomputed for even faster decryption
    pub fn decrypt_given_pairing_powers(
        c_0: &E::G1Affine,
        c: &[E::G1Affine],
        sk: &SecretKey<E::ScalarField>,
        dk: impl Into<PreparedDecryptionKey<E>>,
        g_i: &[E::G1Affine],
        chunk_bit_size: u8,
        pairing_powers: &[Vec<PairingOutput<E>>],
    ) -> crate::Result<(E::ScalarField, E::G1Affine)> {
        let (chunks, nu) = Self::decrypt_to_chunks_given_pairing_powers(
            c_0,
            c,
            sk,
            dk,
            g_i,
            chunk_bit_size,
            Some(pairing_powers),
        )?;
        Ok((utils::compose(&chunks, chunk_bit_size)?, nu))
    }

    /// Same as `Self::decrypt` but takes Groth16's verification key instead of the generators used for Elgamal encryption
    pub fn decrypt_given_groth16_vk(
        c_0: &E::G1Affine,
        c: &[E::G1Affine],
        sk: &SecretKey<E::ScalarField>,
        dk: impl Into<PreparedDecryptionKey<E>>,
        snark_vk: &ark_groth16::VerifyingKey<E>,
        chunk_bit_size: u8,
    ) -> crate::Result<(E::ScalarField, E::G1Affine)> {
        let g_i = saver_groth16::get_gs_for_encryption(snark_vk);
        Self::decrypt(c_0, c, sk, dk, g_i, chunk_bit_size)
    }

    /// Same as `Self::decrypt` but takes Groth16's verification key and the
    /// precomputed pairing powers
    pub fn decrypt_given_groth16_vk_and_pairing_powers(
        c_0: &E::G1Affine,
        c: &[E::G1Affine],
        sk: &SecretKey<E::ScalarField>,
        dk: impl Into<PreparedDecryptionKey<E>>,
        snark_vk: &ark_groth16::VerifyingKey<E>,
        chunk_bit_size: u8,
        pairing_powers: &[Vec<PairingOutput<E>>],
    ) -> crate::Result<(E::ScalarField, E::G1Affine)> {
        let g_i = saver_groth16::get_gs_for_encryption(snark_vk);
        Self::decrypt_given_pairing_powers(c_0, c, sk, dk, g_i, chunk_bit_size, pairing_powers)
    }

    /// Verify that commitment created during encryption opens to the message chunk
    /// Check `e(c_0, Z_0) * e(c_1, Z_1) * ... * e(c_n, Z_n)` mentioned in "Verify_Enc" in algorithm 2
    pub fn verify_ciphertext_commitment(
        c_0: &E::G1Affine,
        c: &[E::G1Affine],
        commitment: &E::G1Affine,
        ek: impl Into<PreparedEncryptionKey<E>>,
        gens: impl Into<PreparedEncryptionGens<E>>,
    ) -> crate::Result<()> {
        let ek = ek.into();
        let gens = gens.into();
        let expected_count = ek.supported_chunks_count()? as usize;
        if c.len() != expected_count {
            return Err(SaverError::IncompatibleEncryptionKey(
                c.len(),
                expected_count,
            ));
        }

        let (a, b) = (
            Self::get_g1_for_ciphertext_commitment_pairing_checks(c_0, c, commitment),
            Self::get_g2_for_ciphertext_commitment_pairing_checks(&ek, &gens),
        );
        if E::multi_pairing(a, b).is_zero() {
            Ok(())
        } else {
            Err(SaverError::InvalidCommitment)
        }
    }

    pub fn verify_commitments_in_batch(
        ciphertexts: &[Ciphertext<E>],
        r_powers: &[E::ScalarField],
        ek: impl Into<PreparedEncryptionKey<E>>,
        gens: impl Into<PreparedEncryptionGens<E>>,
    ) -> crate::Result<()> {
        assert_eq!(r_powers.len(), ciphertexts.len());
        let ek = ek.into();
        let gens = gens.into();
        let expected_count = ek.supported_chunks_count()? as usize;
        for c in ciphertexts {
            if c.enc_chunks.len() != expected_count {
                return Err(SaverError::IncompatibleEncryptionKey(
                    c.enc_chunks.len(),
                    expected_count,
                ));
            }
        }

        let a =
            Self::get_g1_for_ciphertext_commitments_in_batch_pairing_checks(ciphertexts, r_powers);
        let b = Self::get_g2_for_ciphertext_commitment_pairing_checks(&ek, &gens);
        if E::multi_pairing(a, b).is_zero() {
            Ok(())
        } else {
            Err(SaverError::InvalidCommitment)
        }
    }

    /// Decrypt the ciphertext and return each chunk and "commitment" to the randomness
    pub fn decrypt_to_chunks(
        c_0: &E::G1Affine,
        c: &[E::G1Affine],
        sk: &SecretKey<E::ScalarField>,
        dk: impl Into<PreparedDecryptionKey<E>>,
        g_i: &[E::G1Affine],
        chunk_bit_size: u8,
    ) -> crate::Result<(Vec<CHUNK_TYPE>, E::G1Affine)> {
        Self::decrypt_to_chunks_given_pairing_powers(c_0, c, sk, dk, g_i, chunk_bit_size, None)
    }

    /// Decrypt the ciphertext and return each chunk and "commitment" to the randomness.
    /// Same as `Self::decrypt_to_chunks` but takes decryption key and precomputed pairing
    /// powers
    pub fn decrypt_to_chunks_given_pairing_powers(
        c_0: &E::G1Affine,
        c: &[E::G1Affine],
        sk: &SecretKey<E::ScalarField>,
        dk: impl Into<PreparedDecryptionKey<E>>,
        g_i: &[E::G1Affine],
        chunk_bit_size: u8,
        pairing_powers: Option<&[Vec<PairingOutput<E>>]>,
    ) -> crate::Result<(Vec<CHUNK_TYPE>, E::G1Affine)> {
        let dk = dk.into();
        let n = c.len();
        if n != dk.supported_chunks_count()? as usize {
            return Err(SaverError::IncompatibleDecryptionKey(
                n,
                dk.supported_chunks_count()? as usize,
            ));
        }
        if n > g_i.len() {
            return Err(SaverError::VectorShorterThanExpected(n, g_i.len()));
        }
        // c_0 * -rho
        let c_0_rho = c_0.mul_bigint((-sk.0).into_bigint());
        let c_0_rho_prepared = E::G1Prepared::from(c_0_rho.into_affine());
        let mut decrypted_chunks = vec![];
        let chunk_max_val: u32 = (1 << chunk_bit_size) - 1;
        let pairing_powers = if let Some(p) = pairing_powers { p } else { &[] };
        for i in 0..n {
            let p = E::multi_pairing(
                [c[i].into(), c_0_rho_prepared.clone()],
                [dk.V_2[i].clone(), dk.V_1[i].clone()],
            );
            if p.is_zero() {
                decrypted_chunks.push(0);
                continue;
            }

            if pairing_powers.is_empty() {
                // Precomputed powers are not provided, compute the necessary pairings
                let g_i_v_i = E::pairing(E::G1Prepared::from(g_i[i]), dk.V_2[i].clone());
                decrypted_chunks.push(Self::solve_discrete_log(
                    chunk_max_val as CHUNK_TYPE,
                    g_i_v_i,
                    p,
                )?);
            } else {
                decrypted_chunks.push(Self::solve_discrete_log_using_pairing_powers(
                    i,
                    chunk_max_val as CHUNK_TYPE,
                    p,
                    pairing_powers,
                )?);
            }
        }
        Ok((decrypted_chunks, (-c_0_rho).into_affine()))
    }

    /// Encrypt once the message has been broken into chunks
    pub fn encrypt_decomposed_message<R: RngCore>(
        rng: &mut R,
        message_chunks: Vec<CHUNK_TYPE>,
        ek: &EncryptionKey<E>,
        g_i: &[E::G1Affine],
    ) -> crate::Result<(Vec<E::G1Affine>, E::ScalarField)> {
        let expected_count = ek.supported_chunks_count()? as usize;
        if message_chunks.len() != expected_count {
            return Err(SaverError::IncompatibleEncryptionKey(
                message_chunks.len(),
                expected_count,
            ));
        }
        if message_chunks.len() > g_i.len() {
            return Err(SaverError::VectorShorterThanExpected(
                message_chunks.len(),
                g_i.len(),
            ));
        }
        let r = E::ScalarField::rand(rng);
        let r_repr = r.into_bigint();
        let mut ct = vec![];
        ct.push(ek.X_0.mul_bigint(r_repr));
        let mut m = cfg_into_iter!(message_chunks)
            .map(|m_i| <E::ScalarField as PrimeField>::BigInt::from(m_i as u64))
            .collect::<Vec<_>>();
        for i in 0..ek.X.len() {
            ct.push(ek.X[i].mul_bigint(r_repr).add(g_i[i].mul_bigint(m[i])));
        }

        // Commit to the message chunks with randomness `r`
        m.push(r.into_bigint());
        let psi = E::G1::msm_bigint(&ek.commitment_key(), &m);

        ct.push(psi);
        Ok((E::G1::normalize_batch(&ct), r))
    }

    /// generate the re-encrypt share w and randomness g_r
    pub fn re_encrypt_share<R: RngCore>(
        rng: &mut R,
        gens: &EncryptionGens<E>,
        c_0: &E::G1Affine,
        c: &[E::G1Affine],
        ekd: &FEncryptionKeyDest<E>,
        sk: &SecretKeyST<E>,
    ) -> crate::Result<(Vec<E::G1Affine>, E::G1Affine)> {
        
        let n = c.len();
        // r
        let rho = E::ScalarField::rand(rng);
        // g * r
        let g_r = gens.G.mul_bigint(rho.into_bigint()).into_affine();

        let mut w = vec![];
        // w = C_0 * (-s) + pk * r
        for i in 0..n {
            w.push((c_0.mul_bigint((-sk.s[i]).into_bigint()).add(ekd.pk.mul_bigint(rho.into_bigint()))).into_affine());
        }

        Ok((w,g_r))
    }

    /// combine2 shares of w and g_r  
    pub fn combine_re_encrypt_share(
        g_r1: &E::G1Affine,
        g_r2: &E::G1Affine,
        w1: &[E::G1Affine],
        w2: &[E::G1Affine],
    ) -> crate::Result<(Vec<E::G1Affine>, E::G1Affine)> {
        
        let mut w_combined = vec![];
        // w_combined = w1 + w2
        for i in 0..w1.len(){
            w_combined.push(w1[i].add(w2[i]).into_affine());
        }
        // gr_combined = g_r1 + g_r2
        let mut gr_combined = g_r1.add(g_r2).into_affine();
        
        Ok((w_combined,gr_combined))
    }

    /// re-encrypt by making g_r as c[0] and adding w to c[i], i=1..n
    pub fn combine_ct_with_w(
        gens: &EncryptionGens<E>,
        c: &[E::G1Affine],
        g_r: &E::G1Affine,
        w: &[E::G1Affine],
    ) -> crate::Result<Vec<E::G1Affine>> {
        
        let n = c.len();

        let mut ct = vec![];
        // c[0] = g_r
        ct.push(g_r.clone());
        // c[i] = c[i] + w[i]
        for i in 0..n {
            ct.push(c[i].add(w[i]).into_affine());
        }
        Ok(ct)
    }

    /// add two ciphertexts 
    pub fn add_cts(
        ct1: &[E::G1Affine],
        ct2: &[E::G1Affine],
    ) -> crate::Result<Vec<E::G1Affine>> {
   
        let n = ct1.len();

        let mut ct = vec![];

        for i in 0..n {
            ct.push(ct1[i].add(ct2[i]).into_affine());
        }

        Ok(ct)
    }

    /// decrypt ct after re-encryption, takes sk from the destination party
    pub fn decrypt_to_chunks_after_re_encryption(
        c_0: &E::G1Affine,
        c: &[E::G1Affine],
        sk: &SecretKey<E::ScalarField>,
        g_i: &[E::G1Affine],
        chunk_bit_size: u8,
    ) -> crate::Result<Vec<CHUNK_TYPE>> {
        let n = c.len();
        if n > g_i.len() {
            return Err(SaverError::VectorShorterThanExpected(n, g_i.len()));
        }
        // c_0 * -sk
        let c_0_rho = c_0.mul_bigint((-sk.0).into_bigint()).into_affine();
        let gi_mi = (0..n)
        .map(|i| c[i].add(c_0_rho).into_affine())
        .collect::<Vec<_>>();

        let mut decrypted_chunks = vec![];
        let chunk_max_val: u32 = (1 << chunk_bit_size) - 1;
        for i in 0..n {
            if gi_mi[i].is_zero() {
                decrypted_chunks.push(0);
                continue;
            }
            decrypted_chunks.push(Self::solve_discrete_log_for_g(
                chunk_max_val as CHUNK_TYPE,
                &g_i[i],
                &gi_mi[i],
            )?)
        }
        Ok(decrypted_chunks)
    }

    /// Does not use precomputation
    fn solve_discrete_log_for_g(
        chunk_max_val: CHUNK_TYPE,
        g_i_v_i: &E::G1Affine,
        p: &E::G1Affine,
    ) -> crate::Result<CHUNK_TYPE> {
        if p == g_i_v_i {
            return Ok(1);
        }
        let mut cur = g_i_v_i.clone();
        for j in 2..=chunk_max_val {
            cur = cur.add(g_i_v_i).into_affine();
            if cur == *p {
                return Ok(j);
            }
        }
        Err(SaverError::CouldNotFindDiscreteLog)
    }

    /// Does not use precomputation
    fn solve_discrete_log(
        chunk_max_val: CHUNK_TYPE,
        g_i_v_i: PairingOutput<E>,
        p: PairingOutput<E>,
    ) -> crate::Result<CHUNK_TYPE> {
        if p == g_i_v_i {
            return Ok(1);
        }
        let mut cur = g_i_v_i;
        for j in 2..=chunk_max_val {
            cur += g_i_v_i;
            if cur == p {
                return Ok(j);
            }
        }
        Err(SaverError::CouldNotFindDiscreteLog)
    }

    /// Relies on precomputation
    fn solve_discrete_log_using_pairing_powers(
        chunk_index: usize,
        chunk_max_val: CHUNK_TYPE,
        p: PairingOutput<E>,
        pairing_powers: &[Vec<PairingOutput<E>>],
    ) -> crate::Result<CHUNK_TYPE> {
        if pairing_powers.len() < chunk_index {
            return Err(SaverError::InvalidPairingPowers);
        }
        for j in 1..=chunk_max_val {
            let j = j as usize - 1;
            if pairing_powers[chunk_index].len() < j {
                return Err(SaverError::InvalidPairingPowers);
            }
            if pairing_powers[chunk_index][j] == p {
                return Ok(j as CHUNK_TYPE + 1);
            }
        }
        Err(SaverError::CouldNotFindDiscreteLog)
    }

    pub fn get_g1_for_ciphertext_commitment_pairing_checks(
        c_0: &E::G1Affine,
        c: &[E::G1Affine],
        commitment: &E::G1Affine,
    ) -> Vec<E::G1Affine> {
        let mut a = Vec::with_capacity(c.len() + 2);
        a.push(*c_0);
        a.extend_from_slice(c);
        a.push(commitment.into_group().neg().into_affine());
        a
    }

    pub fn get_g1_for_ciphertext_commitments_in_batch_pairing_checks(
        ciphertexts: &[Ciphertext<E>],
        r_powers: &[E::ScalarField],
    ) -> Vec<E::G1Affine> {
        let mut a = Vec::with_capacity(ciphertexts[0].enc_chunks.len() + 2);
        let num = r_powers.len();
        let r_powers_repr = cfg_iter!(r_powers)
            .map(|r| r.into_bigint())
            .collect::<Vec<_>>();

        let mut bases = vec![];
        for i in 0..num {
            bases.push(ciphertexts[i].X_r);
        }
        a.push(E::G1::msm_bigint(&bases, &r_powers_repr));

        for j in 0..ciphertexts[0].enc_chunks.len() {
            let mut bases = vec![];
            for i in 0..num {
                bases.push(ciphertexts[i].enc_chunks[j]);
            }
            a.push(E::G1::msm_bigint(&bases, &r_powers_repr));
        }

        let mut bases = vec![];
        for i in 0..num {
            bases.push(ciphertexts[i].commitment);
        }
        a.push(E::G1::msm_bigint(&bases, &r_powers_repr).neg());
        E::G1::normalize_batch(&a)
    }

    pub fn get_g2_for_ciphertext_commitment_pairing_checks(
        ek: &PreparedEncryptionKey<E>,
        gens: &PreparedEncryptionGens<E>,
    ) -> Vec<E::G2Prepared> {
        let mut b = Vec::with_capacity(ek.Z.len() + 1);
        b.push(ek.Z[0].clone());
        for i in 1..ek.Z.len() {
            b.push(ek.Z[i].clone());
        }
        b.push(gens.H.clone());
        b
    }
}

impl<E: Pairing> Ciphertext<E> {
    impl_enc_funcs!();

    /// Verify ciphertext commitment and snark proof
    pub fn verify_commitment_and_proof(
        &self,
        proof: &ark_groth16::Proof<E>,
        snark_vk: &ark_groth16::PreparedVerifyingKey<E>,
        ek: impl Into<PreparedEncryptionKey<E>>,
        gens: impl Into<PreparedEncryptionGens<E>>,
    ) -> crate::Result<()> {
        self.verify_commitment(ek, gens)?;
        saver_groth16::verify_proof(snark_vk, proof, self)
    }

    pub fn decrypt_given_groth16_vk(
        &self,
        sk: &SecretKey<E::ScalarField>,
        dk: impl Into<PreparedDecryptionKey<E>>,
        snark_vk: &ark_groth16::VerifyingKey<E>,
        chunk_bit_size: u8,
    ) -> crate::Result<(E::ScalarField, E::G1Affine)> {
        let g_i = saver_groth16::get_gs_for_encryption(snark_vk);
        self.decrypt(sk, dk, g_i, chunk_bit_size)
    }

    pub fn decrypt_given_groth16_vk_and_pairing_powers(
        &self,
        sk: &SecretKey<E::ScalarField>,
        dk: impl Into<PreparedDecryptionKey<E>>,
        snark_vk: &ark_groth16::VerifyingKey<E>,
        chunk_bit_size: u8,
        pairing_powers: &[Vec<PairingOutput<E>>],
    ) -> crate::Result<(E::ScalarField, E::G1Affine)> {
        let g_i = saver_groth16::get_gs_for_encryption(snark_vk);
        self.decrypt_given_pairing_powers(sk, dk, g_i, chunk_bit_size, pairing_powers)
    }
}


#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    use crate::{
        keygen::{keygen, FkeygenPartial, combineEKs,combineP1share,genP1share,keygenFinal, Fkeygen_dest,DecryptionKey},
        setup::EncryptionGens,
        utils::{chunks_count, decompose},
    };
    use ark_bls12_381::Bls12_381;
    use ark_ff::One;
    use ark_std::rand::{prelude::StdRng, SeedableRng};

    type Fr = <Bls12_381 as Pairing>::ScalarField;

    pub fn enc_setup<R: RngCore>(
        chunk_bit_size: u8,
        rng: &mut R,
    ) -> (
        EncryptionGens<Bls12_381>,
        Vec<<Bls12_381 as Pairing>::G1Affine>,
        SecretKey<<Bls12_381 as Pairing>::ScalarField>,
        EncryptionKey<Bls12_381>,
        DecryptionKey<Bls12_381>,
    ) {
        let n = chunks_count::<Fr>(chunk_bit_size) as usize;
        let gens = EncryptionGens::<Bls12_381>::new_using_rng(rng);
        let g_i = (0..n)
            .map(|_| <Bls12_381 as Pairing>::G1Affine::rand(rng))
            .collect::<Vec<_>>();
        let delta = Fr::rand(rng);
        let gamma = Fr::rand(rng);
        let g_delta = gens.G.mul_bigint(delta.into_bigint()).into_affine();
        let g_gamma = gens.G.mul_bigint(gamma.into_bigint()).into_affine();
        let (sk, ek, dk) = keygen(rng, chunk_bit_size, &gens, &g_i, &g_delta, &g_gamma).unwrap();
        (gens, g_i, sk, ek, dk)
    }

    pub fn enc_setup_distributed<R: RngCore>(
        chunk_bit_size: u8,
        rng: &mut R,
    ) -> (
        EncryptionGens<Bls12_381>,
        Vec<<Bls12_381 as Pairing>::G1Affine>,
        EncryptionKey<Bls12_381>,
        FEncryptionKeyDest<Bls12_381>,
        SecretKey<<Bls12_381 as Pairing>::ScalarField>,
        SecretKeyST<Bls12_381>,
        SecretKeyST<Bls12_381>,
    ) {
        let n = chunks_count::<Fr>(chunk_bit_size) as usize;
        let gens = EncryptionGens::<Bls12_381>::new_using_rng(rng);
        let g_i = (0..n)
            .map(|_| <Bls12_381 as Pairing>::G1Affine::rand(rng))
            .collect::<Vec<_>>();
        let delta = Fr::rand(rng);
        let gamma = Fr::rand(rng);
        let g_delta = gens.G.mul_bigint(delta.into_bigint()).into_affine();
        let g_gamma = gens.G.mul_bigint(gamma.into_bigint()).into_affine();
        let (sk1, ek1) =
            FkeygenPartial(rng, chunk_bit_size, &gens, &g_i, &g_delta, &g_gamma).unwrap();
        let (sk2, ek2) =
            FkeygenPartial(rng, chunk_bit_size, &gens, &g_i, &g_delta, &g_gamma).unwrap();
        let ek3 = combineEKs(&g_gamma,&[ek1,ek2]);
        let p11 = genP1share(&ek3,&sk1);
        let p12 = genP1share(&ek3,&sk2);
        let p_combined = combineP1share(&[p11,p12]);
        let ek_fin = keygenFinal(&ek3,&p_combined);
        let (skd, ekd) = Fkeygen_dest(rng, &gens).unwrap();
        (gens, g_i, ek_fin, ekd, skd, sk1,sk2)
    }

    pub fn gen_messages<R: RngCore>(
        rng: &mut R,
        count: usize,
        chunk_bit_size: u8,
    ) -> Vec<CHUNK_TYPE> {
        (0..count)
            .map(|_| (u32::rand(rng) & ((1 << chunk_bit_size) - 1)) as CHUNK_TYPE)
            .collect()
    }

    #[test]
    fn encrypt_decrypt() {
        fn check(chunk_bit_size: u8) {
            let mut rng = StdRng::seed_from_u64(0u64);
            let n = chunks_count::<Fr>(chunk_bit_size) as usize;
            // Get random numbers that are of chunk_bit_size at most
            let m = gen_messages(&mut rng, n, chunk_bit_size);
            let (gens, g_i, sk, ek, dk) = enc_setup(chunk_bit_size, &mut rng);

            let prepared_gens = PreparedEncryptionGens::from(gens.clone());
            let prepared_ek = PreparedEncryptionKey::from(ek.clone());
            let prepared_dk = PreparedDecryptionKey::from(dk.clone());

            let start = Instant::now();
            let (ct, _) =
                Encryption::encrypt_decomposed_message(&mut rng, m.clone(), &ek, &g_i).unwrap();
            println!(
                "Time taken to encrypt {}-bit chunks {:?}",
                chunk_bit_size,
                start.elapsed()
            );

            assert_eq!(ct.len(), m.len() + 2);
            let start = Instant::now();
            Encryption::verify_ciphertext_commitment(
                &ct[0],
                &ct[1..m.len() + 1],
                &ct[m.len() + 1],
                ek.clone(),
                gens.clone(),
            )
            .unwrap();
            println!(
                "Time taken to verify commitment of {}-bit chunks {:?}",
                chunk_bit_size,
                start.elapsed()
            );

            let start = Instant::now();
            Encryption::verify_ciphertext_commitment(
                &ct[0],
                &ct[1..m.len() + 1],
                &ct[m.len() + 1],
                prepared_ek,
                prepared_gens.clone(),
            )
            .unwrap();
            println!(
                "Time taken to verify commitment of {}-bit chunks using prepared parameters {:?}",
                chunk_bit_size,
                start.elapsed()
            );

            let start = Instant::now();
            let (m_, _) = Encryption::decrypt_to_chunks(
                &ct[0],
                &ct[1..m.len() + 1],
                &sk,
                dk.clone(),
                &g_i,
                chunk_bit_size,
            )
            .unwrap();
            println!(
                "Time taken to decrypt {}-bit chunks {:?}",
                chunk_bit_size,
                start.elapsed()
            );
            assert_eq!(m_, m);

            let start = Instant::now();
            let (m_, _) = Encryption::decrypt_to_chunks(
                &ct[0],
                &ct[1..m.len() + 1],
                &sk,
                prepared_dk.clone(),
                &g_i,
                chunk_bit_size,
            )
            .unwrap();
            println!(
                "Time taken to decrypt {}-bit chunks using prepared parameters {:?}",
                chunk_bit_size,
                start.elapsed()
            );
            assert_eq!(m_, m);

            let pairing_powers = prepared_dk.pairing_powers(chunk_bit_size, &g_i).unwrap();
            let start = Instant::now();
            let (m_, nu) = Encryption::decrypt_to_chunks_given_pairing_powers(
                &ct[0],
                &ct[1..m.len() + 1],
                &sk,
                prepared_dk.clone(),
                &g_i,
                chunk_bit_size,
                Some(&pairing_powers),
            )
            .unwrap();
            println!(
                "Time taken to decrypt {}-bit chunks using prepared parameters and pairing powers {:?}",
                chunk_bit_size,
                start.elapsed()
            );
            assert_eq!(m_, m);
        }

        check(4);
        check(8);
        check(16);
    }

    #[test]
    fn encrypt_decrypt_distributed() {
        fn check(chunk_bit_size: u8) {
            let mut rng = StdRng::seed_from_u64(0u64);
            let n = chunks_count::<Fr>(chunk_bit_size) as usize;
            // Get random numbers that are of chunk_bit_size at most
            let m = gen_messages(&mut rng, n, chunk_bit_size);
            let (gens, g_i, ek, ekd, skd, sk1, sk2) = enc_setup_distributed(chunk_bit_size, &mut rng);

            let prepared_gens = PreparedEncryptionGens::from(gens.clone());
            let prepared_ek = PreparedEncryptionKey::from(ek.clone());
            
            let start = Instant::now();
            let (ct, _) =
                Encryption::encrypt_decomposed_message(&mut rng, m.clone(), &ek, &g_i).unwrap();
            println!(
                "Time taken to encrypt {}-bit chunks {:?}",
                chunk_bit_size,
                start.elapsed()
            );

            assert_eq!(ct.len(), m.len() + 2);
            let start = Instant::now();
            Encryption::verify_ciphertext_commitment(
                &ct[0],
                &ct[1..m.len() + 1],
                &ct[m.len() + 1],
                ek.clone(),
                gens.clone(),
            )
            .unwrap();
            println!(
                "Time taken to verify commitment of {}-bit chunks {:?}",
                chunk_bit_size,
                start.elapsed()
            );

            let start = Instant::now();
            Encryption::verify_ciphertext_commitment(
                &ct[0],
                &ct[1..m.len() + 1],
                &ct[m.len() + 1],
                prepared_ek,
                prepared_gens.clone(),
            )
            .unwrap();
            println!(
                "Time taken to verify commitment of {}-bit chunks using prepared parameters {:?}",
                chunk_bit_size,
                start.elapsed()
            );

            let (w1,gr1) = 
                Encryption::re_encrypt_share(
                    &mut rng,
                    &gens,
                    &ct[0],
                    &ct[1..m.len() + 1],
                    &ekd,
                    &sk1,
                ).unwrap();
            let (w2,gr2) = 
                Encryption::re_encrypt_share(
                    &mut rng,
                    &gens,
                    &ct[0],
                    &ct[1..m.len() + 1],
                    &ekd,
                    &sk2,
                ).unwrap();
            
            let (wc,grc) = 
                Encryption::<Bls12_381>::combine_re_encrypt_share(
                    &gr1,
                    &gr2,
                    &w1,
                    &w2,
                ).unwrap();

            let ct2 = Encryption::combine_ct_with_w(
                &gens,
                &ct[1..m.len() + 1],
                &grc,
                &wc,
            ).unwrap();

            let start = Instant::now();
            let m_ = 
            Encryption::<Bls12_381>::decrypt_to_chunks_after_re_encryption(
                &ct2[0],
                &ct2[1..m.len() + 1],
                &skd,
                &g_i,
                chunk_bit_size,
            ).unwrap();
            println!(
                "Time taken to decrypt {}-bit chunks {:?}",
                chunk_bit_size,
                start.elapsed()
            );
            assert_eq!(m_, m);
        }

        check(4);
        // check(8);
        // check(16);
    }



    // #[test]
    // fn encrypt_decrypt_timing() {
    //     fn check(chunk_bit_size: u8, count: u8) {
    //         let mut rng = StdRng::seed_from_u64(0u64);
    //         let (gens, g_i, sk, ek, dk) = enc_setup(chunk_bit_size, &mut rng);
    //         let prepared_gens = PreparedEncryptionGens::from(gens.clone());
    //         let prepared_ek = PreparedEncryptionKey::from(ek.clone());
    //         let prepared_dk = PreparedDecryptionKey::from(dk.clone());
    //         let pairing_powers = prepared_dk.pairing_powers(chunk_bit_size, &g_i).unwrap();

    //         let mut total_enc = Duration::default();
    //         let mut total_ver_com = Duration::default();
    //         let mut total_ver_com_prep = Duration::default();
    //         let mut total_dec = Duration::default();
    //         let mut total_dec_prep = Duration::default();
    //         let mut total_dec_prep_powers = Duration::default();
    //         let mut total_ver_dec = Duration::default();
    //         let mut total_ver_dec_prep = Duration::default();

    //         for _ in 0..count {
    //             let m = Fr::rand(&mut rng);

    //             let start = Instant::now();
    //             let (ct, _) = Encryption::encrypt(&mut rng, &m, &ek, &g_i, chunk_bit_size).unwrap();
    //             total_enc += start.elapsed();

    //             let start = Instant::now();
    //             ct.verify_commitment(ek.clone(), gens.clone()).unwrap();
    //             total_ver_com += start.elapsed();

    //             let start = Instant::now();
    //             ct.verify_commitment(prepared_ek.clone(), prepared_gens.clone())
    //                 .unwrap();
    //             total_ver_com_prep += start.elapsed();

    //             let (chunks, nu) = Encryption::decrypt_to_chunks(
    //                 &ct.X_r,
    //                 &ct.enc_chunks,
    //                 &sk,
    //                 dk.clone(),
    //                 &g_i,
    //                 chunk_bit_size,
    //             )
    //             .unwrap();

    //             let decomposed = decompose(&m, chunk_bit_size).unwrap();
    //             assert_eq!(decomposed, chunks);

    //             let start = Instant::now();
    //             let (m_, nu_) = ct.decrypt(&sk, dk.clone(), &g_i, chunk_bit_size).unwrap();
    //             total_dec += start.elapsed();
    //             assert_eq!(m, m_);
    //             assert_eq!(nu, nu_);

    //             let start = Instant::now();
    //             let (m_, nu_) = ct
    //                 .decrypt(&sk, prepared_dk.clone(), &g_i, chunk_bit_size)
    //                 .unwrap();
    //             total_dec_prep += start.elapsed();
    //             assert_eq!(m, m_);
    //             assert_eq!(nu, nu_);

    //             let start = Instant::now();
    //             let (m_, nu_) = ct
    //                 .decrypt_given_pairing_powers(
    //                     &sk,
    //                     prepared_dk.clone(),
    //                     &g_i,
    //                     chunk_bit_size,
    //                     &pairing_powers,
    //                 )
    //                 .unwrap();
    //             total_dec_prep_powers += start.elapsed();
    //             assert_eq!(m, m_);
    //             assert_eq!(nu, nu_);

    //             let start = Instant::now();
    //             ct.verify_decryption(&m, &nu, chunk_bit_size, dk.clone(), &g_i, gens.clone())
    //                 .unwrap();
    //             total_ver_dec += start.elapsed();

    //             let start = Instant::now();
    //             ct.verify_decryption(
    //                 &m,
    //                 &nu,
    //                 chunk_bit_size,
    //                 prepared_dk.clone(),
    //                 &g_i,
    //                 prepared_gens.clone(),
    //             )
    //             .unwrap();
    //             total_ver_dec_prep += start.elapsed();
    //         }

    //         println!(
    //             "Time taken for {} iterations and {}-bit chunk size:",
    //             count, chunk_bit_size
    //         );
    //         println!("Encryption {:?}", total_enc);
    //         println!("Verifying commitment {:?}", total_ver_com);
    //         println!(
    //             "Verifying commitment using prepared {:?}",
    //             total_ver_com_prep
    //         );
    //         println!("Decryption {:?}", total_dec);
    //         println!("Decryption using prepared {:?}", total_dec_prep);
    //         println!(
    //             "Decryption using prepared and pairing powers {:?}",
    //             total_dec_prep_powers
    //         );
    //         println!("Verifying decryption {:?}", total_ver_dec);
    //         println!(
    //             "Verifying decryption using prepared {:?}",
    //             total_ver_dec_prep
    //         );
    //     }
    //     check(4, 10);
    //     check(8, 10);
    //     check(16, 4);
    // }

}
