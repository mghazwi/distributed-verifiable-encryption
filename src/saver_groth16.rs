//! Using SAVER with Groth16
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup, Group, VariableBaseMSM};
use ark_ff::{Field, PrimeField};
use ark_relations::r1cs::{ConstraintSynthesizer, SynthesisError};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{
    ops::{AddAssign},
    rand::{Rng, RngCore},
    vec::Vec,
    UniformRand,
};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

use crate::{
    encryption::Ciphertext,
    serde_utils::*,
    keygen::{EncryptionKey,SecretKey},
    setup::EncryptionGens
};
pub use ark_groth16::{
    prepare_verifying_key, Groth16, PreparedVerifyingKey, Proof, ProvingKey as Groth16ProvingKey,
    VerifyingKey,
};

use crate::error::SaverError;

#[serde_as]
#[derive(
    Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize, Serialize, Deserialize,
)]
pub struct ProvingKey<E: Pairing> {
    /// Groth16's proving key
    #[serde_as(as = "ArkObjectBytes")]
    pub pk: Groth16ProvingKey<E>,
    /// The element `-gamma * G` in `E::G1`.
    #[serde_as(as = "ArkObjectBytes")]
    pub gamma_g1: E::G1Affine,
}

/// These parameters are needed for setting up keys for encryption/decryption
pub fn get_gs_for_encryption<E: Pairing>(vk: &VerifyingKey<E>) -> &[E::G1Affine] {
    &vk.gamma_abc_g1[1..]
}

/// Generate Groth16 SRS
pub fn generate_srs<E: Pairing, R: RngCore, C: ConstraintSynthesizer<E::ScalarField>>(
    circuit: C,
    gens: &EncryptionGens<E>,
    rng: &mut R,
) -> Result<ProvingKey<E>, SaverError> {
    let alpha = E::ScalarField::rand(rng);
    let beta = E::ScalarField::rand(rng);
    let gamma = E::ScalarField::rand(rng);
    let delta = E::ScalarField::rand(rng);

    let g1_generator = gens.G.into_group();
    let neg_gamma_g1 = g1_generator.mul_bigint((-gamma).into_bigint());

    let pk = Groth16::<E>::generate_parameters_with_qap::<C>(
        circuit,
        alpha,
        beta,
        gamma,
        delta,
        g1_generator,
        gens.H.into_group(),
        rng,
    )?;

    Ok(ProvingKey {
        pk,
        gamma_g1: neg_gamma_g1.into_affine(),
    })
}

/// `r` is the randomness used during the encryption
pub fn create_proof<E, C, R>(
    circuit: C,
    r: &E::ScalarField,
    pk: &ProvingKey<E>,
    encryption_key: &EncryptionKey<E>,
    rng: &mut R,
) -> Result<Proof<E>, SaverError>
where
    E: Pairing,
    C: ConstraintSynthesizer<E::ScalarField>,
    R: Rng,
{
    let t = E::ScalarField::rand(rng);
    let s = E::ScalarField::rand(rng);
    let mut proof = Groth16::<E>::create_proof_with_reduction(circuit, &pk.pk, t, s)?;

    // proof.c = proof.c + r * P_2
    let mut c = proof.c.into_group();
    c.add_assign(encryption_key.P_2.mul_bigint(r.into_bigint()));
    proof.c = c.into_affine();

    Ok(proof)
}

pub fn verify_proof<E: Pairing>(
    pvk: &PreparedVerifyingKey<E>,
    proof: &Proof<E>,
    ciphertext: &Ciphertext<E>,
) -> Result<(), SaverError> {
    verify_qap_proof(
        pvk,
        proof.a,
        proof.b,
        proof.c,
        calculate_d(pvk, ciphertext)?,
    )
}

pub fn calculate_d<E: Pairing>(
    pvk: &PreparedVerifyingKey<E>,
    ciphertext: &Ciphertext<E>,
) -> Result<E::G1Affine, SaverError> {
    let mut d = ciphertext.X_r.into_group();
    for c in ciphertext.enc_chunks.iter() {
        d.add_assign(c.into_group())
    }
    d.add_assign(&pvk.vk.gamma_abc_g1[0]);
    Ok(d.into_affine())
}

pub fn verify_qap_proof<E: Pairing>(
    pvk: &PreparedVerifyingKey<E>,
    a: E::G1Affine,
    b: E::G2Affine,
    c: E::G1Affine,
    d: E::G1Affine,
) -> crate::Result<()> {
    let qap = E::multi_miller_loop(
        [a, c, d],
        [
            b.into(),
            pvk.delta_g2_neg_pc.clone(),
            pvk.gamma_g2_neg_pc.clone(),
        ],
    );

    if E::final_exponentiation(qap)
        .ok_or(SynthesisError::UnexpectedIdentity)?
        .0
        != pvk.alpha_g1_beta_g2
    {
        return Err(SaverError::PairingCheckFailed);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        circuit::BitsizeCheckCircuit,
        encryption::{tests::gen_messages, Encryption},
        keygen::{keygen, FkeygenPartial, combineEKs,combineP1share,genP1share,keygenFinal, Fkeygen_dest},
        utils::chunks_count,
    };
    use ark_bls12_381::Bls12_381;
    use ark_std::rand::{prelude::StdRng, SeedableRng};
    use std::time::Instant;

    type Fr = <Bls12_381 as Pairing>::ScalarField;

    #[test]
    fn encrypt_and_snark_verification() {
        fn check(chunk_bit_size: u8) {
            let mut rng = StdRng::seed_from_u64(0u64);
            let gens = EncryptionGens::<Bls12_381>::new_using_rng(&mut rng);
            let n = chunks_count::<Fr>(chunk_bit_size);
            // Get random numbers that are of chunk_bit_size at most
            let msgs = gen_messages(&mut rng, n as usize, chunk_bit_size);
            let msgs_as_field_elems = msgs.iter().map(|m| Fr::from(*m as u64)).collect::<Vec<_>>();

            let circuit = BitsizeCheckCircuit::new(chunk_bit_size, Some(n), None, true);
            let snark_srs = generate_srs::<Bls12_381, _, _>(circuit, &gens, &mut rng).unwrap();

            println!(
                "For chunk_bit_size {}, Snark SRS has compressed size {} and uncompressed size {}",
                chunk_bit_size,
                snark_srs.compressed_size(),
                snark_srs.uncompressed_size()
            );

            let g_i = get_gs_for_encryption(&snark_srs.pk.vk);
            let (sk, ek, dk) = keygen(
                &mut rng,
                chunk_bit_size,
                &gens,
                g_i,
                &snark_srs.pk.delta_g1,
                &snark_srs.gamma_g1,
            )
            .unwrap();

            println!("For chunk_bit_size {}, encryption key has compressed size {} and uncompressed size {}", chunk_bit_size, ek.compressed_size(), ek.uncompressed_size());

            let (ct, r) =
                Encryption::encrypt_decomposed_message(&mut rng, msgs.clone(), &ek, g_i).unwrap();

            let (m_, _) = Encryption::decrypt_to_chunks(
                &ct[0],
                &ct[1..n as usize + 1],
                &sk,
                dk,
                g_i,
                chunk_bit_size,
            )
            .unwrap();

            assert_eq!(m_, msgs);

            let circuit =
                BitsizeCheckCircuit::new(chunk_bit_size, Some(n), Some(msgs_as_field_elems), true);

            let start = Instant::now();
            let proof = create_proof(circuit, &r, &snark_srs, &ek, &mut rng).unwrap();
            println!(
                "Time taken to create Groth16 proof with chunk_bit_size {}: {:?}",
                chunk_bit_size,
                start.elapsed()
            );

            let start = Instant::now();
            Encryption::verify_ciphertext_commitment(
                &ct[0],
                &ct[1..n as usize + 1],
                &ct[n as usize + 1],
                ek.clone(),
                gens.clone(),
            )
            .unwrap();
            let pvk = prepare_verifying_key::<Bls12_381>(&snark_srs.pk.vk);

            let ct = Ciphertext {
                X_r: ct[0],
                enc_chunks: ct[1..n as usize + 1].to_vec(),
                commitment: ct[n as usize + 1],
            };
            verify_proof(&pvk, &proof, &ct).unwrap();
            println!(
                "Time taken to verify Groth16 proof with chunk_bit_size {}: {:?}",
                chunk_bit_size,
                start.elapsed()
            );
        }
        check(4);
        // check(8);
        // check(16);
    }

    #[test]
    fn encrypt_and_ver_distributed() {
        fn check(chunk_bit_size: u8) {
            let mut rng = StdRng::seed_from_u64(0u64);
            let gens = EncryptionGens::<Bls12_381>::new_using_rng(&mut rng);
            let n = chunks_count::<Fr>(chunk_bit_size);
            // Get random numbers that are of chunk_bit_size at most
            let msgs = gen_messages(&mut rng, n as usize, chunk_bit_size);
            let msgs_as_field_elems = msgs.iter().map(|m| Fr::from(*m as u64)).collect::<Vec<_>>();

            let circuit = BitsizeCheckCircuit::new(chunk_bit_size, Some(n), None, true);
            let snark_srs = generate_srs::<Bls12_381, _, _>(circuit, &gens, &mut rng).unwrap();

            println!(
                "For chunk_bit_size {}, Snark SRS has compressed size {} and uncompressed size {}",
                chunk_bit_size,
                snark_srs.compressed_size(),
                snark_srs.uncompressed_size()
            );

            let g_i = get_gs_for_encryption(&snark_srs.pk.vk);
            let (sk1, ek1) =
                FkeygenPartial(&mut rng, chunk_bit_size, &gens, &g_i, &snark_srs.pk.delta_g1,&snark_srs.gamma_g1).unwrap();
            let (sk2, ek2) =
                FkeygenPartial(&mut rng, chunk_bit_size, &gens, &g_i, &snark_srs.pk.delta_g1,&snark_srs.gamma_g1).unwrap();
            let ek3 = combineEKs(&snark_srs.gamma_g1,&[ek1,ek2]);
            let p11 = genP1share(&ek3,&sk1);
            let p12 = genP1share(&ek3,&sk2);
            let p_combined = combineP1share(&[p11,p12]);
            let ek_fin = keygenFinal(&ek3,&p_combined);

            let (skd, ekd) = Fkeygen_dest(&mut rng, &gens).unwrap();

            println!("For chunk_bit_size {}, encryption key has compressed size {} and uncompressed size {}", chunk_bit_size, ek_fin.compressed_size(), ek_fin.uncompressed_size());

            let (ct, r) =
                Encryption::encrypt_decomposed_message(&mut rng, msgs.clone(), &ek_fin, g_i).unwrap();

            let circuit =
                BitsizeCheckCircuit::new(chunk_bit_size, Some(n), Some(msgs_as_field_elems), true);

            let start = Instant::now();
            let proof = create_proof(circuit, &r, &snark_srs, &ek_fin, &mut rng).unwrap();
            println!(
                "Time taken to create Groth16 proof with chunk_bit_size {}: {:?}",
                chunk_bit_size,
                start.elapsed()
            );

            let start = Instant::now();
            Encryption::verify_ciphertext_commitment(
                &ct[0],
                &ct[1..n as usize + 1],
                &ct[n as usize + 1],
                ek_fin.clone(),
                gens.clone(),
            )
            .unwrap();
            let pvk = prepare_verifying_key::<Bls12_381>(&snark_srs.pk.vk);

            let cts = Ciphertext {
                X_r: ct[0],
                enc_chunks: ct[1..n as usize + 1].to_vec(),
                commitment: ct[n as usize + 1],
            };
            verify_proof(&pvk, &proof, &cts).unwrap();
            println!(
                "Time taken to verify Groth16 proof with chunk_bit_size {}: {:?}",
                chunk_bit_size,
                start.elapsed()
            );

            let (w1,gr1) = 
                Encryption::re_encrypt_share(
                    &mut rng,
                    &gens,
                    &ct[0],
                    &ct[1..msgs.len() + 1],
                    &ekd,
                    &sk1,
                ).unwrap();
            let (w2,gr2) = 
                Encryption::re_encrypt_share(
                    &mut rng,
                    &gens,
                    &ct[0],
                    &ct[1..msgs.len() + 1],
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

            let ct_re_enc = Encryption::combine_ct_with_w(
                &gens,
                &ct[1..msgs.len() + 1],
                &grc,
                &wc,
            ).unwrap();

            let m_ = Encryption::<Bls12_381>::decrypt_to_chunks_after_re_encryption(
                &ct_re_enc[0],
                &ct_re_enc[1..n as usize + 1],
                &skd,
                g_i,
                chunk_bit_size*2,
            )
            .unwrap();

            assert_eq!(m_, msgs);

        }
        check(4);
        // check(8);
        // check(16);
    }

    #[test]
    fn encrypt_and_ver_distributed_with_addition() {
        fn check(chunk_bit_size: u8) {
            let mut rng = StdRng::seed_from_u64(0u64);
            let gens = EncryptionGens::<Bls12_381>::new_using_rng(&mut rng);
            let n = chunks_count::<Fr>(chunk_bit_size);
            // Get random numbers that are of chunk_bit_size at most
            let msgs = gen_messages(&mut rng, n as usize, chunk_bit_size);
            let msgs_as_field_elems = msgs.iter().map(|m| Fr::from(*m as u64)).collect::<Vec<_>>();
            // another random msg
            let msgs2 = gen_messages(&mut rng, n as usize, chunk_bit_size);
            let msgs_as_field_elems2 = msgs2.iter().map(|m| Fr::from(*m as u64)).collect::<Vec<_>>();

            let circuit = BitsizeCheckCircuit::new(chunk_bit_size, Some(n), None, true);
            let snark_srs = generate_srs::<Bls12_381, _, _>(circuit, &gens, &mut rng).unwrap();

            println!(
                "For chunk_bit_size {}, Snark SRS has compressed size {} and uncompressed size {}",
                chunk_bit_size,
                snark_srs.compressed_size(),
                snark_srs.uncompressed_size()
            );

            let g_i = get_gs_for_encryption(&snark_srs.pk.vk);
            // party 1 partial key
            let (sk1, ek1) =
                FkeygenPartial(&mut rng, chunk_bit_size, &gens, &g_i, &snark_srs.pk.delta_g1,&snark_srs.gamma_g1).unwrap();
            // party 2 partial key
            let (sk2, ek2) =
                FkeygenPartial(&mut rng, chunk_bit_size, &gens, &g_i, &snark_srs.pk.delta_g1,&snark_srs.gamma_g1).unwrap();
            // combine partial keys (done by both parties)
            let ek3 = combineEKs(&snark_srs.gamma_g1,&[ek1,ek2]);
            // party 1 P1 share
            let p11 = genP1share(&ek3,&sk1);
            // party 2 P1 share
            let p12 = genP1share(&ek3,&sk2);
            // combine P1 (done by both parties)
            let p_combined = combineP1share(&[p11,p12]);
            
            let ek_fin = keygenFinal(&ek3,&p_combined);

            // gen key for destination party (party that can decrypt)
            let (skd, ekd) = Fkeygen_dest(&mut rng, &gens).unwrap();

            println!("For chunk_bit_size {}, encryption key has compressed size {} and uncompressed size {}", chunk_bit_size, ek_fin.compressed_size(), ek_fin.uncompressed_size());
            // encrypt msg 1
            let (ct, r) =
                Encryption::encrypt_decomposed_message(&mut rng, msgs.clone(), &ek_fin, g_i).unwrap();
            // encrypt msg 2
            let (ct2, r2) =
            Encryption::encrypt_decomposed_message(&mut rng, msgs2.clone(), &ek_fin, g_i).unwrap();
            // combine ciphertexts
            let ct_combined = Encryption::<Bls12_381>::add_cts(&ct,&ct2).unwrap();

            let circuit =
                BitsizeCheckCircuit::new(chunk_bit_size, Some(n), Some(msgs_as_field_elems), true);

            let start = Instant::now();
            let proof = create_proof(circuit, &r, &snark_srs, &ek_fin, &mut rng).unwrap();
            println!(
                "Time taken to create Groth16 proof with chunk_bit_size {}: {:?}",
                chunk_bit_size,
                start.elapsed()
            );

            let start = Instant::now();
            Encryption::verify_ciphertext_commitment(
                &ct[0],
                &ct[1..n as usize + 1],
                &ct[n as usize + 1],
                ek_fin.clone(),
                gens.clone(),
            )
            .unwrap();
            let pvk = prepare_verifying_key::<Bls12_381>(&snark_srs.pk.vk);

            let cts = Ciphertext {
                X_r: ct[0],
                enc_chunks: ct[1..n as usize + 1].to_vec(),
                commitment: ct[n as usize + 1],
            };
            verify_proof(&pvk, &proof, &cts).unwrap();
            println!(
                "Time taken to verify Groth16 proof with chunk_bit_size {}: {:?}",
                chunk_bit_size,
                start.elapsed()
            );

            let circuit =
                BitsizeCheckCircuit::new(chunk_bit_size, Some(n), Some(msgs_as_field_elems2), true);

            let start = Instant::now();
            let proof = create_proof(circuit, &r2, &snark_srs, &ek_fin, &mut rng).unwrap();
            println!(
                "Time taken to create Groth16 proof with chunk_bit_size {}: {:?}",
                chunk_bit_size,
                start.elapsed()
            );

            let start = Instant::now();
            Encryption::verify_ciphertext_commitment(
                &ct2[0],
                &ct2[1..n as usize + 1],
                &ct2[n as usize + 1],
                ek_fin.clone(),
                gens.clone(),
            )
            .unwrap();
            let pvk = prepare_verifying_key::<Bls12_381>(&snark_srs.pk.vk);

            let cts = Ciphertext {
                X_r: ct2[0],
                enc_chunks: ct2[1..n as usize + 1].to_vec(),
                commitment: ct2[n as usize + 1],
            };
            verify_proof(&pvk, &proof, &cts).unwrap();
            println!(
                "Time taken to verify Groth16 proof with chunk_bit_size {}: {:?}",
                chunk_bit_size,
                start.elapsed()
            );

            // generate w and g_r for party 1
            let (w1,gr1) = 
            Encryption::re_encrypt_share(
                &mut rng,
                &gens,
                &ct_combined[0],
                &ct_combined[1..msgs.len() + 1],
                &ekd,
                &sk1,
            ).unwrap();
            // generate w and g_r for party 2
            let (w2,gr2) = 
                Encryption::re_encrypt_share(
                    &mut rng,
                    &gens,
                    &ct_combined[0],
                    &ct_combined[1..msgs.len() + 1],
                    &ekd,
                    &sk2,
                ).unwrap();
            // combine w and g_r from both party
            let (wc,grc) = 
                Encryption::<Bls12_381>::combine_re_encrypt_share(
                    &gr1,
                    &gr2,
                    &w1,
                    &w2,
                ).unwrap();
            // perform the re-encryption by combining w and g_r with the ciphertext
            let ct_re_enc = Encryption::combine_ct_with_w(
                &gens,
                &ct_combined[1..msgs.len() + 1],
                &grc,
                &wc,
            ).unwrap();
            // decrypt using destination party's sk
            let m_ = Encryption::<Bls12_381>::decrypt_to_chunks_after_re_encryption(
                &ct_re_enc[0],
                &ct_re_enc[1..n as usize + 1],
                &skd,
                g_i,
                chunk_bit_size*2,
            )
            .unwrap();

            assert_eq!(m_[4], msgs[4]+msgs2[4]);

        }
        check(4);
        // check(8);
        // check(16);
    }

}
